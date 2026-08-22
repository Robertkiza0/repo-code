import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.cceval_adapter import load_cceval_example, locate_repo_index, run_one_example, verify_and_run_task
from generation.generator import CompletionGenerator
from indexer.repo_parser import RepoParser
from retrieval.bm25_retriever import BM25Retriever
from retrieval.symbol_retriever import SymbolRetriever
from retrieval.dependency_retriever import DependencyRetriever
from retrieval.candidate_pipeline import CandidatePipeline
from selection.backends import SelectionBackend
from selection.llm_selector import LLMSelector

SAMPLE_JSONL = Path(__file__).parent.parent / "data" / "cceval" / "samples" / "line_completion_20.jsonl"
SAMPLE_REPO = Path(__file__).parent / "sample_repo"  # module_a.py + pkg/module_b.py


def _fake_clone_and_checkout(owner, repo, commit, dest):
    """Stand-in for a real git clone: just copies our local test fixture repo."""
    shutil.copytree(SAMPLE_REPO, dest)


class LoadCcevalExampleTest(unittest.TestCase):
    def test_extracts_expected_fields_from_real_sample_file(self):
        example = load_cceval_example(str(SAMPLE_JSONL), index=0)
        self.assertEqual(set(example.keys()), {"task_id", "repository", "file", "prompt", "groundtruth"})
        self.assertTrue(example["task_id"])
        self.assertTrue(example["repository"])
        self.assertTrue(example["file"])
        self.assertIsInstance(example["prompt"], str)
        self.assertIsInstance(example["groundtruth"], str)

    def test_out_of_range_index_raises(self):
        with self.assertRaises(IndexError):
            load_cceval_example(str(SAMPLE_JSONL), index=9999)


class LocateRepoIndexTest(unittest.TestCase):
    def test_clones_and_indexes_when_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repos_dir = Path(tmp_dir) / "repos"
            index_dir = Path(tmp_dir) / "indexes"

            with patch(
                "evaluation.cceval_adapter.resolve_owner_repo", return_value=("someowner", "somerepo", "abc1234")
            ) as mock_resolve, patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ) as mock_clone:
                chunks = locate_repo_index("someowner-somerepo-abc1234", repos_dir=str(repos_dir), index_dir=str(index_dir))

            mock_resolve.assert_called_once_with("someowner-somerepo-abc1234")
            mock_clone.assert_called_once()
            self.assertGreater(len(chunks), 0)
            self.assertTrue((index_dir / "someowner-somerepo-abc1234.json").exists())

    def test_uses_cache_on_second_call_without_reclonning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repos_dir = Path(tmp_dir) / "repos"
            index_dir = Path(tmp_dir) / "indexes"

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                locate_repo_index("o-r-c", repos_dir=str(repos_dir), index_dir=str(index_dir))

            with patch("evaluation.cceval_adapter.resolve_owner_repo") as mock_resolve_2, patch(
                "evaluation.cceval_adapter.clone_and_checkout"
            ) as mock_clone_2:
                chunks = locate_repo_index("o-r-c", repos_dir=str(repos_dir), index_dir=str(index_dir))

            mock_resolve_2.assert_not_called()
            mock_clone_2.assert_not_called()
            self.assertGreater(len(chunks), 0)


class FakeSelectionBackend(SelectionBackend):
    def __init__(self, n=1):
        self.n = n

    def generate(self, prompt: str) -> str:
        ids = [line.split("chunk_id: ", 1)[1] for line in prompt.splitlines() if line.startswith("chunk_id: ")]
        return json.dumps({"selected_chunk_ids": ids[: self.n]})


class RunOneExampleTest(unittest.TestCase):
    def test_end_to_end_never_passes_groundtruth_or_right_context_downstream(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repos_dir = Path(tmp_dir) / "repos"
            index_dir = Path(tmp_dir) / "indexes"

            # Load the real example first so we know what groundtruth actually is.
            example = load_cceval_example(str(SAMPLE_JSONL), index=0)

            with patch(
                "evaluation.cceval_adapter.resolve_owner_repo", return_value=("turboderp", "exllama", "a544085")
            ), patch("evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout):
                # Peek at the chunks so we can build a selector/generator that spy on their inputs.
                chunks = locate_repo_index(example["repository"], repos_dir=str(repos_dir), index_dir=str(index_dir))

                selector = LLMSelector(chunks, backend=FakeSelectionBackend(n=1))

                generation_backend = MagicMock()
                generation_backend.generate.return_value = "some completion"
                generator = CompletionGenerator(chunks, backend=generation_backend)

                result = run_one_example(
                    str(SAMPLE_JSONL),
                    index=0,
                    selector=selector,
                    generator=generator,
                    repos_dir=str(repos_dir),
                    index_dir=str(index_dir),
                )

        # The generation backend only ever sees the assembled context prompt --
        # confirm groundtruth text never ended up anywhere in it.
        sent_prompt = generation_backend.generate.call_args[0][0]
        self.assertNotIn(example["groundtruth"], sent_prompt)

        self.assertEqual(result["task_id"], example["task_id"])
        self.assertEqual(result["repository"], example["repository"])
        self.assertEqual(result["target_file"], example["file"])
        self.assertEqual(result["prompt"], example["prompt"])
        self.assertEqual(result["groundtruth"], example["groundtruth"])
        self.assertIsInstance(result["num_candidates"], int)
        self.assertGreater(result["num_candidates"], 0)
        self.assertEqual(len(result["candidates"]), result["num_candidates"])
        self.assertIsInstance(result["selected_chunk_ids"], list)
        self.assertEqual(result["completion"], "some completion")
        self.assertEqual(set(result.keys()), {
            "task_id", "repository", "target_file", "prompt", "groundtruth",
            "num_candidates", "candidates", "selected_chunk_ids", "rejected_hallucinated_ids",
            "raw_response", "parse_status", "selection_parse_error", "completion",
        })


class PrintResultTest(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "chunk_id": "a.py::foo::function:1-2",
                "file_path": "a.py",
                "name": "foo",
                "source_code": "def foo():\n    return 1",
            },
            {
                "chunk_id": "b.py::bar::function:1-3",
                "file_path": "b.py",
                "name": "bar",
                "source_code": "def bar():\n    x = 1\n    return x",
            },
        ]
        self.result = {
            "task_id": "project_cc_python/1",
            "repository": "owner-repo-abc1234",
            "target_file": "c.py",
            "prompt": "line1\nline2\nresult = foo(",
            "groundtruth": ")",
            "num_candidates": 2,
            "candidates": [
                {"chunk_id": "a.py::foo::function:1-2", "file_path": "a.py", "name": "foo", "sources": ["bm25"], "scores": {}},
                {"chunk_id": "b.py::bar::function:1-3", "file_path": "b.py", "name": "bar", "sources": ["dependency"], "scores": {}},
            ],
            "selected_chunk_ids": ["a.py::foo::function:1-2"],
            "completion": ")",
        }

    def test_includes_all_requested_sections(self):
        from io import StringIO
        from evaluation.cceval_adapter import print_result

        buf = StringIO()
        with patch("sys.stdout", buf):
            print_result(self.chunks, self.result)
        output = buf.getvalue()

        self.assertIn("project_cc_python/1", output)
        self.assertIn("owner-repo-abc1234", output)
        self.assertIn("c.py", output)
        self.assertIn("result = foo(", output)  # incomplete code (prompt tail)
        self.assertIn("a.py::foo::function:1-2", output)  # candidate chunk_id
        self.assertIn("b.py::bar::function:1-3", output)  # candidate chunk_id
        self.assertIn("def foo():", output)  # candidate preview source
        self.assertIn("def bar():", output)  # candidate preview source
        self.assertIn(")", output)  # completion / groundtruth (both are ")")

    def test_selected_chunk_gets_full_source_not_just_preview(self):
        from io import StringIO
        from evaluation.cceval_adapter import print_result

        long_source = "def foo():\n" + "\n".join(f"    line{i}" for i in range(10))
        self.chunks[0]["source_code"] = long_source

        buf = StringIO()
        with patch("sys.stdout", buf):
            print_result(self.chunks, self.result, preview_lines=2)
        output = buf.getvalue()

        self.assertIn("line9", output)  # full source for the *selected* chunk reaches the end


class FindExampleIndexByTaskIdTest(unittest.TestCase):
    def test_finds_the_real_task_at_index_zero(self):
        from evaluation.cceval_adapter import find_example_index_by_task_id

        index = find_example_index_by_task_id(str(SAMPLE_JSONL), "project_cc_python/62")
        self.assertEqual(index, 0)

    def test_unknown_task_id_raises(self):
        from evaluation.cceval_adapter import find_example_index_by_task_id

        with self.assertRaises(LookupError):
            find_example_index_by_task_id(str(SAMPLE_JSONL), "not-a-real-task-id")


class LooksLikeGenerationArtifactTest(unittest.TestCase):
    def test_plain_generated_text_is_not_flagged(self):
        from evaluation.cceval_adapter import _looks_like_generation_artifact

        warning = _looks_like_generation_artifact("return self.value + 1", [], "def foo():")
        self.assertIsNone(warning)

    def test_empty_completion_is_not_flagged(self):
        from evaluation.cceval_adapter import _looks_like_generation_artifact

        self.assertIsNone(_looks_like_generation_artifact("", [], "def foo():"))

    def test_completion_identical_to_prompt_is_flagged(self):
        from evaluation.cceval_adapter import _looks_like_generation_artifact

        warning = _looks_like_generation_artifact("def foo():", [], "def foo():")
        self.assertIsNotNone(warning)
        self.assertIn("prompt", warning)

    def test_completion_identical_to_a_chunk_source_is_flagged(self):
        from evaluation.cceval_adapter import _looks_like_generation_artifact

        chunks = [{"chunk_id": "a.py::foo::function:1-2", "source_code": "def foo():\n    return 1"}]
        warning = _looks_like_generation_artifact("def foo():\n    return 1", chunks, "something else")
        self.assertIsNotNone(warning)
        self.assertIn("a.py::foo::function:1-2", warning)

    def test_object_repr_is_flagged(self):
        from evaluation.cceval_adapter import _looks_like_generation_artifact

        warning = _looks_like_generation_artifact("<Tensor object at 0x7f1234>", [], "prompt")
        self.assertIsNotNone(warning)


class VerifyAndRunTaskTest(unittest.TestCase):
    def test_isolation_flags_and_completion_check_on_a_clean_run(self):
        chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
        pipeline = CandidatePipeline(BM25Retriever(chunks), SymbolRetriever(chunks), DependencyRetriever(chunks))
        candidates = pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py")
        chosen_id = candidates[0]["chunk_id"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Build a jsonl with a real-shaped task pointing at our test fixture repo.
            jsonl_path = Path(tmp_dir) / "one_task.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "prompt": "result = Greeter(",
                        "groundtruth": "default_greeting='Hi').greet('world')",
                        "right_context": "\nprint(result)\n",
                        "metadata": {
                            "task_id": "project_cc_python/62",
                            "repository": "someowner-somerepo-abc1234",
                            "file": "pkg/module_b.py",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            selection_backend = FakeSelectionBackend(n=1)
            selector = LLMSelector(chunks, backend=selection_backend)

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "a plausible generated line"
            generator = CompletionGenerator(chunks, backend=generation_backend)

            with patch(
                "evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")
            ), patch("evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout):
                result = verify_and_run_task(
                    "project_cc_python/62",
                    jsonl_path=str(jsonl_path),
                    selector=selector,
                    generator=generator,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

        self.assertEqual(
            result["isolation_flags"],
            {
                "groundtruth_used_for_retrieval": False,
                "groundtruth_used_for_selection": False,
                "groundtruth_used_for_generation": False,
                "groundtruth_used_for_evaluation": True,
            },
        )
        self.assertIsNone(result["completion_artifact_warning"])
        self.assertEqual(result["completion"], "a plausible generated line")
        self.assertEqual(result["groundtruth"], "default_greeting='Hi').greet('world')")

        # the model's actual raw text response is captured, not just the
        # parsed ids -- needed to tell an intentional empty selection apart
        # from a parsing failure.
        self.assertIn("selected_chunk_ids", result["raw_response"])
        self.assertEqual(result["rejected_hallucinated_ids"], [])

        # groundtruth/right_context text must never have reached the generation prompt.
        sent_prompt = generation_backend.generate.call_args[0][0]
        self.assertNotIn("default_greeting='Hi').greet('world')", sent_prompt)
        self.assertNotIn("print(result)", sent_prompt)

    def test_flags_a_completion_that_leaked_raw_chunk_source(self):
        chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]

        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "one_task.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "prompt": "result = Greeter(",
                        "groundtruth": "whatever",
                        "right_context": "",
                        "metadata": {
                            "task_id": "project_cc_python/62",
                            "repository": "someowner-somerepo-abc1234",
                            "file": "pkg/module_b.py",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            selector = LLMSelector(chunks, backend=FakeSelectionBackend(n=1))

            leaked_source = next(c["source_code"] for c in chunks if c["name"] == "Greeter")
            generation_backend = MagicMock()
            generation_backend.generate.return_value = leaked_source
            generator = CompletionGenerator(chunks, backend=generation_backend)

            with patch(
                "evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")
            ), patch("evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout):
                result = verify_and_run_task(
                    "project_cc_python/62",
                    jsonl_path=str(jsonl_path),
                    selector=selector,
                    generator=generator,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

        self.assertIsNotNone(result["completion_artifact_warning"])
        self.assertIn("source_code", result["completion_artifact_warning"])


class PrintSideBySideTest(unittest.TestCase):
    def test_shows_completion_and_groundtruth_with_match_flag(self):
        from io import StringIO
        from evaluation.cceval_adapter import print_side_by_side

        result = {"completion": "x + 1", "groundtruth": "x + 1"}
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_side_by_side(result)
        output = buf.getvalue()

        self.assertIn("x + 1", output)
        self.assertIn("exact match: True", output)

    def test_mismatch_reported(self):
        from io import StringIO
        from evaluation.cceval_adapter import print_side_by_side

        result = {"completion": "x + 1", "groundtruth": "x + 2"}
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_side_by_side(result)
        self.assertIn("exact match: False", buf.getvalue())


class AssignCandidateLabelsTest(unittest.TestCase):
    def test_labels_assigned_in_nomination_order(self):
        from evaluation.cceval_adapter import assign_candidate_labels

        candidates = [
            {"chunk_id": "a.py::foo::function:1-2"},
            {"chunk_id": "b.py::bar::function:1-2"},
            {"chunk_id": "c.py::baz::function:1-2"},
        ]
        labels = assign_candidate_labels(candidates)
        self.assertEqual(
            labels,
            {
                "a.py::foo::function:1-2": "C1",
                "b.py::bar::function:1-2": "C2",
                "c.py::baz::function:1-2": "C3",
            },
        )

    def test_empty_candidates_gives_empty_mapping(self):
        from evaluation.cceval_adapter import assign_candidate_labels

        self.assertEqual(assign_candidate_labels([]), {})


class PrintExperimentLogTest(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"chunk_id": "a.py::foo::function:1-2", "file_path": "a.py", "name": "foo", "source_code": "def foo():\n    return 1"},
            {"chunk_id": "b.py::bar::function:1-3", "file_path": "b.py", "name": "bar", "source_code": "def bar():\n    return 2"},
        ]
        self.result = {
            "task_id": "project_cc_python/62",
            "repository": "owner-repo-abc1234",
            "target_file": "c.py",
            "num_candidates": 2,
            "candidates": [
                {"chunk_id": "a.py::foo::function:1-2", "file_path": "a.py", "name": "foo", "sources": ["bm25"], "scores": {}},
                {"chunk_id": "b.py::bar::function:1-3", "file_path": "b.py", "name": "bar", "sources": ["dependency"], "scores": {}},
            ],
            "selected_chunk_ids": ["a.py::foo::function:1-2"],
            "completion": "return 1",
            "groundtruth": "return 1",
        }

    def test_uses_short_labels_not_raw_chunk_ids_for_candidates_and_selection(self):
        from io import StringIO
        from evaluation.cceval_adapter import print_experiment_log

        buf = StringIO()
        with patch("sys.stdout", buf):
            print_experiment_log(self.chunks, self.result)
        output = buf.getvalue()

        self.assertIn("C1", output)
        self.assertIn("C2", output)
        self.assertIn("Qwen selected: C1", output)
        # the raw chunk_id text must not leak into this concise view
        self.assertNotIn("a.py::foo::function:1-2", output)
        self.assertNotIn("b.py::bar::function:1-3", output)

    def test_shows_file_symbol_and_source_per_candidate(self):
        from io import StringIO
        from evaluation.cceval_adapter import print_experiment_log

        buf = StringIO()
        with patch("sys.stdout", buf):
            print_experiment_log(self.chunks, self.result)
        output = buf.getvalue()

        self.assertIn("a.py::foo", output)
        self.assertIn("b.py::bar", output)
        self.assertIn("['bm25']", output)
        self.assertIn("['dependency']", output)

    def test_does_not_dump_full_selected_source_code(self):
        from io import StringIO
        from evaluation.cceval_adapter import print_experiment_log

        self.chunks[0]["source_code"] = "def foo():\n    " + "\n    ".join(f"line{i}" for i in range(20))
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_experiment_log(self.chunks, self.result)
        output = buf.getvalue()

        self.assertNotIn("line19", output)  # concise: no full source dump for selections

    def test_original_result_dict_is_untouched(self):
        from evaluation.cceval_adapter import print_experiment_log

        original = json.loads(json.dumps(self.result))  # deep copy
        with patch("sys.stdout", MagicMock()):
            print_experiment_log(self.chunks, self.result)
        self.assertEqual(self.result, original)


if __name__ == "__main__":
    unittest.main()
