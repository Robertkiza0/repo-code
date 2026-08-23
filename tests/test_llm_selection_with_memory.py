import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.llm_selection_with_memory import (
    print_llm_selection_with_memory_isolation_flags,
    print_llm_selection_with_memory_summary_table,
    print_llm_selection_with_memory_task_table,
    run_llm_selection_with_memory_experiment,
    run_one_example_llm_selection_with_memory,
    summarize_llm_selection_with_memory,
)
from generation.generator import CompletionGenerator
from selection.backends import SelectionBackend
from selection.llm_selector import LLMSelector

SAMPLE_REPO = Path(__file__).parent / "sample_repo"


def _fake_clone_and_checkout(owner, repo, commit, dest):
    shutil.copytree(SAMPLE_REPO, dest)


def _write_jsonl(path: Path, tasks: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")


def _make_task(task_id: str, groundtruth: str = "whatever") -> dict:
    return {
        "prompt": "result = Greeter(",
        "groundtruth": groundtruth,
        "right_context": "\nprint(result)\n",
        "metadata": {
            "task_id": task_id,
            "repository": "someowner-somerepo-abc1234",
            "file": "pkg/module_b.py",
        },
    }


class FixedSelectionBackend(SelectionBackend):
    """Always selects the first `n` candidate labels shown in the prompt."""

    def __init__(self, n: int = 1):
        self.n = n

    def generate(self, prompt: str) -> str:
        labels = re.findall(r"^C\d+$", prompt, re.MULTILINE)
        return json.dumps({"selected_chunk_ids": labels[: self.n]})


class RunOneExampleTest(unittest.TestCase):
    def test_memory_is_built_and_passed_to_selection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1", "x")])

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "x"

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                selector = LLMSelector(chunks, backend=FixedSelectionBackend(n=1))
                generator = CompletionGenerator(chunks, backend=generation_backend)

                result = run_one_example_llm_selection_with_memory(
                    selector,
                    generator,
                    jsonl_path=str(jsonl_path),
                    index=0,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            self.assertEqual(result["task_id"], "t1")
            self.assertGreater(len(result["memory_symbols_found"]), 0)
            self.assertIn("Greeter", result["memory_symbols_found"])
            self.assertEqual(len(result["selected_chunk_ids"]), 1)

    def test_max_selected_caps_selection_even_if_model_returns_more(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1", "x")])

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "x"

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                selector = LLMSelector(chunks, backend=FixedSelectionBackend(n=8))  # tries to select all 8
                generator = CompletionGenerator(chunks, backend=generation_backend)

                result = run_one_example_llm_selection_with_memory(
                    selector,
                    generator,
                    jsonl_path=str(jsonl_path),
                    index=0,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                    max_selected=4,
                )

            self.assertEqual(len(result["selected_chunk_ids"]), 4)

    def test_memory_cache_is_reused_across_tasks_in_the_same_repository(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1", "x"), _make_task("t2", "y")])

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "x"

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ), patch(
                "evaluation.llm_selection_with_memory.build_repository_memory", wraps=__import__(
                    "memory.repository_memory", fromlist=["build_repository_memory"]
                ).build_repository_memory
            ) as build_spy:
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                selector = LLMSelector(chunks, backend=FixedSelectionBackend(n=1))
                generator = CompletionGenerator(chunks, backend=generation_backend)
                cache = {}

                run_one_example_llm_selection_with_memory(
                    selector, generator, jsonl_path=str(jsonl_path), index=0,
                    repos_dir=str(Path(tmp_dir) / "repos"), index_dir=str(Path(tmp_dir) / "indexes"),
                    memory_cache=cache,
                )
                run_one_example_llm_selection_with_memory(
                    selector, generator, jsonl_path=str(jsonl_path), index=1,
                    repos_dir=str(Path(tmp_dir) / "repos"), index_dir=str(Path(tmp_dir) / "indexes"),
                    memory_cache=cache,
                )

            build_spy.assert_called_once()

    def test_groundtruth_never_reaches_selection_or_generation_backend(self):
        secret = "THIS_MUST_NOT_LEAK"
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1", secret)])

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "some completion"

            captured_prompts = []

            class CapturingSelectionBackend(SelectionBackend):
                def generate(self, prompt: str) -> str:
                    captured_prompts.append(prompt)
                    return json.dumps({"selected_chunk_ids": []})

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                selector = LLMSelector(chunks, backend=CapturingSelectionBackend())
                generator = CompletionGenerator(chunks, backend=generation_backend)

                run_one_example_llm_selection_with_memory(
                    selector, generator, jsonl_path=str(jsonl_path), index=0,
                    repos_dir=str(Path(tmp_dir) / "repos"), index_dir=str(Path(tmp_dir) / "indexes"),
                )

            self.assertTrue(all(secret not in p for p in captured_prompts))
            context_arg = generation_backend.generate.call_args[0][0]
            self.assertNotIn(secret, context_arg)


class RunExperimentTest(unittest.TestCase):
    def test_all_tasks_succeed_and_saves_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1", "x"), _make_task("t2", "y")])

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "x"

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                selector = LLMSelector(chunks, backend=FixedSelectionBackend(n=1))
                generator = CompletionGenerator(chunks, backend=generation_backend)

                outcome = run_llm_selection_with_memory_experiment(
                    selector, generator, n_tasks=2, jsonl_path=str(jsonl_path),
                    results_dir=str(Path(tmp_dir) / "results"),
                    repos_dir=str(Path(tmp_dir) / "repos"), index_dir=str(Path(tmp_dir) / "indexes"),
                )

            results = outcome["results"]
            summary = outcome["summary"]
            self.assertEqual(len(results), 2)
            self.assertEqual(summary["num_successful"], 2)
            self.assertEqual(summary["exact_match_rate"], 0.5)
            self.assertIn("memory_assisted_selections", summary)
            self.assertIn("avg_memory_relationships_found", summary)
            self.assertIn("selection_count_distribution", summary)
            for r in results:
                self.assertIsNone(r["error"])
                self.assertGreater(r["memory_symbols_found_count"], 0)

            results_path = Path(tmp_dir) / "results" / "cceval_2_llm_selection_with_memory.jsonl"
            summary_path = Path(tmp_dir) / "results" / "cceval_2_llm_selection_with_memory_summary.json"
            self.assertTrue(results_path.exists())
            self.assertTrue(summary_path.exists())

    def test_one_task_failing_does_not_stop_the_others(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1"), _make_task("t2"), _make_task("t3")])

            generation_backend = MagicMock()
            call_count = {"n": 0}

            def _generate(context):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise RuntimeError("simulated backend failure")
                return "some completion"

            generation_backend.generate.side_effect = _generate

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                selector = LLMSelector(chunks, backend=FixedSelectionBackend(n=1))
                generator = CompletionGenerator(chunks, backend=generation_backend)

                outcome = run_llm_selection_with_memory_experiment(
                    selector, generator, n_tasks=3, jsonl_path=str(jsonl_path),
                    results_dir=str(Path(tmp_dir) / "results"),
                    repos_dir=str(Path(tmp_dir) / "repos"), index_dir=str(Path(tmp_dir) / "indexes"),
                )

            results = outcome["results"]
            self.assertEqual(len(results), 3)
            self.assertIsNone(results[0]["error"])
            self.assertIn("simulated backend failure", results[1]["error"])
            self.assertIsNone(results[2]["error"])

    def test_does_not_touch_llm_selection_result_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1")])
            results_dir = Path(tmp_dir) / "results"

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "some completion"

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser
                from evaluation.experiment import run_experiment_v2

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                selector1 = LLMSelector(chunks, backend=FixedSelectionBackend(n=1))
                generator1 = CompletionGenerator(chunks, backend=generation_backend)
                run_experiment_v2(
                    n_tasks=1, jsonl_path=str(jsonl_path), selector=selector1, generator=generator1,
                    results_dir=str(results_dir), repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )
                v2_results_path = results_dir / "cceval_1_results_v2.jsonl"
                v2_mtime_before = v2_results_path.stat().st_mtime

                selector2 = LLMSelector(chunks, backend=FixedSelectionBackend(n=1))
                generator2 = CompletionGenerator(chunks, backend=generation_backend)
                run_llm_selection_with_memory_experiment(
                    selector2, generator2, n_tasks=1, jsonl_path=str(jsonl_path),
                    results_dir=str(results_dir), repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            self.assertEqual(v2_results_path.stat().st_mtime, v2_mtime_before)
            self.assertTrue((results_dir / "cceval_1_llm_selection_with_memory.jsonl").exists())


class SummarizeTest(unittest.TestCase):
    def test_counts_memory_assisted_selections(self):
        results = [
            {"task_id": "t1", "error": None, "candidate_count": 8, "selected_count": 1,
             "memory_relationships_found_count": 5, "memory_augmented_candidate_count": 2,
             "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 1.0},
            {"task_id": "t2", "error": None, "candidate_count": 8, "selected_count": 0,
             "memory_relationships_found_count": 0, "memory_augmented_candidate_count": 0,
             "exact_match": False, "ES": 0.5, "ID-F1": 0.5, "generation_time": 1.0},
        ]
        summary = summarize_llm_selection_with_memory(results)
        self.assertEqual(summary["memory_assisted_selections"], 1)
        self.assertAlmostEqual(summary["avg_memory_relationships_found"], 2.5)
        self.assertEqual(summary["selection_count_distribution"], {"0": 1, "1": 1, "2": 0, ">=3": 0})


class PrintFunctionsTest(unittest.TestCase):
    def test_print_summary_table_handles_none_averages(self):
        from io import StringIO

        summary = {
            "total_tasks": 0, "num_successful": 0, "num_failed": 0, "failed_task_ids": [],
            "avg_candidate_count": None, "avg_selected_count": None, "exact_match_rate": None,
            "avg_ES": None, "avg_ID_F1": None, "avg_generation_time": None,
            "memory_assisted_selections": 0, "avg_memory_relationships_found": None,
            "selection_count_distribution": {"0": 0, "1": 0, "2": 0, ">=3": 0},
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_llm_selection_with_memory_summary_table(summary)
        self.assertIn("n/a", buf.getvalue())

    def test_print_task_table_shows_errors_inline(self):
        from io import StringIO

        results = [
            {"task_id": "t1", "error": None, "candidate_count": 8, "selected_count": 1,
             "memory_relationships_found_count": 5, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 0.5},
            {"task_id": "t2", "error": "RuntimeError: boom"},
        ]
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_llm_selection_with_memory_task_table(results)
        output = buf.getvalue()
        self.assertIn("t1", output)
        self.assertIn("ERROR: RuntimeError: boom", output)

    def test_print_isolation_flags(self):
        from io import StringIO

        buf = StringIO()
        with patch("sys.stdout", buf):
            print_llm_selection_with_memory_isolation_flags()
        output = buf.getvalue()
        self.assertIn("groundtruth_used_for_selection = False", output)


if __name__ == "__main__":
    unittest.main()
