import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.cceval_adapter import load_cceval_example, locate_repo_index, run_one_example
from generation.generator import CompletionGenerator
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
        self.assertIsInstance(result["num_candidates"], int)
        self.assertGreater(result["num_candidates"], 0)
        self.assertIsInstance(result["selected_chunk_ids"], list)
        self.assertEqual(result["completion"], "some completion")
        self.assertEqual(set(result.keys()), {
            "task_id", "repository", "target_file", "num_candidates", "selected_chunk_ids", "completion"
        })


if __name__ == "__main__":
    unittest.main()
