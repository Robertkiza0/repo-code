import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.random2_selection import (
    RANDOM_N,
    RANDOM_SEED,
    print_random2_summary_table,
    print_random2_task_table,
    run_one_example_random2,
    run_random2_experiment,
    summarize_random2,
)
from generation.generator import CompletionGenerator

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


class RunOneExampleRandom2Test(unittest.TestCase):
    def _run(self, tmp_dir, generation_backend=None):
        jsonl_path = Path(tmp_dir) / "tasks.jsonl"
        _write_jsonl(jsonl_path, [_make_task("t1", "x")])

        if generation_backend is None:
            generation_backend = MagicMock()
            generation_backend.generate.return_value = "x"

        with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
            "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
        ):
            from indexer.repo_parser import RepoParser

            chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
            generator = CompletionGenerator(chunks, backend=generation_backend)
            return run_one_example_random2(
                generator,
                jsonl_path=str(jsonl_path),
                index=0,
                repos_dir=str(Path(tmp_dir) / "repos"),
                index_dir=str(Path(tmp_dir) / "indexes"),
            )

    def test_selects_exactly_two_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self._run(tmp_dir)
        self.assertGreater(result["num_candidates"], RANDOM_N)
        self.assertEqual(len(result["selected_chunk_ids"]), RANDOM_N)
        candidate_ids = {c["chunk_id"] for c in result["candidates"]}
        for chunk_id in result["selected_chunk_ids"]:
            self.assertIn(chunk_id, candidate_ids)

    def test_same_seed_gives_the_same_selection_every_time(self):
        with tempfile.TemporaryDirectory() as tmp_dir1:
            result1 = self._run(tmp_dir1)
        with tempfile.TemporaryDirectory() as tmp_dir2:
            result2 = self._run(tmp_dir2)
        self.assertEqual(result1["selected_chunk_ids"], result2["selected_chunk_ids"])

    def test_selected_ids_preserve_original_pool_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self._run(tmp_dir)
        pool_order = [c["chunk_id"] for c in result["candidates"]]
        pool_positions = [pool_order.index(cid) for cid in result["selected_chunk_ids"]]
        self.assertEqual(pool_positions, sorted(pool_positions))

    def test_fewer_than_two_candidates_selects_all_available(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1", "x")])

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "x"

            one_candidate = [{"chunk_id": "only_one", "file_path": "a.py", "name": "f", "sources": ["bm25"], "scores": {"bm25": 1.0}}]

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ), patch("retrieval.candidate_pipeline.CandidatePipeline.nominate", return_value=one_candidate):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                generator = CompletionGenerator(chunks, backend=generation_backend)
                result = run_one_example_random2(
                    generator,
                    jsonl_path=str(jsonl_path),
                    index=0,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            self.assertEqual(result["num_candidates"], 1)
            self.assertEqual(result["selected_chunk_ids"], ["only_one"])

    def test_zero_candidates_selects_none_and_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1", "x")])

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "x"

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ), patch("retrieval.candidate_pipeline.CandidatePipeline.nominate", return_value=[]):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                generator = CompletionGenerator(chunks, backend=generation_backend)
                result = run_one_example_random2(
                    generator,
                    jsonl_path=str(jsonl_path),
                    index=0,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            self.assertEqual(result["num_candidates"], 0)
            self.assertEqual(result["selected_chunk_ids"], [])

    def test_groundtruth_never_reaches_generation_backend(self):
        secret = "THIS_MUST_NOT_LEAK"
        generation_backend = MagicMock()
        generation_backend.generate.return_value = "some completion"
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1", secret)])
            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                generator = CompletionGenerator(chunks, backend=generation_backend)
                run_one_example_random2(
                    generator,
                    jsonl_path=str(jsonl_path),
                    index=0,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )
            context_arg = generation_backend.generate.call_args[0][0]
            self.assertNotIn(secret, context_arg)


class RunRandom2ExperimentTest(unittest.TestCase):
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
                generator = CompletionGenerator(chunks, backend=generation_backend)

                outcome = run_random2_experiment(
                    generator,
                    n_tasks=2,
                    jsonl_path=str(jsonl_path),
                    results_dir=str(Path(tmp_dir) / "results"),
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            results = outcome["results"]
            summary = outcome["summary"]
            self.assertEqual(len(results), 2)
            self.assertEqual(summary["num_successful"], 2)
            for r in results:
                self.assertIsNone(r["error"])
                self.assertEqual(r["selected_count"], RANDOM_N)

            results_path = Path(tmp_dir) / "results" / "cceval_2_random2_selection.jsonl"
            summary_path = Path(tmp_dir) / "results" / "cceval_2_random2_selection_summary.json"
            self.assertTrue(results_path.exists())
            self.assertTrue(summary_path.exists())

    def test_uses_the_same_seed_for_every_task(self):
        # Two different tasks resolving to the same candidate pool (same
        # repository/prompt/file) must get the same random selection --
        # confirms the seed isn't advancing across tasks in the batch.
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
                generator = CompletionGenerator(chunks, backend=generation_backend)

                outcome = run_random2_experiment(
                    generator,
                    n_tasks=2,
                    jsonl_path=str(jsonl_path),
                    results_dir=str(Path(tmp_dir) / "results"),
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            results = outcome["results"]
            self.assertEqual(results[0]["selected_candidate_ids"], results[1]["selected_candidate_ids"])

    def test_default_seed_constant_is_used(self):
        self.assertEqual(RANDOM_SEED, 42)


class SummarizeRandom2Test(unittest.TestCase):
    def test_averages_over_successful_tasks_only(self):
        results = [
            {"task_id": "t1", "error": None, "candidate_count": 10, "selected_count": 2, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 1.0},
            {"task_id": "t2", "error": "boom", "candidate_count": None, "selected_count": None, "exact_match": None, "ES": None, "ID-F1": None, "generation_time": None},
        ]
        summary = summarize_random2(results)
        self.assertEqual(summary["num_successful"], 1)
        self.assertAlmostEqual(summary["avg_selected_count"], 2.0)


class PrintFunctionsTest(unittest.TestCase):
    def test_print_random2_summary_table_handles_none_averages(self):
        from io import StringIO

        summary = {
            "total_tasks": 0, "num_successful": 0, "num_failed": 0, "failed_task_ids": [],
            "avg_candidate_count": None, "avg_selected_count": None, "exact_match_rate": None,
            "avg_ES": None, "avg_ID_F1": None, "avg_generation_time": None,
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_random2_summary_table(summary)
        self.assertIn("n/a", buf.getvalue())

    def test_print_random2_task_table_shows_errors_inline(self):
        from io import StringIO

        results = [
            {"task_id": "t1", "error": None, "candidate_count": 5, "selected_count": 2, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 0.5},
            {"task_id": "t2", "error": "RuntimeError: boom"},
        ]
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_random2_task_table(results)
        output = buf.getvalue()
        self.assertIn("t1", output)
        self.assertIn("ERROR: RuntimeError: boom", output)


if __name__ == "__main__":
    unittest.main()
