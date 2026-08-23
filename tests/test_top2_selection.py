import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.top2_selection import (
    TOP_N,
    print_top2_summary_table,
    print_top2_task_table,
    run_one_example_top2,
    run_top2_experiment,
    summarize_top2,
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


class RunOneExampleTop2Test(unittest.TestCase):
    def test_selects_exactly_the_first_two_candidates_in_pool_order(self):
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
                generator = CompletionGenerator(chunks, backend=generation_backend)

                with patch.object(generator, "generate", wraps=generator.generate) as generate_spy:
                    result = run_one_example_top2(
                        generator,
                        jsonl_path=str(jsonl_path),
                        index=0,
                        repos_dir=str(Path(tmp_dir) / "repos"),
                        index_dir=str(Path(tmp_dir) / "indexes"),
                    )

            self.assertEqual(result["task_id"], "t1")
            self.assertGreater(result["num_candidates"], TOP_N)
            expected = [c["chunk_id"] for c in result["candidates"][:TOP_N]]
            self.assertEqual(result["selected_chunk_ids"], expected)
            self.assertEqual(len(result["selected_chunk_ids"]), TOP_N)

            called_selected_ids = generate_spy.call_args[0][2]
            self.assertEqual(called_selected_ids, expected)

    def test_groundtruth_never_reaches_generation_backend(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            secret = "THIS_MUST_NOT_LEAK"
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1", secret)])

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "some completion"

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                generator = CompletionGenerator(chunks, backend=generation_backend)

                run_one_example_top2(
                    generator,
                    jsonl_path=str(jsonl_path),
                    index=0,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            context_arg = generation_backend.generate.call_args[0][0]
            self.assertNotIn(secret, context_arg)


class RunTop2ExperimentTest(unittest.TestCase):
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

                outcome = run_top2_experiment(
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
            self.assertEqual(summary["exact_match_rate"], 0.5)
            for r in results:
                self.assertIsNone(r["error"])
                self.assertEqual(r["selected_count"], TOP_N)
                self.assertEqual(len(r["selected_candidate_ids"]), TOP_N)

            results_path = Path(tmp_dir) / "results" / "cceval_2_top2_selection.jsonl"
            summary_path = Path(tmp_dir) / "results" / "cceval_2_top2_selection_summary.json"
            self.assertTrue(results_path.exists())
            self.assertTrue(summary_path.exists())
            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_summary, summary)

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
                generator = CompletionGenerator(chunks, backend=generation_backend)

                outcome = run_top2_experiment(
                    generator,
                    n_tasks=3,
                    jsonl_path=str(jsonl_path),
                    results_dir=str(Path(tmp_dir) / "results"),
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            results = outcome["results"]
            self.assertEqual(len(results), 3)
            self.assertIsNone(results[0]["error"])
            self.assertIn("simulated backend failure", results[1]["error"])
            self.assertIsNone(results[2]["error"])


class SummarizeTop2Test(unittest.TestCase):
    def test_averages_over_successful_tasks_only(self):
        results = [
            {"task_id": "t1", "error": None, "candidate_count": 10, "selected_count": 2, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 1.0},
            {"task_id": "t2", "error": "boom", "candidate_count": None, "selected_count": None, "exact_match": None, "ES": None, "ID-F1": None, "generation_time": None},
        ]
        summary = summarize_top2(results)
        self.assertEqual(summary["num_successful"], 1)
        self.assertEqual(summary["num_failed"], 1)
        self.assertAlmostEqual(summary["avg_selected_count"], 2.0)
        self.assertEqual(summary["exact_match_rate"], 1.0)


class PrintFunctionsTest(unittest.TestCase):
    def test_print_top2_summary_table_handles_none_averages(self):
        from io import StringIO

        summary = {
            "total_tasks": 0, "num_successful": 0, "num_failed": 0, "failed_task_ids": [],
            "avg_candidate_count": None, "avg_selected_count": None, "exact_match_rate": None,
            "avg_ES": None, "avg_ID_F1": None, "avg_generation_time": None,
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_top2_summary_table(summary)
        self.assertIn("n/a", buf.getvalue())

    def test_print_top2_task_table_shows_errors_inline(self):
        from io import StringIO

        results = [
            {"task_id": "t1", "error": None, "candidate_count": 5, "selected_count": 2, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 0.5},
            {"task_id": "t2", "error": "RuntimeError: boom"},
        ]
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_top2_task_table(results)
        output = buf.getvalue()
        self.assertIn("t1", output)
        self.assertIn("ERROR: RuntimeError: boom", output)


if __name__ == "__main__":
    unittest.main()
