import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.baseline import (
    print_baseline_summary_table,
    print_baseline_task_table,
    run_baseline_experiment,
    run_one_example_baseline,
    summarize_baseline,
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


class RunOneExampleBaselineTest(unittest.TestCase):
    def test_all_candidates_passed_to_generator_no_selector_involved(self):
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
                    result = run_one_example_baseline(
                        jsonl_path=str(jsonl_path),
                        index=0,
                        generator=generator,
                        repos_dir=str(Path(tmp_dir) / "repos"),
                        index_dir=str(Path(tmp_dir) / "indexes"),
                    )

            self.assertEqual(result["task_id"], "t1")
            self.assertGreater(result["num_candidates"], 0)
            self.assertEqual(result["completion"], "x")

            # generator.generate() must have been called exactly once, with
            # every nominated candidate id -- not a Qwen-selected subset,
            # since no selection happened.
            generate_spy.assert_called_once()
            called_selected_ids = generate_spy.call_args[0][2]
            all_candidate_ids = [c["chunk_id"] for c in result["candidates"]]
            self.assertEqual(set(called_selected_ids), set(all_candidate_ids))
            self.assertEqual(len(called_selected_ids), len(all_candidate_ids))

    def test_groundtruth_never_reaches_generation_backend(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            secret_groundtruth = "THIS_MUST_NOT_LEAK"
            _write_jsonl(jsonl_path, [_make_task("t1", secret_groundtruth)])

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "some completion"

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                generator = CompletionGenerator(chunks, backend=generation_backend)

                run_one_example_baseline(
                    jsonl_path=str(jsonl_path),
                    index=0,
                    generator=generator,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            context_arg = generation_backend.generate.call_args[0][0]
            self.assertNotIn(secret_groundtruth, context_arg)


class RunBaselineExperimentTest(unittest.TestCase):
    def test_all_tasks_succeed_and_saves_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1", "x"), _make_task("t2", "y")])

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "x"  # matches t1's groundtruth, not t2's

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                generator = CompletionGenerator(chunks, backend=generation_backend)

                outcome = run_baseline_experiment(
                    n_tasks=2,
                    jsonl_path=str(jsonl_path),
                    generator=generator,
                    results_dir=str(Path(tmp_dir) / "results"),
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            results = outcome["results"]
            summary = outcome["summary"]

            self.assertEqual(len(results), 2)
            self.assertEqual(summary["num_successful"], 2)
            self.assertEqual(summary["num_failed"], 0)
            self.assertEqual(summary["exact_match_rate"], 0.5)  # t1 matches, t2 doesn't
            for r in results:
                self.assertIsNone(r["error"])
                self.assertGreater(r["candidate_count"], 0)
                self.assertIsInstance(r["ES"], float)
                self.assertIsInstance(r["ID-F1"], float)
                self.assertIsInstance(r["generation_time"], float)
                # no selection-related fields -- there is no selection stage
                self.assertNotIn("selected_count", r)
                self.assertNotIn("selected_candidate_ids", r)

            results_path = Path(tmp_dir) / "results" / "cceval_2_baseline.jsonl"
            summary_path = Path(tmp_dir) / "results" / "cceval_2_baseline_summary.json"
            self.assertTrue(results_path.exists())
            self.assertTrue(summary_path.exists())

            saved = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(saved), 2)
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

                outcome = run_baseline_experiment(
                    n_tasks=3,
                    jsonl_path=str(jsonl_path),
                    generator=generator,
                    results_dir=str(Path(tmp_dir) / "results"),
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            results = outcome["results"]
            summary = outcome["summary"]

            self.assertEqual(len(results), 3)
            self.assertIsNone(results[0]["error"])
            self.assertIsNotNone(results[1]["error"])
            self.assertIn("simulated backend failure", results[1]["error"])
            self.assertIsNone(results[2]["error"])

            self.assertEqual(summary["num_successful"], 2)
            self.assertEqual(summary["num_failed"], 1)
            self.assertEqual(summary["failed_task_ids"], ["t2"])

    def test_does_not_touch_selection_experiment_result_files(self):
        # Guards the "do not overwrite the existing selection experiment"
        # requirement: run_experiment's v1 output must be untouched by a
        # baseline run into the same results_dir.
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
                from evaluation.experiment import run_experiment
                from selection.backends import SelectionBackend
                from selection.llm_selector import LLMSelector

                class _FixedSelectionBackend(SelectionBackend):
                    def generate(self, prompt: str) -> str:
                        # candidates are shown as bare "C<n>" label lines now, not "chunk_id: X"
                        labels = re.findall(r"^C\d+$", prompt, re.MULTILINE)
                        return json.dumps({"selected_chunk_ids": labels[:1]})

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                selector = LLMSelector(chunks, backend=_FixedSelectionBackend())
                generator1 = CompletionGenerator(chunks, backend=generation_backend)
                run_experiment(
                    n_tasks=1, jsonl_path=str(jsonl_path), selector=selector, generator=generator1,
                    results_dir=str(results_dir), repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

                v1_results_path = results_dir / "cceval_1_results.jsonl"
                v1_mtime_before = v1_results_path.stat().st_mtime

                generator2 = CompletionGenerator(chunks, backend=generation_backend)
                run_baseline_experiment(
                    n_tasks=1, jsonl_path=str(jsonl_path), generator=generator2,
                    results_dir=str(results_dir), repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            self.assertTrue(v1_results_path.exists())
            self.assertEqual(v1_results_path.stat().st_mtime, v1_mtime_before)
            self.assertTrue((results_dir / "cceval_1_baseline.jsonl").exists())
            self.assertTrue((results_dir / "cceval_1_baseline_summary.json").exists())


class SummarizeBaselineTest(unittest.TestCase):
    def test_averages_over_successful_tasks_only(self):
        results = [
            {"task_id": "t1", "error": None, "candidate_count": 10, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 1.0},
            {"task_id": "t2", "error": None, "candidate_count": 4, "exact_match": False, "ES": 0.5, "ID-F1": 0.5, "generation_time": 2.0},
            {"task_id": "t3", "error": "boom", "candidate_count": None, "exact_match": None, "ES": None, "ID-F1": None, "generation_time": None},
        ]
        summary = summarize_baseline(results)
        self.assertEqual(summary["total_tasks"], 3)
        self.assertEqual(summary["num_successful"], 2)
        self.assertEqual(summary["num_failed"], 1)
        self.assertEqual(summary["failed_task_ids"], ["t3"])
        self.assertAlmostEqual(summary["avg_candidate_count"], 7.0)
        self.assertEqual(summary["exact_match_rate"], 0.5)
        self.assertAlmostEqual(summary["avg_ES"], 0.75)
        self.assertAlmostEqual(summary["avg_ID_F1"], 0.75)
        self.assertAlmostEqual(summary["avg_generation_time"], 1.5)
        # no selection-related keys -- there is no selection stage in the baseline
        self.assertNotIn("avg_selected_count", summary)
        self.assertNotIn("avg_selection_ratio", summary)

    def test_all_failed_gives_none_averages_not_a_crash(self):
        results = [
            {"task_id": "t1", "error": "boom", "candidate_count": None, "exact_match": None, "ES": None, "ID-F1": None, "generation_time": None},
        ]
        summary = summarize_baseline(results)
        self.assertEqual(summary["num_successful"], 0)
        self.assertIsNone(summary["avg_ES"])
        self.assertIsNone(summary["exact_match_rate"])
        self.assertIsNone(summary["avg_candidate_count"])


class PrintFunctionsTest(unittest.TestCase):
    def test_print_baseline_summary_table_handles_none_averages(self):
        from io import StringIO

        summary = {
            "total_tasks": 0, "num_successful": 0, "num_failed": 0, "failed_task_ids": [],
            "avg_candidate_count": None, "exact_match_rate": None, "avg_ES": None,
            "avg_ID_F1": None, "avg_generation_time": None,
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_baseline_summary_table(summary)
        self.assertIn("n/a", buf.getvalue())

    def test_print_baseline_task_table_matches_requested_format(self):
        from io import StringIO

        results = [
            {"task_id": "t1", "error": None, "candidate_count": 12, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 0.5},
            {"task_id": "t2", "error": "RuntimeError: boom"},
        ]
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_baseline_task_table(results)
        output = buf.getvalue()
        self.assertIn("task_id", output)
        self.assertIn("candidates", output)
        self.assertIn("t1", output)
        self.assertIn("t2", output)
        self.assertIn("ERROR: RuntimeError: boom", output)


if __name__ == "__main__":
    unittest.main()
