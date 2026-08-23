import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.min2_selection import (
    MIN_SELECTED_CANDIDATES,
    print_min2_summary_table,
    print_min2_task_table,
    print_three_way_comparison,
    run_min2_selection_experiment,
    run_one_example_min2,
    summarize_min2,
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
    """Always returns a specific list of selected chunk_ids from the
    candidates it's offered -- lets tests force Qwen to under-select."""

    def __init__(self, num_to_select: int):
        self.num_to_select = num_to_select

    def generate(self, prompt: str) -> str:
        # candidates are shown as bare "C<n>" label lines now, not "chunk_id: X"
        labels = re.findall(r"^C\d+$", prompt, re.MULTILINE)
        return json.dumps({"selected_chunk_ids": labels[: self.num_to_select]})


class RunOneExampleMin2Test(unittest.TestCase):
    def _run(self, tmp_dir, num_to_select, groundtruth="x", completion="x"):
        jsonl_path = Path(tmp_dir) / "tasks.jsonl"
        _write_jsonl(jsonl_path, [_make_task("t1", groundtruth)])

        generation_backend = MagicMock()
        generation_backend.generate.return_value = completion

        with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
            "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
        ):
            from indexer.repo_parser import RepoParser

            chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
            selector = LLMSelector(chunks, backend=FixedSelectionBackend(num_to_select))
            generator = CompletionGenerator(chunks, backend=generation_backend)

            with patch.object(generator, "generate", wraps=generator.generate) as generate_spy:
                result = run_one_example_min2(
                    selector,
                    generator,
                    jsonl_path=str(jsonl_path),
                    index=0,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )
        return result, generate_spy

    def test_qwen_selecting_zero_is_filled_to_two(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result, generate_spy = self._run(tmp_dir, num_to_select=0)

        self.assertEqual(result["qwen_selected_ids"], [])
        self.assertEqual(len(result["final_selected_ids"]), MIN_SELECTED_CANDIDATES)
        # the fill must come from the original ranked candidate pool -- i.e.
        # the first MIN_SELECTED_CANDIDATES candidates in nomination order.
        expected_fill = [c["chunk_id"] for c in result["candidates"][:MIN_SELECTED_CANDIDATES]]
        self.assertEqual(result["final_selected_ids"], expected_fill)

        called_selected_ids = generate_spy.call_args[0][2]
        self.assertEqual(called_selected_ids, result["final_selected_ids"])

    def test_qwen_selecting_one_gets_exactly_one_more(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result, _ = self._run(tmp_dir, num_to_select=1)

        self.assertEqual(len(result["qwen_selected_ids"]), 1)
        self.assertEqual(len(result["final_selected_ids"]), MIN_SELECTED_CANDIDATES)
        # Qwen's original pick is preserved, not discarded.
        self.assertEqual(result["final_selected_ids"][0], result["qwen_selected_ids"][0])
        # the added candidate must come from the ranked pool and not duplicate Qwen's pick.
        self.assertNotEqual(result["final_selected_ids"][1], result["qwen_selected_ids"][0])
        candidate_ids = [c["chunk_id"] for c in result["candidates"]]
        self.assertIn(result["final_selected_ids"][1], candidate_ids)

    def test_qwen_selecting_two_or_more_is_used_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result, generate_spy = self._run(tmp_dir, num_to_select=3)

        self.assertEqual(len(result["qwen_selected_ids"]), 3)
        self.assertEqual(result["final_selected_ids"], result["qwen_selected_ids"])
        called_selected_ids = generate_spy.call_args[0][2]
        self.assertEqual(called_selected_ids, result["qwen_selected_ids"])

    def test_groundtruth_never_reaches_selection_or_generation_backend(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            secret = "THIS_MUST_NOT_LEAK"
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

                run_one_example_min2(
                    selector,
                    generator,
                    jsonl_path=str(jsonl_path),
                    index=0,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            self.assertTrue(all(secret not in p for p in captured_prompts))
            context_arg = generation_backend.generate.call_args[0][0]
            self.assertNotIn(secret, context_arg)


class RunMin2SelectionExperimentTest(unittest.TestCase):
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
                selector = LLMSelector(chunks, backend=FixedSelectionBackend(num_to_select=0))
                generator = CompletionGenerator(chunks, backend=generation_backend)

                outcome = run_min2_selection_experiment(
                    selector,
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
            self.assertEqual(summary["num_failed"], 0)
            self.assertEqual(summary["num_tasks_filled_to_minimum"], 2)  # both had 0 qwen selections
            for r in results:
                self.assertIsNone(r["error"])
                self.assertEqual(r["final_selected_count"], MIN_SELECTED_CANDIDATES)
                self.assertEqual(r["qwen_selected_ids"], [])

            results_path = Path(tmp_dir) / "results" / "cceval_2_min2_selection.jsonl"
            summary_path = Path(tmp_dir) / "results" / "cceval_2_min2_selection_summary.json"
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
                selector = LLMSelector(chunks, backend=FixedSelectionBackend(num_to_select=1))
                generator = CompletionGenerator(chunks, backend=generation_backend)

                outcome = run_min2_selection_experiment(
                    selector,
                    generator,
                    n_tasks=3,
                    jsonl_path=str(jsonl_path),
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

    def test_does_not_touch_baseline_or_selection_experiment_result_files(self):
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
                from evaluation.baseline import run_baseline_experiment

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]

                selector1 = LLMSelector(chunks, backend=FixedSelectionBackend(num_to_select=1))
                generator1 = CompletionGenerator(chunks, backend=generation_backend)
                run_experiment(
                    n_tasks=1, jsonl_path=str(jsonl_path), selector=selector1, generator=generator1,
                    results_dir=str(results_dir), repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )
                v1_results_path = results_dir / "cceval_1_results.jsonl"
                v1_mtime_before = v1_results_path.stat().st_mtime

                generator2 = CompletionGenerator(chunks, backend=generation_backend)
                run_baseline_experiment(
                    generator2, n_tasks=1, jsonl_path=str(jsonl_path),
                    results_dir=str(results_dir), repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )
                baseline_results_path = results_dir / "cceval_1_baseline.jsonl"
                baseline_mtime_before = baseline_results_path.stat().st_mtime

                selector3 = LLMSelector(chunks, backend=FixedSelectionBackend(num_to_select=1))
                generator3 = CompletionGenerator(chunks, backend=generation_backend)
                run_min2_selection_experiment(
                    selector3, generator3, n_tasks=1, jsonl_path=str(jsonl_path),
                    results_dir=str(results_dir), repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            self.assertEqual(v1_results_path.stat().st_mtime, v1_mtime_before)
            self.assertEqual(baseline_results_path.stat().st_mtime, baseline_mtime_before)
            self.assertTrue((results_dir / "cceval_1_min2_selection.jsonl").exists())
            self.assertTrue((results_dir / "cceval_1_min2_selection_summary.json").exists())


class SummarizeMin2Test(unittest.TestCase):
    def test_averages_and_fill_count(self):
        results = [
            {"task_id": "t1", "error": None, "candidate_count": 10, "qwen_selected_ids": [], "final_selected_count": 2, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 1.0},
            {"task_id": "t2", "error": None, "candidate_count": 4, "qwen_selected_ids": ["a"], "final_selected_count": 2, "exact_match": False, "ES": 0.5, "ID-F1": 0.5, "generation_time": 2.0},
            {"task_id": "t3", "error": None, "candidate_count": 6, "qwen_selected_ids": ["a", "b", "c"], "final_selected_count": 3, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 1.0},
        ]
        summary = summarize_min2(results)
        self.assertEqual(summary["num_tasks_filled_to_minimum"], 2)  # t1, t2 had < 2 qwen selections
        self.assertAlmostEqual(summary["avg_final_selected_count"], (2 + 2 + 3) / 3)
        self.assertAlmostEqual(summary["exact_match_rate"], 2 / 3)

    def test_all_failed_gives_none_averages_not_a_crash(self):
        results = [
            {"task_id": "t1", "error": "boom", "candidate_count": None, "qwen_selected_ids": None, "final_selected_count": None, "exact_match": None, "ES": None, "ID-F1": None, "generation_time": None},
        ]
        summary = summarize_min2(results)
        self.assertEqual(summary["num_successful"], 0)
        self.assertIsNone(summary["avg_ES"])
        self.assertEqual(summary["num_tasks_filled_to_minimum"], 0)


class PrintFunctionsTest(unittest.TestCase):
    def test_print_min2_summary_table_handles_none_averages(self):
        from io import StringIO

        summary = {
            "total_tasks": 0, "num_successful": 0, "num_failed": 0, "failed_task_ids": [],
            "avg_candidate_count": None, "avg_final_selected_count": None,
            "num_tasks_filled_to_minimum": 0, "exact_match_rate": None, "avg_ES": None,
            "avg_ID_F1": None, "avg_generation_time": None,
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_min2_summary_table(summary)
        self.assertIn("n/a", buf.getvalue())

    def test_print_min2_task_table_shows_errors_inline(self):
        from io import StringIO

        results = [
            {"task_id": "t1", "error": None, "candidate_count": 3, "qwen_selected_ids": [], "final_selected_count": 2, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 0.5},
            {"task_id": "t2", "error": "RuntimeError: boom"},
        ]
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_min2_task_table(results)
        output = buf.getvalue()
        self.assertIn("t1", output)
        self.assertIn("t2", output)
        self.assertIn("ERROR: RuntimeError: boom", output)

    def test_print_three_way_comparison_runs(self):
        from io import StringIO

        baseline_summary = {"avg_candidate_count": 10.89, "exact_match_rate": 0.389, "avg_ES": 0.641, "avg_ID_F1": 0.655, "avg_generation_time": 37.01}
        qwen_summary = {"avg_candidate_count": 10.89, "avg_selected_count": 1.89, "exact_match_rate": 0.278, "avg_ES": 0.563, "avg_ID_F1": 0.587, "avg_generation_time": 34.65}
        min2_summary = {"avg_candidate_count": 10.89, "avg_final_selected_count": 2.3, "exact_match_rate": 0.35, "avg_ES": 0.6, "avg_ID_F1": 0.6, "avg_generation_time": 35.0}

        buf = StringIO()
        with patch("sys.stdout", buf):
            print_three_way_comparison(baseline_summary, qwen_summary, min2_summary)
        output = buf.getvalue()
        self.assertIn("no selection", output)
        self.assertIn("qwen selection", output)
        self.assertIn("min2_selection", output)


if __name__ == "__main__":
    unittest.main()
