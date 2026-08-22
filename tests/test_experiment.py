import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

from evaluation.experiment import preflight_check, print_summary_table, print_task_table, run_experiment, summarize
from generation.generator import CompletionGenerator
from selection.backends import SelectionBackend
from selection.llm_selector import LLMSelector

SAMPLE_REPO = Path(__file__).parent / "sample_repo"


def _fake_clone_and_checkout(owner, repo, commit, dest):
    shutil.copytree(SAMPLE_REPO, dest)


class FailOnCallSelectionBackend(SelectionBackend):
    """Selects candidates normally, except raises on exactly the Nth call
    (1-indexed) -- simulates one task in a batch failing partway through the
    pipeline while its neighbors succeed."""

    def __init__(self, fail_on_call: Optional[int] = None):
        self.fail_on_call = fail_on_call
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated backend failure")
        ids = [line.split("chunk_id: ", 1)[1] for line in prompt.splitlines() if line.startswith("chunk_id: ")]
        return json.dumps({"selected_chunk_ids": ids[:1]})


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


class PreflightCheckTest(unittest.TestCase):
    def test_returns_mapping_for_each_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1"), _make_task("t2")])
            mapping = preflight_check(str(jsonl_path), 2)
        self.assertEqual([m["task_id"] for m in mapping], ["t1", "t2"])
        self.assertEqual(mapping[0]["repository"], "someowner-somerepo-abc1234")

    def test_requesting_more_tasks_than_available_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1")])
            with self.assertRaises(ValueError):
                preflight_check(str(jsonl_path), 5)


class RunExperimentTest(unittest.TestCase):
    def test_all_tasks_succeed(self):
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
                selector = LLMSelector(chunks, backend=FailOnCallSelectionBackend())
                generator = CompletionGenerator(chunks, backend=generation_backend)

                outcome = run_experiment(
                    n_tasks=2,
                    jsonl_path=str(jsonl_path),
                    selector=selector,
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
                self.assertTrue(r["candidate_ids"][0].startswith("C"))  # display labels, not raw chunk_ids
                self.assertIsInstance(r["ES"], float)
                self.assertIsInstance(r["ID-F1"], float)
                self.assertIsInstance(r["generation_time"], float)

            # results file actually saved
            saved = [
                json.loads(line)
                for line in (Path(tmp_dir) / "results" / "cceval_2_results.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(saved), 2)
            saved_summary = json.loads((Path(tmp_dir) / "results" / "cceval_2_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary, summary)

    def test_one_task_failing_does_not_stop_the_others(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1"), _make_task("t2"), _make_task("t3")])

            generation_backend = MagicMock()
            generation_backend.generate.return_value = "some completion"

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                # fails on exactly the 2nd selection call (i.e. task 2 only)
                selector = LLMSelector(chunks, backend=FailOnCallSelectionBackend(fail_on_call=2))
                generator = CompletionGenerator(chunks, backend=generation_backend)

                outcome = run_experiment(
                    n_tasks=3,
                    jsonl_path=str(jsonl_path),
                    selector=selector,
                    generator=generator,
                    results_dir=str(Path(tmp_dir) / "results"),
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            results = outcome["results"]
            summary = outcome["summary"]

            self.assertEqual(len(results), 3)  # all 3 recorded, despite task 2 failing
            self.assertIsNone(results[0]["error"])
            self.assertIsNotNone(results[1]["error"])
            self.assertIn("simulated backend failure", results[1]["error"])
            self.assertIsNone(results[2]["error"])  # task 3 still ran after task 2's failure

            self.assertEqual(summary["num_successful"], 2)
            self.assertEqual(summary["num_failed"], 1)
            self.assertEqual(summary["failed_task_ids"], ["t2"])


class SummarizeTest(unittest.TestCase):
    def test_selection_ratio_averages_per_task_ratios(self):
        results = [
            {"error": None, "candidate_count": 10, "selected_count": 5, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 1.0},
            {"error": None, "candidate_count": 4, "selected_count": 1, "exact_match": False, "ES": 0.5, "ID-F1": 0.5, "generation_time": 2.0},
        ]
        summary = summarize(results)
        self.assertAlmostEqual(summary["avg_selection_ratio"], (0.5 + 0.25) / 2)
        self.assertEqual(summary["exact_match_rate"], 0.5)
        self.assertAlmostEqual(summary["avg_ES"], 0.75)
        self.assertAlmostEqual(summary["avg_generation_time"], 1.5)

    def test_all_failed_gives_none_averages_not_a_crash(self):
        results = [
            {
                "task_id": "t1",
                "error": "boom",
                **{k: None for k in ("candidate_count", "selected_count", "exact_match", "ES", "ID-F1", "generation_time")},
            }
        ]
        summary = summarize(results)
        self.assertEqual(summary["num_successful"], 0)
        self.assertIsNone(summary["avg_ES"])
        self.assertIsNone(summary["exact_match_rate"])
        self.assertIsNone(summary["avg_selection_ratio"])


class PrintFunctionsTest(unittest.TestCase):
    def test_print_summary_table_handles_none_averages(self):
        from io import StringIO

        summary = {
            "total_tasks": 0, "num_successful": 0, "num_failed": 0, "failed_task_ids": [],
            "avg_candidate_count": None, "avg_selected_count": None, "avg_selection_ratio": None,
            "exact_match_rate": None, "avg_ES": None, "avg_ID_F1": None, "avg_generation_time": None,
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_summary_table(summary)
        self.assertIn("n/a", buf.getvalue())

    def test_print_task_table_shows_errors_inline(self):
        from io import StringIO

        results = [
            {"task_id": "t1", "error": None, "candidate_count": 3, "selected_count": 1, "exact_match": True, "ES": 1.0, "ID-F1": 1.0, "generation_time": 0.5},
            {"task_id": "t2", "error": "RuntimeError: boom"},
        ]
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_task_table(results)
        output = buf.getvalue()
        self.assertIn("t1", output)
        self.assertIn("t2", output)
        self.assertIn("ERROR: RuntimeError: boom", output)


class CheckSelectionValidityTest(unittest.TestCase):
    def test_valid_non_empty_selection(self):
        from evaluation.experiment import check_selection_validity

        results = [
            {
                "task_id": "t1",
                "error": None,
                "candidate_count": 3,
                "selected_count": 2,
                "candidate_ids": ["C1", "C2", "C3"],
                "selected_candidate_ids": ["C1", "C2"],
            }
        ]
        rows = check_selection_validity(results)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["selection_valid"])
        self.assertIn("2/3", rows[0]["reason"])

    def test_empty_selection_is_flagged_but_not_resolved(self):
        from evaluation.experiment import check_selection_validity

        results = [
            {
                "task_id": "t1",
                "error": None,
                "candidate_count": 5,
                "selected_count": 0,
                "candidate_ids": ["C1", "C2", "C3", "C4", "C5"],
                "selected_candidate_ids": [],
            }
        ]
        rows = check_selection_validity(results)
        self.assertTrue(rows[0]["selection_valid"])  # trivially true: empty set is a subset of anything
        self.assertIn("cannot tell intentional vs. parsing failure", rows[0]["reason"])

    def test_selected_id_outside_candidate_pool_is_invalid(self):
        from evaluation.experiment import check_selection_validity

        results = [
            {
                "task_id": "t1",
                "error": None,
                "candidate_count": 2,
                "selected_count": 1,
                "candidate_ids": ["C1", "C2"],
                "selected_candidate_ids": ["C99"],  # not in the pool -- shouldn't happen, but verify detection
            }
        ]
        rows = check_selection_validity(results)
        self.assertFalse(rows[0]["selection_valid"])
        self.assertIn("INVALID", rows[0]["reason"])

    def test_failed_task_is_marked_not_applicable(self):
        from evaluation.experiment import check_selection_validity

        results = [{"task_id": "t1", "error": "LookupError: boom"}]
        rows = check_selection_validity(results)
        self.assertIsNone(rows[0]["selection_valid"])
        self.assertIn("boom", rows[0]["reason"])

    def test_print_selection_validity_table_runs(self):
        from io import StringIO
        from evaluation.experiment import check_selection_validity, print_selection_validity_table

        results = [
            {
                "task_id": "t1", "error": None, "candidate_count": 3, "selected_count": 2,
                "candidate_ids": ["C1", "C2", "C3"], "selected_candidate_ids": ["C1", "C2"],
            }
        ]
        rows = check_selection_validity(results)
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_selection_validity_table(rows)
        self.assertIn("t1", buf.getvalue())


class InspectTaskSelectionTest(unittest.TestCase):
    def test_intentional_empty_selection_is_diagnosed_correctly(self):
        from evaluation.experiment import inspect_task_selection, print_task_selection_diagnosis

        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "one_task.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1")])

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                empty_selection_backend = MagicMock()
                empty_selection_backend.generate.return_value = json.dumps({"selected_chunk_ids": []})
                selector = LLMSelector(chunks, backend=empty_selection_backend)

                diagnosis = inspect_task_selection(
                    "t1",
                    jsonl_path=str(jsonl_path),
                    selector=selector,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

        self.assertEqual(diagnosis["selected_labels"], [])
        self.assertEqual(diagnosis["raw_response"], '{"selected_chunk_ids": []}')
        self.assertEqual(diagnosis["parse_status"], "ok")

        from io import StringIO
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_task_selection_diagnosis(diagnosis)
        self.assertIn("INTENTIONAL EMPTY SELECTION", buf.getvalue())

    def test_fenced_empty_selection_is_diagnosed_as_intentional_not_a_parse_failure(self):
        # Reproduces the exact live bug: Qwen wrapped valid JSON in a
        # markdown code fence, and it used to get misclassified as a
        # parsing failure before the fence-stripping fix.
        from evaluation.experiment import inspect_task_selection, print_task_selection_diagnosis

        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "one_task.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1")])

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                fenced_backend = MagicMock()
                fenced_backend.generate.return_value = "```json\n[]\n```"
                selector = LLMSelector(chunks, backend=fenced_backend)

                diagnosis = inspect_task_selection(
                    "t1",
                    jsonl_path=str(jsonl_path),
                    selector=selector,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

        self.assertEqual(diagnosis["parse_status"], "ok")
        self.assertEqual(diagnosis["parsed_selection"], [])

        from io import StringIO
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_task_selection_diagnosis(diagnosis)
        output = buf.getvalue()
        self.assertIn("INTENTIONAL EMPTY SELECTION", output)
        self.assertNotIn("PARSING FAILURE", output)

    def test_unparseable_response_is_diagnosed_as_parsing_failure(self):
        from evaluation.experiment import inspect_task_selection, print_task_selection_diagnosis

        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "one_task.jsonl"
            _write_jsonl(jsonl_path, [_make_task("t1")])

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                from indexer.repo_parser import RepoParser

                chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
                garbage_backend = MagicMock()
                garbage_backend.generate.return_value = "I cannot help with that request."
                selector = LLMSelector(chunks, backend=garbage_backend)

                diagnosis = inspect_task_selection(
                    "t1",
                    jsonl_path=str(jsonl_path),
                    selector=selector,
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

        self.assertEqual(diagnosis["selected_labels"], [])
        self.assertEqual(diagnosis["parse_status"], "parse_error")

        from io import StringIO
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_task_selection_diagnosis(diagnosis)
        self.assertIn("PARSING FAILURE", buf.getvalue())

    def test_print_parse_diagnosis_matches_the_requested_compact_format(self):
        from evaluation.experiment import print_parse_diagnosis

        diagnosis = {
            "task_id": "project_cc_python/67",
            "raw_response": "```json\n[]\n```",
            "parsed_selection": [],
            "parse_status": "ok",
        }
        from io import StringIO
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_parse_diagnosis(diagnosis)
        output = buf.getvalue()
        self.assertIn("task_id: project_cc_python/67", output)
        self.assertIn("raw_response:", output)
        self.assertIn("parsed_selection: []", output)
        self.assertIn("parse_status: ok", output)


if __name__ == "__main__":
    unittest.main()
