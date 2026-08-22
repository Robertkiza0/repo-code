import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.selection_diagnosis import (
    MEANINGFUL_OVERLAP_RATIO,
    diagnose_task,
    find_selection_only_failures,
    print_diagnosis,
    print_diagnosis_summary,
    run_selection_diagnosis,
    summarize_diagnosis_categories,
)

SAMPLE_REPO = Path(__file__).parent / "sample_repo"

# Fixed by tests/sample_repo + prompt "result = Greeter(" / file "pkg/module_b.py"
# (same fixture used throughout the other CCEval tests) -- retrieval is
# deterministic, so this label -> chunk ordering never changes:
#   C1 Greeter (class)        C5 Greeter.greet (method, nested _format)
#   C2 Greeter.__init__       C6 Greeter.shout
#   C3 _format                C7 LoudGreeter (class)
#   C4 greet (module fn)      C8 LoudGreeter.greet


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


def _result_record(task_id, error=None, exact_match=None, ES=0.0, id_f1=0.0, selected_candidate_ids=None):
    record = {"task_id": task_id, "error": error, "exact_match": exact_match, "ES": ES, "ID-F1": id_f1}
    if selected_candidate_ids is not None:
        record["selected_candidate_ids"] = selected_candidate_ids
    return record


class FindSelectionOnlyFailuresTest(unittest.TestCase):
    def test_only_flags_baseline_true_qwen_false(self):
        baseline_by_task = {
            "t1": _result_record("t1", exact_match=True),
            "t2": _result_record("t2", exact_match=False),
            "t3": _result_record("t3", exact_match=True),
        }
        qwen_by_task = {
            "t1": _result_record("t1", exact_match=False),
            "t2": _result_record("t2", exact_match=False),
            "t3": _result_record("t3", exact_match=True),
        }
        self.assertEqual(find_selection_only_failures(baseline_by_task, qwen_by_task), ["t1"])

    def test_excludes_tasks_with_errors(self):
        baseline_by_task = {"t1": _result_record("t1", error="boom", exact_match=None)}
        qwen_by_task = {"t1": _result_record("t1", exact_match=False)}
        self.assertEqual(find_selection_only_failures(baseline_by_task, qwen_by_task), [])

    def test_excludes_tasks_missing_from_qwen_results(self):
        baseline_by_task = {"t1": _result_record("t1", exact_match=True)}
        qwen_by_task = {}
        self.assertEqual(find_selection_only_failures(baseline_by_task, qwen_by_task), [])


class DiagnoseTaskTest(unittest.TestCase):
    def _diagnose(self, tmp_dir, groundtruth, selected_labels):
        jsonl_path = Path(tmp_dir) / "tasks.jsonl"
        _write_jsonl(jsonl_path, [_make_task("t1", groundtruth)])

        with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
            "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
        ):
            baseline_record = _result_record("t1", exact_match=True, ES=1.0, id_f1=1.0)
            qwen_record = _result_record(
                "t1", exact_match=False, ES=0.5, id_f1=0.5, selected_candidate_ids=selected_labels
            )
            return diagnose_task(
                "t1",
                baseline_record,
                qwen_record,
                jsonl_path=str(jsonl_path),
                repos_dir=str(Path(tmp_dir) / "repos"),
                index_dir=str(Path(tmp_dir) / "indexes"),
            )

    def test_missed_relevant_candidate(self):
        # "default_greeting" only appears in C1/C2/C5's source; Qwen picked
        # only C7 (LoudGreeter), which has zero overlap.
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(tmp_dir, groundtruth="self.default_greeting", selected_labels=["C7"])

        self.assertEqual(diagnosis["category"], "missed_relevant_candidate")
        c2 = next(c for c in diagnosis["candidates"] if c["label"] == "C2")
        self.assertFalse(c2["selected_by_qwen"])
        self.assertIn("default_greeting", c2["shared_identifiers"])

    def test_candidate_pool_missing_context(self):
        # No candidate's source contains this identifier at all.
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(
                tmp_dir, groundtruth="self.totally_unrelated_field_xyz", selected_labels=["C7"]
            )

        self.assertEqual(diagnosis["category"], "candidate_pool_missing_context")
        self.assertTrue(all(c["overlap_ratio"] == 0.0 for c in diagnosis["candidates"]))

    def test_unclear_when_qwen_selected_the_best_available_candidate(self):
        # Qwen picked C2, which has full (1.0) overlap with groundtruth --
        # by this heuristic Qwen chose well, yet EM still failed.
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(tmp_dir, groundtruth="self.default_greeting", selected_labels=["C2"])

        self.assertEqual(diagnosis["category"], "unclear")

    def test_unclear_when_groundtruth_has_no_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(tmp_dir, groundtruth=")", selected_labels=["C7"])

        self.assertEqual(diagnosis["category"], "unclear")
        self.assertIn("no extractable", diagnosis["category_reason"])

    def test_selected_irrelevant_context_when_only_weak_overlap_exists(self):
        # 4 groundtruth identifiers, only "default_greeting" matches
        # anything (ratio 0.25, below the 0.34 meaningful threshold) --
        # Qwen picked the one candidate tied for the pool's best (weak) match.
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(
                tmp_dir,
                groundtruth="default_greeting_val = default_greeting + alpha + beta",
                selected_labels=["C2"],
            )

        self.assertEqual(diagnosis["category"], "selected_irrelevant_context")
        c2 = next(c for c in diagnosis["candidates"] if c["label"] == "C2")
        self.assertLess(c2["overlap_ratio"], MEANINGFUL_OVERLAP_RATIO)

    def test_no_selection_and_qwen_selection_results_are_passed_through_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(tmp_dir, groundtruth="self.default_greeting", selected_labels=["C7"])

        self.assertEqual(diagnosis["no_selection_result"], {"exact_match": True, "ES": 1.0, "ID-F1": 1.0})
        self.assertEqual(diagnosis["qwen_selection_result"], {"exact_match": False, "ES": 0.5, "ID-F1": 0.5})
        self.assertEqual(diagnosis["qwen_selected_labels"], ["C7"])


class RunSelectionDiagnosisTest(unittest.TestCase):
    def test_end_to_end_reads_saved_results_and_diagnoses_each_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(
                jsonl_path,
                [
                    _make_task("t1", "self.default_greeting"),  # selection-only failure
                    _make_task("t2", "x"),  # both succeed -- not a selection-only failure
                ],
            )

            results_dir = Path(tmp_dir)
            baseline_jsonl = results_dir / "baseline.jsonl"
            qwen_jsonl = results_dir / "qwen.jsonl"

            with baseline_jsonl.open("w", encoding="utf-8") as f:
                f.write(json.dumps(_result_record("t1", exact_match=True, ES=1.0, id_f1=1.0)) + "\n")
                f.write(json.dumps(_result_record("t2", exact_match=True, ES=1.0, id_f1=1.0)) + "\n")

            with qwen_jsonl.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        _result_record("t1", exact_match=False, ES=0.5, id_f1=0.5, selected_candidate_ids=["C7"])
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        _result_record("t2", exact_match=True, ES=1.0, id_f1=1.0, selected_candidate_ids=["C1"])
                    )
                    + "\n"
                )

            with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
                "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
            ):
                diagnoses = run_selection_diagnosis(
                    baseline_jsonl=str(baseline_jsonl),
                    qwen_jsonl=str(qwen_jsonl),
                    jsonl_path=str(jsonl_path),
                    repos_dir=str(Path(tmp_dir) / "repos"),
                    index_dir=str(Path(tmp_dir) / "indexes"),
                )

            self.assertEqual(len(diagnoses), 1)  # only t1 qualifies
            self.assertEqual(diagnoses[0]["task_id"], "t1")
            self.assertEqual(diagnoses[0]["category"], "missed_relevant_candidate")

            counts = summarize_diagnosis_categories(diagnoses)
            self.assertEqual(counts["missed_relevant_candidate"], 1)
            self.assertEqual(sum(counts.values()), 1)


class PrintFunctionsTest(unittest.TestCase):
    def test_print_diagnosis_runs_and_shows_key_fields(self):
        from io import StringIO

        diagnosis = {
            "task_id": "t1",
            "candidates": [
                {
                    "label": "C1",
                    "chunk_id": "abc",
                    "file_path": "module_a.py",
                    "name": "Greeter",
                    "sources": ["bm25", "symbol"],
                    "selected_by_qwen": False,
                    "shared_identifiers": ["default_greeting"],
                    "overlap_ratio": 1.0,
                },
                {
                    "label": "C7",
                    "chunk_id": "def",
                    "file_path": "pkg/module_b.py",
                    "name": "LoudGreeter",
                    "sources": ["bm25"],
                    "selected_by_qwen": True,
                    "shared_identifiers": [],
                    "overlap_ratio": 0.0,
                },
            ],
            "qwen_selected_labels": ["C7"],
            "no_selection_result": {"exact_match": True, "ES": 1.0, "ID-F1": 1.0},
            "qwen_selection_result": {"exact_match": False, "ES": 0.5, "ID-F1": 0.5},
            "category": "missed_relevant_candidate",
            "category_reason": "an unselected candidate has higher overlap",
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_diagnosis(diagnosis)
        output = buf.getvalue()
        self.assertIn("task_id: t1", output)
        self.assertIn("C1", output)
        self.assertIn("selected by Qwen", output)
        self.assertIn("category: missed_relevant_candidate", output)

    def test_print_diagnosis_summary_runs(self):
        from io import StringIO

        diagnoses = [{"category": "missed_relevant_candidate"}, {"category": "unclear"}]
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_diagnosis_summary(diagnoses)
        output = buf.getvalue()
        self.assertIn("missed_relevant_candidate", output)
        self.assertIn("unclear", output)


if __name__ == "__main__":
    unittest.main()
