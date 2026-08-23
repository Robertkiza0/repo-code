import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.selection_diagnosis import (
    CATEGORIES,
    MEANINGFUL_OVERLAP_RATIO,
    diagnose_task,
    find_selection_only_failures,
    print_diagnosis,
    print_diagnosis_isolation_flags,
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


def _result_record(task_id, error=None, exact_match=None, ES=0.0, id_f1=0.0, selected_candidate_ids=None, completion=""):
    record = {
        "task_id": task_id,
        "error": error,
        "exact_match": exact_match,
        "ES": ES,
        "ID-F1": id_f1,
        "completion": completion,
    }
    if selected_candidate_ids is not None:
        record["selected_candidate_ids"] = selected_candidate_ids
    return record


class FindSelectionOnlyFailuresTest(unittest.TestCase):
    def test_only_flags_baseline_true_llm_selection_false(self):
        baseline_by_task = {
            "t1": _result_record("t1", exact_match=True),
            "t2": _result_record("t2", exact_match=False),
            "t3": _result_record("t3", exact_match=True),
        }
        llm_selection_by_task = {
            "t1": _result_record("t1", exact_match=False),
            "t2": _result_record("t2", exact_match=False),
            "t3": _result_record("t3", exact_match=True),
        }
        self.assertEqual(find_selection_only_failures(baseline_by_task, llm_selection_by_task), ["t1"])

    def test_excludes_tasks_with_errors(self):
        baseline_by_task = {"t1": _result_record("t1", error="boom", exact_match=None)}
        llm_selection_by_task = {"t1": _result_record("t1", exact_match=False)}
        self.assertEqual(find_selection_only_failures(baseline_by_task, llm_selection_by_task), [])

    def test_excludes_tasks_missing_from_llm_selection_results(self):
        baseline_by_task = {"t1": _result_record("t1", exact_match=True)}
        llm_selection_by_task = {}
        self.assertEqual(find_selection_only_failures(baseline_by_task, llm_selection_by_task), [])


class DiagnoseTaskTest(unittest.TestCase):
    def _diagnose(self, tmp_dir, groundtruth, selected_labels):
        jsonl_path = Path(tmp_dir) / "tasks.jsonl"
        _write_jsonl(jsonl_path, [_make_task("t1", groundtruth)])

        with patch("evaluation.cceval_adapter.resolve_owner_repo", return_value=("o", "r", "c")), patch(
            "evaluation.cceval_adapter.clone_and_checkout", side_effect=_fake_clone_and_checkout
        ):
            baseline_record = _result_record("t1", exact_match=True, ES=1.0, id_f1=1.0, completion="no-sel completion")
            llm_selection_record = _result_record(
                "t1", exact_match=False, ES=0.5, id_f1=0.5, selected_candidate_ids=selected_labels,
                completion="llm-sel completion",
            )
            return diagnose_task(
                "t1",
                baseline_record,
                llm_selection_record,
                jsonl_path=str(jsonl_path),
                repos_dir=str(Path(tmp_dir) / "repos"),
                index_dir=str(Path(tmp_dir) / "indexes"),
            )

    def test_candidate_pool_missing_context(self):
        # No candidate's source contains this identifier at all.
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(
                tmp_dir, groundtruth="self.totally_unrelated_field_xyz", selected_labels=["C7"]
            )
        self.assertEqual(diagnosis["category"], "candidate_pool_missing_context")
        self.assertTrue(all(c["overlap_ratio"] == 0.0 for c in diagnosis["candidates"]))

    def test_unclear_when_groundtruth_has_no_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(tmp_dir, groundtruth=")", selected_labels=["C7"])
        self.assertEqual(diagnosis["category"], "unclear")
        self.assertIn("no extractable", diagnosis["category_reason"])

    def test_selected_irrelevant_context_when_only_weak_overlap_exists(self):
        # 4 groundtruth identifiers, only "default_greeting" matches
        # anything (ratio 0.25, below the 0.34 meaningful threshold) --
        # the selector picked the one candidate tied for the pool's best
        # (weak) match, and nothing else in the pool is any better.
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(
                tmp_dir,
                groundtruth="default_greeting_val = default_greeting + alpha + beta",
                selected_labels=["C2"],
            )
        self.assertEqual(diagnosis["category"], "selected_irrelevant_context")
        c2 = next(c for c in diagnosis["candidates"] if c["label"] == "C2")
        self.assertLess(c2["overlap_ratio"], MEANINGFUL_OVERLAP_RATIO)
        self.assertIn("irrelevant_selected", diagnosis["category_evidence"])

    def test_missed_relevant_candidate_when_only_one_relevant_candidate_exists(self):
        # "Hello"/"a" (plus a nonexistent id, to keep the ratio below 1.0)
        # only appear in C4 (the standalone module-level greet() function) --
        # exactly one relevant candidate in the whole pool. Selecting C7
        # (zero overlap) instead of C4 is a clean miss, not an under-selection.
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(
                tmp_dir, groundtruth="Hello a nonexistentxyz", selected_labels=["C7"]
            )
        self.assertEqual(diagnosis["category"], "missed_relevant_candidate")
        self.assertEqual(diagnosis["category_evidence"]["missed_candidate"]["label"], "C4")

    def test_selected_too_few_when_multiple_relevant_candidates_exist_but_fewer_are_selected(self):
        # "_format"/"default_greeting" (+ a nonexistent id) are both fully
        # present in C1 and C5 (ratio 0.667 each, >= threshold) -- 2 relevant
        # candidates exist, but only 1 was selected.
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(
                tmp_dir,
                groundtruth="default_greeting _format nonexistentxyz123",
                selected_labels=["C1"],
            )
        self.assertEqual(diagnosis["category"], "selected_too_few")
        self.assertEqual(diagnosis["category_evidence"]["relevant_count"], 2)
        self.assertEqual(diagnosis["category_evidence"]["selected_count"], 1)
        missed_labels = {c["label"] for c in diagnosis["category_evidence"]["missed_relevant"]}
        self.assertEqual(missed_labels, {"C5"})

    def test_selected_too_many_when_most_selected_candidates_show_no_evidence(self):
        # Only C4 is relevant (see missed_relevant_candidate test above), but
        # 4 candidates were selected including C4 -- most of the selection
        # shows no evidence of relevance.
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(
                tmp_dir, groundtruth="Hello a nonexistentxyz", selected_labels=["C4", "C6", "C7", "C8"]
            )
        self.assertEqual(diagnosis["category"], "selected_too_many")
        self.assertEqual(diagnosis["category_evidence"]["selected_count"], 4)

    def test_relevant_candidate_present_but_generation_failed(self):
        # C4 is the single relevant candidate and it WAS selected (alone) --
        # the selector chose well by this heuristic, yet EM still failed.
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(tmp_dir, groundtruth="Hello a nonexistentxyz", selected_labels=["C4"])
        self.assertEqual(diagnosis["category"], "relevant_candidate_present_but_generation_failed")
        self.assertEqual(diagnosis["category_evidence"]["relevant_selected"]["label"], "C4")

    def test_completions_and_results_are_passed_through_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnosis = self._diagnose(
                tmp_dir, groundtruth="self.totally_unrelated_field_xyz", selected_labels=["C7"]
            )
        self.assertEqual(diagnosis["no_selection_result"], {"exact_match": True, "ES": 1.0, "ID-F1": 1.0})
        self.assertEqual(diagnosis["llm_selection_result"], {"exact_match": False, "ES": 0.5, "ID-F1": 0.5})
        self.assertEqual(diagnosis["no_selection_completion"], "no-sel completion")
        self.assertEqual(diagnosis["llm_selection_completion"], "llm-sel completion")
        self.assertEqual(diagnosis["selected_labels"], ["C7"])
        self.assertNotIn("C7", diagnosis["not_selected_labels"])
        self.assertIn("C1", diagnosis["not_selected_labels"])


class RunSelectionDiagnosisTest(unittest.TestCase):
    def test_end_to_end_reads_saved_results_and_diagnoses_each_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "tasks.jsonl"
            _write_jsonl(
                jsonl_path,
                [
                    _make_task("t1", "Hello a nonexistentxyz"),  # selection-only failure
                    _make_task("t2", "x"),  # both succeed -- not a selection-only failure
                ],
            )

            results_dir = Path(tmp_dir)
            baseline_jsonl = results_dir / "baseline.jsonl"
            llm_selection_jsonl = results_dir / "llm_selection.jsonl"

            with baseline_jsonl.open("w", encoding="utf-8") as f:
                f.write(json.dumps(_result_record("t1", exact_match=True, ES=1.0, id_f1=1.0)) + "\n")
                f.write(json.dumps(_result_record("t2", exact_match=True, ES=1.0, id_f1=1.0)) + "\n")

            with llm_selection_jsonl.open("w", encoding="utf-8") as f:
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
                    llm_selection_jsonl=str(llm_selection_jsonl),
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
            self.assertEqual(set(counts.keys()), set(CATEGORIES))


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
                    "type": "class",
                    "signature": "class Greeter:",
                    "sources": ["bm25", "symbol"],
                    "selected_by_llm": False,
                    "shared_identifiers": ["default_greeting"],
                    "overlap_ratio": 1.0,
                },
                {
                    "label": "C7",
                    "chunk_id": "def",
                    "file_path": "pkg/module_b.py",
                    "name": "LoudGreeter",
                    "type": "class",
                    "signature": "class LoudGreeter(Greeter):",
                    "sources": ["bm25"],
                    "selected_by_llm": True,
                    "shared_identifiers": [],
                    "overlap_ratio": 0.0,
                },
            ],
            "selected_labels": ["C7"],
            "not_selected_labels": ["C1"],
            "groundtruth": "self.default_greeting",
            "no_selection_completion": "no-sel completion",
            "llm_selection_completion": "llm-sel completion",
            "no_selection_result": {"exact_match": True, "ES": 1.0, "ID-F1": 1.0},
            "llm_selection_result": {"exact_match": False, "ES": 0.5, "ID-F1": 0.5},
            "category": "missed_relevant_candidate",
            "category_reason": "an unselected candidate has higher overlap",
            "category_evidence": {
                "missed_candidate": {
                    "label": "C1", "file_path": "module_a.py", "name": "Greeter",
                    "overlap_ratio": 1.0, "shared_identifiers": ["default_greeting"],
                }
            },
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_diagnosis(diagnosis)
        output = buf.getvalue()
        self.assertIn("task_id: t1", output)
        self.assertIn("C1", output)
        self.assertIn("selected by LLM selection", output)
        self.assertIn("category: missed_relevant_candidate", output)
        self.assertIn("missed candidate: C1", output)

    def test_print_diagnosis_summary_runs_and_lists_all_categories(self):
        from io import StringIO

        diagnoses = [{"category": "missed_relevant_candidate"}, {"category": "unclear"}]
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_diagnosis_summary(diagnoses)
        output = buf.getvalue()
        for category in CATEGORIES:
            self.assertIn(category, output)

    def test_print_diagnosis_isolation_flags_includes_diagnosis_flag(self):
        from io import StringIO

        buf = StringIO()
        with patch("sys.stdout", buf):
            print_diagnosis_isolation_flags()
        output = buf.getvalue()
        self.assertIn("groundtruth_used_for_diagnosis = True", output)
        self.assertIn("groundtruth_used_for_selection = False", output)


if __name__ == "__main__":
    unittest.main()
