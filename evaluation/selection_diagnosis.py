"""Analysis-only diagnosis of the no-selection-vs-Qwen-selection quality gap.

Reads the already-saved no-selection baseline and Qwen-selection (v2)
result files -- never reruns generation and never touches the pipeline.
The only new work done here is re-running RETRIEVAL (BM25 + symbol +
dependency nomination, identical and unmodified) for each task under
investigation, purely to reconstruct the C1..Cn candidate pool for display
-- retrieval only ever sees prompt/target_file, exactly like every other
call site in this project, so groundtruth still never reaches it.

Groundtruth is read here ONLY to classify failures after the fact (never
fed into retrieval, Qwen selection, or generation) via a simple, documented
heuristic: does a candidate's source code share identifiers with the
groundtruth completion? This is a heuristic aid for triage, not a claim of
ground truth about "relevance" -- the raw per-candidate overlap is always
printed alongside the auto-assigned category so a human can judge for
themselves.
"""
import json
import keyword
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

from evaluation.cceval_adapter import (
    DEFAULT_INDEX_DIR,
    DEFAULT_JSONL,
    DEFAULT_REPOS_DIR,
    assign_candidate_labels,
    find_example_index_by_task_id,
    load_cceval_example,
    locate_repo_index,
)
from evaluation.metrics import extract_identifiers
from retrieval.bm25_retriever import BM25Retriever
from retrieval.candidate_pipeline import CandidatePipeline
from retrieval.dependency_retriever import DependencyRetriever
from retrieval.symbol_retriever import SymbolRetriever

DEFAULT_BASELINE_JSONL = "results/cceval_20_baseline.jsonl"
DEFAULT_QWEN_JSONL = "results/cceval_20_results_v2.jsonl"

# A candidate needs at least this fraction of groundtruth's (non-trivial)
# identifiers present in its own source to count as "meaningfully relevant"
# -- below this, overlap is treated as noise (shared keywords/short names)
# rather than a real signal that the candidate contains what was needed.
MEANINGFUL_OVERLAP_RATIO = 0.34

# Filtered out of identifier-overlap comparisons: Python keywords plus a
# few near-universal names that would otherwise make almost every
# candidate look "relevant" to almost every groundtruth.
_STOPWORDS = set(keyword.kwlist) | {"self", "cls", "print"}


def _load_results_by_task(jsonl_path: Union[str, Path]) -> Dict[str, Dict]:
    records = {}
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[record["task_id"]] = record
    return records


def find_selection_only_failures(baseline_by_task: Dict[str, Dict], qwen_by_task: Dict[str, Dict]) -> List[str]:
    """task_ids present (and successful) in both result sets where the
    no-selection baseline got an exact match but Qwen selection didn't."""
    task_ids = []
    for task_id, b in baseline_by_task.items():
        q = qwen_by_task.get(task_id)
        if q is None or b.get("error") is not None or q.get("error") is not None:
            continue
        if b["exact_match"] is True and q["exact_match"] is False:
            task_ids.append(task_id)
    return task_ids


def _meaningful_identifiers(text: str) -> Set[str]:
    return set(extract_identifiers(text)) - _STOPWORDS


def _categorize(per_candidate: List[Dict], gt_ids: Set[str]) -> Tuple[str, str]:
    if not gt_ids:
        return "unclear", "groundtruth has no extractable (non-trivial) identifiers -- heuristic inapplicable"

    selected_ratios = [c["overlap_ratio"] for c in per_candidate if c["selected_by_qwen"]]
    unselected_ratios = [c["overlap_ratio"] for c in per_candidate if not c["selected_by_qwen"]]
    best_selected = max(selected_ratios) if selected_ratios else 0.0
    best_unselected = max(unselected_ratios) if unselected_ratios else 0.0
    best_overall = max(best_selected, best_unselected)

    if best_overall == 0.0:
        return (
            "candidate_pool_missing_context",
            "no candidate in the pool shares any non-trivial identifier with groundtruth",
        )
    if best_unselected > best_selected:
        return (
            "missed_relevant_candidate",
            f"an unselected candidate has higher groundtruth-identifier overlap ({best_unselected:.2f}) "
            f"than anything Qwen selected ({best_selected:.2f})",
        )
    if best_selected >= MEANINGFUL_OVERLAP_RATIO:
        return (
            "unclear",
            f"Qwen selected a candidate with meaningful overlap ({best_selected:.2f}) yet generation "
            "still failed EM -- likely not a selection problem",
        )
    return (
        "selected_irrelevant_context",
        f"only weak overlap exists anywhere in the pool (best={best_overall:.2f}, below the "
        f"{MEANINGFUL_OVERLAP_RATIO} meaningful-overlap threshold) -- Qwen's picks aren't meaningfully "
        "better than anything else available",
    )


def diagnose_task(
    task_id: str,
    baseline_record: Dict,
    qwen_record: Dict,
    jsonl_path: str = DEFAULT_JSONL,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Dict:
    """Re-runs retrieval ONLY (no selection, no generation -- Qwen's saved
    selection from the v2 run is reused as-is) to reconstruct the candidate
    pool with C1..Cn labels, then classifies why the Qwen-selection run
    failed EM where the no-selection baseline succeeded. Only
    example["prompt"]/["file"] ever reach retrieval; groundtruth is read
    here only for the post-hoc identifier-overlap classification.
    """
    index = find_example_index_by_task_id(jsonl_path, task_id)
    example = load_cceval_example(jsonl_path, index)
    chunks = locate_repo_index(example["repository"], repos_dir=repos_dir, index_dir=index_dir)
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    pipeline = CandidatePipeline(BM25Retriever(chunks), SymbolRetriever(chunks), DependencyRetriever(chunks))
    candidates = pipeline.nominate(example["prompt"], target_file=example["file"])
    labels = assign_candidate_labels(candidates)
    label_to_chunk_id = {label: chunk_id for chunk_id, label in labels.items()}

    qwen_selected_labels = qwen_record.get("selected_candidate_ids") or []
    qwen_selected_chunk_ids = {label_to_chunk_id[l] for l in qwen_selected_labels if l in label_to_chunk_id}

    gt_ids = _meaningful_identifiers(example["groundtruth"])

    per_candidate = []
    for candidate in candidates:
        chunk_id = candidate["chunk_id"]
        source_code = chunk_lookup.get(chunk_id, {}).get("source_code", "")
        cand_ids = _meaningful_identifiers(source_code)
        shared = sorted(cand_ids & gt_ids)
        ratio = (len(shared) / len(gt_ids)) if gt_ids else 0.0
        per_candidate.append(
            {
                "label": labels[chunk_id],
                "chunk_id": chunk_id,
                "file_path": candidate["file_path"],
                "name": candidate["name"],
                "sources": candidate["sources"],
                "selected_by_qwen": chunk_id in qwen_selected_chunk_ids,
                "shared_identifiers": shared,
                "overlap_ratio": ratio,
            }
        )

    category, reason = _categorize(per_candidate, gt_ids)

    return {
        "task_id": task_id,
        "candidates": per_candidate,
        "qwen_selected_labels": qwen_selected_labels,
        "no_selection_result": {
            "exact_match": baseline_record["exact_match"],
            "ES": baseline_record["ES"],
            "ID-F1": baseline_record["ID-F1"],
        },
        "qwen_selection_result": {
            "exact_match": qwen_record["exact_match"],
            "ES": qwen_record["ES"],
            "ID-F1": qwen_record["ID-F1"],
        },
        "category": category,
        "category_reason": reason,
    }


def run_selection_diagnosis(
    baseline_jsonl: str = DEFAULT_BASELINE_JSONL,
    qwen_jsonl: str = DEFAULT_QWEN_JSONL,
    jsonl_path: str = DEFAULT_JSONL,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> List[Dict]:
    """Finds every task where the no-selection baseline got EM=True and
    Qwen selection got EM=False (both already-saved, successful runs), and
    returns a diagnosis for each. Does not modify or rerun any part of
    either experiment.
    """
    baseline_by_task = _load_results_by_task(baseline_jsonl)
    qwen_by_task = _load_results_by_task(qwen_jsonl)
    failing_task_ids = find_selection_only_failures(baseline_by_task, qwen_by_task)

    return [
        diagnose_task(
            task_id,
            baseline_by_task[task_id],
            qwen_by_task[task_id],
            jsonl_path=jsonl_path,
            repos_dir=repos_dir,
            index_dir=index_dir,
        )
        for task_id in failing_task_ids
    ]


def summarize_diagnosis_categories(diagnoses: List[Dict]) -> Dict[str, int]:
    counts = {
        "missed_relevant_candidate": 0,
        "selected_irrelevant_context": 0,
        "candidate_pool_missing_context": 0,
        "unclear": 0,
    }
    for d in diagnoses:
        counts[d["category"]] = counts.get(d["category"], 0) + 1
    return counts


def print_diagnosis(diagnosis: Dict) -> None:
    print(f"task_id: {diagnosis['task_id']}")
    print()
    print("candidate pool:")
    for c in diagnosis["candidates"]:
        marker = "  <- selected by Qwen" if c["selected_by_qwen"] else ""
        overlap = f"overlap={c['overlap_ratio']:.2f}"
        if c["shared_identifiers"]:
            overlap += f" shared={c['shared_identifiers']}"
        symbol = f"{c['file_path']}::{c['name']}"
        print(f"  {c['label']:4s} {symbol:45s} [{','.join(c['sources'])}]  {overlap}{marker}")
    print()

    selected = diagnosis["qwen_selected_labels"]
    print(f"Qwen selected: {', '.join(selected) if selected else '(none)'}")
    print()

    b = diagnosis["no_selection_result"]
    q = diagnosis["qwen_selection_result"]
    print(f"no-selection result:    EM={b['exact_match']}  ES={b['ES']:.3f}  ID-F1={b['ID-F1']:.3f}")
    print(f"qwen-selection result:  EM={q['exact_match']}  ES={q['ES']:.3f}  ID-F1={q['ID-F1']:.3f}")
    print()
    print(f"category: {diagnosis['category']}")
    print(f"reason:   {diagnosis['category_reason']}")


def print_diagnosis_summary(diagnoses: List[Dict]) -> None:
    counts = summarize_diagnosis_categories(diagnoses)
    print(f"=== Selection diagnosis summary ({len(diagnoses)} task(s) analyzed) ===")
    for category, count in counts.items():
        print(f"  {category:32s} {count}")
