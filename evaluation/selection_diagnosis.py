"""Analysis-only diagnosis of the no-selection-vs-LLM-selection quality gap.

Reads the already-saved no-selection baseline and LLM selection (v2)
result files -- never reruns generation and never touches the pipeline.
The only new work done here is re-running RETRIEVAL (BM25 + symbol +
dependency nomination, identical and unmodified) for each task under
investigation, purely to reconstruct the C1..Cn candidate pool for display
-- retrieval only ever sees prompt/target_file, exactly like every other
call site in this project, so groundtruth still never reaches it.

Groundtruth is read here ONLY to classify failures after the fact (never
fed into retrieval, LLM selection, or generation) via a simple, documented
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
DEFAULT_LLM_SELECTION_JSONL = "results/cceval_20_results_v2.jsonl"

# A candidate needs at least this fraction of groundtruth's (non-trivial)
# identifiers present in its own source to count as "meaningfully relevant"
# -- below this, overlap is treated as noise (shared keywords/short names)
# rather than a real signal that the candidate contains what was needed.
MEANINGFUL_OVERLAP_RATIO = 0.34

# "selected_too_many" is only considered at/above this many selected
# candidates -- LLM selection in this study averages ~1.3 selected, so 3+
# is already well outside the normal range.
MANY_SELECTED_THRESHOLD = 3
# ... and only when fewer than this fraction of them show meaningful evidence.
MANY_SELECTED_IRRELEVANT_FRACTION = 0.5

# Filtered out of identifier-overlap comparisons: Python keywords plus a
# few near-universal names that would otherwise make almost every
# candidate look "relevant" to almost every groundtruth.
_STOPWORDS = set(keyword.kwlist) | {"self", "cls", "print"}

CATEGORIES = [
    "missed_relevant_candidate",
    "selected_irrelevant_context",
    "selected_too_few",
    "selected_too_many",
    "relevant_candidate_present_but_generation_failed",
    "candidate_pool_missing_context",
    "unclear",
]


def _load_results_by_task(jsonl_path: Union[str, Path]) -> Dict[str, Dict]:
    records = {}
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[record["task_id"]] = record
    return records


def find_selection_only_failures(baseline_by_task: Dict[str, Dict], llm_selection_by_task: Dict[str, Dict]) -> List[str]:
    """task_ids present (and successful) in both result sets where the
    no-selection baseline got an exact match but LLM selection didn't."""
    task_ids = []
    for task_id, b in baseline_by_task.items():
        s = llm_selection_by_task.get(task_id)
        if s is None or b.get("error") is not None or s.get("error") is not None:
            continue
        if b["exact_match"] is True and s["exact_match"] is False:
            task_ids.append(task_id)
    return task_ids


def _meaningful_identifiers(text: str) -> Set[str]:
    return set(extract_identifiers(text)) - _STOPWORDS


def _categorize(per_candidate: List[Dict], gt_ids: Set[str]) -> Tuple[str, str, Dict]:
    """Returns (category, human-readable reason, structured evidence dict
    used only for printing). Priority order (first match wins) is chosen so
    every task lands in exactly one of the 7 categories in CATEGORIES.
    """
    if not gt_ids:
        return "unclear", "groundtruth has no extractable (non-trivial) identifiers -- heuristic inapplicable", {}

    relevant = [c for c in per_candidate if c["overlap_ratio"] >= MEANINGFUL_OVERLAP_RATIO]
    selected = [c for c in per_candidate if c["selected_by_llm"]]
    unselected = [c for c in per_candidate if not c["selected_by_llm"]]
    unselected_relevant = [c for c in unselected if c["overlap_ratio"] >= MEANINGFUL_OVERLAP_RATIO]
    selected_relevant = [c for c in selected if c["overlap_ratio"] >= MEANINGFUL_OVERLAP_RATIO]

    best_selected = max((c["overlap_ratio"] for c in selected), default=0.0)
    best_unselected = max((c["overlap_ratio"] for c in unselected), default=0.0)
    best_overall = max(best_selected, best_unselected)

    if best_overall == 0.0:
        return (
            "candidate_pool_missing_context",
            "no candidate in the pool shares any non-trivial identifier with groundtruth",
            {},
        )

    if len(relevant) >= 2 and len(selected) < len(relevant):
        missed = sorted(unselected_relevant, key=lambda c: -c["overlap_ratio"])
        return (
            "selected_too_few",
            f"{len(relevant)} candidates show meaningful groundtruth-identifier overlap but only "
            f"{len(selected)} were selected -- multiple pieces of context were likely necessary",
            {"relevant_count": len(relevant), "selected_count": len(selected), "missed_relevant": missed},
        )

    if best_unselected > best_selected:
        pool_for_best = unselected_relevant or unselected
        best_missed = max(pool_for_best, key=lambda c: c["overlap_ratio"])
        return (
            "missed_relevant_candidate",
            f"an unselected candidate ({best_missed['label']}) has higher groundtruth-identifier overlap "
            f"({best_missed['overlap_ratio']:.2f}) than anything selected ({best_selected:.2f})",
            {"missed_candidate": best_missed},
        )

    if len(selected) >= MANY_SELECTED_THRESHOLD and (
        len(selected_relevant) / len(selected) < MANY_SELECTED_IRRELEVANT_FRACTION
    ):
        irrelevant_selected = [c for c in selected if c["overlap_ratio"] < MEANINGFUL_OVERLAP_RATIO]
        return (
            "selected_too_many",
            f"{len(selected)} candidates were selected but only {len(selected_relevant)} show meaningful "
            "groundtruth-identifier overlap -- most selected candidates appear unnecessary",
            {"selected_count": len(selected), "irrelevant_selected": irrelevant_selected},
        )

    if best_selected >= MEANINGFUL_OVERLAP_RATIO:
        relevant_selected = max(selected, key=lambda c: c["overlap_ratio"])
        return (
            "relevant_candidate_present_but_generation_failed",
            f"the selector chose {relevant_selected['label']} (overlap={relevant_selected['overlap_ratio']:.2f}), "
            "which shows meaningful evidence of relevance, yet generation still failed EM -- not a "
            "selection-targeting problem",
            {"relevant_selected": relevant_selected},
        )

    irrelevant_selected = [c for c in selected if c["overlap_ratio"] < MEANINGFUL_OVERLAP_RATIO]
    return (
        "selected_irrelevant_context",
        f"only weak overlap exists anywhere in the pool (best={best_overall:.2f}, below the "
        f"{MEANINGFUL_OVERLAP_RATIO} meaningful-overlap threshold) -- the selected candidate(s) aren't "
        "meaningfully better than anything else available",
        {"irrelevant_selected": irrelevant_selected},
    )


def diagnose_task(
    task_id: str,
    baseline_record: Dict,
    llm_selection_record: Dict,
    jsonl_path: str = DEFAULT_JSONL,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Dict:
    """Re-runs retrieval ONLY (no selection, no generation -- LLM
    selection's saved choice from the v2 run is reused as-is) to
    reconstruct the candidate pool with C1..Cn labels, then classifies why
    the LLM-selection run failed EM where the no-selection baseline
    succeeded. Only example["prompt"]/["file"] ever reach retrieval;
    groundtruth is read here only for the post-hoc identifier-overlap
    classification.
    """
    index = find_example_index_by_task_id(jsonl_path, task_id)
    example = load_cceval_example(jsonl_path, index)
    chunks = locate_repo_index(example["repository"], repos_dir=repos_dir, index_dir=index_dir)
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    pipeline = CandidatePipeline(BM25Retriever(chunks), SymbolRetriever(chunks), DependencyRetriever(chunks))
    candidates = pipeline.nominate(example["prompt"], target_file=example["file"])
    labels = assign_candidate_labels(candidates)
    label_to_chunk_id = {label: chunk_id for chunk_id, label in labels.items()}

    llm_selected_labels = llm_selection_record.get("selected_candidate_ids") or []
    llm_selected_chunk_ids = {label_to_chunk_id[l] for l in llm_selected_labels if l in label_to_chunk_id}

    gt_ids = _meaningful_identifiers(example["groundtruth"])

    per_candidate = []
    for candidate in candidates:
        chunk_id = candidate["chunk_id"]
        chunk = chunk_lookup.get(chunk_id, {})
        source_code = chunk.get("source_code", "")
        cand_ids = _meaningful_identifiers(source_code)
        shared = sorted(cand_ids & gt_ids)
        ratio = (len(shared) / len(gt_ids)) if gt_ids else 0.0
        per_candidate.append(
            {
                "label": labels[chunk_id],
                "chunk_id": chunk_id,
                "file_path": candidate["file_path"],
                "name": candidate["name"],
                "type": chunk.get("type", "?"),
                "signature": chunk.get("signature", "?"),
                "sources": candidate["sources"],
                "selected_by_llm": chunk_id in llm_selected_chunk_ids,
                "shared_identifiers": shared,
                "overlap_ratio": ratio,
            }
        )

    category, reason, evidence = _categorize(per_candidate, gt_ids)

    selected_labels = [c["label"] for c in per_candidate if c["selected_by_llm"]]
    not_selected_labels = [c["label"] for c in per_candidate if not c["selected_by_llm"]]

    return {
        "task_id": task_id,
        "candidates": per_candidate,
        "selected_labels": selected_labels,
        "not_selected_labels": not_selected_labels,
        "groundtruth": example["groundtruth"],
        "no_selection_completion": baseline_record["completion"],
        "llm_selection_completion": llm_selection_record["completion"],
        "no_selection_result": {
            "exact_match": baseline_record["exact_match"],
            "ES": baseline_record["ES"],
            "ID-F1": baseline_record["ID-F1"],
        },
        "llm_selection_result": {
            "exact_match": llm_selection_record["exact_match"],
            "ES": llm_selection_record["ES"],
            "ID-F1": llm_selection_record["ID-F1"],
        },
        "category": category,
        "category_reason": reason,
        "category_evidence": evidence,
    }


def run_selection_diagnosis(
    baseline_jsonl: str = DEFAULT_BASELINE_JSONL,
    llm_selection_jsonl: str = DEFAULT_LLM_SELECTION_JSONL,
    jsonl_path: str = DEFAULT_JSONL,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> List[Dict]:
    """Finds every task where the no-selection baseline got EM=True and LLM
    selection got EM=False (both already-saved, successful runs), and
    returns a diagnosis for each. Does not modify or rerun any part of
    either experiment.
    """
    baseline_by_task = _load_results_by_task(baseline_jsonl)
    llm_selection_by_task = _load_results_by_task(llm_selection_jsonl)
    failing_task_ids = find_selection_only_failures(baseline_by_task, llm_selection_by_task)

    return [
        diagnose_task(
            task_id,
            baseline_by_task[task_id],
            llm_selection_by_task[task_id],
            jsonl_path=jsonl_path,
            repos_dir=repos_dir,
            index_dir=index_dir,
        )
        for task_id in failing_task_ids
    ]


def summarize_diagnosis_categories(diagnoses: List[Dict]) -> Dict[str, int]:
    counts = {category: 0 for category in CATEGORIES}
    for d in diagnoses:
        counts[d["category"]] = counts.get(d["category"], 0) + 1
    return counts


def print_diagnosis_isolation_flags() -> None:
    print("=== Data isolation verification (LLM selection failure diagnosis) ===")
    print("groundtruth_used_for_retrieval = False")
    print("groundtruth_used_for_selection = False")
    print("groundtruth_used_for_generation = False")
    print("groundtruth_used_for_evaluation = True")
    print("groundtruth_used_for_diagnosis = True")


def print_diagnosis(diagnosis: Dict) -> None:
    print(f"task_id: {diagnosis['task_id']}")
    print()
    print("candidate pool:")
    for c in diagnosis["candidates"]:
        marker = "  <- selected by LLM selection" if c["selected_by_llm"] else ""
        overlap = f"overlap={c['overlap_ratio']:.2f}"
        if c["shared_identifiers"]:
            overlap += f" shared={c['shared_identifiers']}"
        symbol = f"{c['file_path']}::{c['name']}"
        print(f"  {c['label']:4s} {symbol:45s} type={c['type']:<10} [{','.join(c['sources'])}]  {overlap}{marker}")
    print()

    print(f"selected by LLM selection: {', '.join(diagnosis['selected_labels']) if diagnosis['selected_labels'] else '(none)'}")
    print(f"not selected:              {', '.join(diagnosis['not_selected_labels']) if diagnosis['not_selected_labels'] else '(none)'}")
    print()

    print(f"groundtruth completion:   {diagnosis['groundtruth']!r}")
    print(f"no-selection completion:  {diagnosis['no_selection_completion']!r}")
    print(f"LLM-selection completion: {diagnosis['llm_selection_completion']!r}")
    print()

    b = diagnosis["no_selection_result"]
    s = diagnosis["llm_selection_result"]
    print(f"no-selection result:    EM={b['exact_match']}  ES={b['ES']:.3f}  ID-F1={b['ID-F1']:.3f}")
    print(f"LLM-selection result:   EM={s['exact_match']}  ES={s['ES']:.3f}  ID-F1={s['ID-F1']:.3f}")
    print()

    print(f"category: {diagnosis['category']}")
    print(f"reason:   {diagnosis['category_reason']}")

    evidence = diagnosis.get("category_evidence") or {}
    if "missed_candidate" in evidence:
        mc = evidence["missed_candidate"]
        print(
            f"  missed candidate: {mc['label']}  file={mc['file_path']}::{mc['name']}  "
            f"overlap={mc['overlap_ratio']:.2f}  shared={mc['shared_identifiers']}"
        )
    if "missed_relevant" in evidence:
        for mc in evidence["missed_relevant"]:
            print(
                f"  missed relevant candidate: {mc['label']}  file={mc['file_path']}::{mc['name']}  "
                f"overlap={mc['overlap_ratio']:.2f}"
            )
    if "irrelevant_selected" in evidence:
        for ic in evidence["irrelevant_selected"]:
            print(
                f"  irrelevant selected candidate: {ic['label']}  file={ic['file_path']}::{ic['name']}  "
                f"overlap={ic['overlap_ratio']:.2f} (below meaningful threshold)"
            )
    if "relevant_selected" in evidence:
        rs = evidence["relevant_selected"]
        print(
            f"  relevant selected candidate: {rs['label']}  file={rs['file_path']}::{rs['name']}  "
            f"overlap={rs['overlap_ratio']:.2f}"
        )


def print_diagnosis_summary(diagnoses: List[Dict]) -> None:
    counts = summarize_diagnosis_categories(diagnoses)
    print("=== LLM Selection Failure Diagnosis ===")
    print()
    for category in CATEGORIES:
        print(f"{category:<50s} {counts[category]}")
