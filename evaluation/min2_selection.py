"""min2_selection ablation: identical retrieval, candidate nomination, Qwen
selection prompt, candidate ordering, StarCoder generation, and evaluation
as the existing selection experiment. The only change: after Qwen produces
its (validated, hallucination-filtered) selection, if it selected fewer
than MIN_SELECTED_CANDIDATES, the remainder is filled using the
next-highest-ranked candidates from the ORIGINAL candidate pool (the same
order CandidatePipeline.nominate() already produced) -- no new ranking or
retrieval logic is introduced.

Tests whether the current selector is simply too aggressive (recoverable by
a floor on selection size) or choosing the wrong chunks outright.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.cceval_adapter import (
    DEFAULT_INDEX_DIR,
    DEFAULT_JSONL,
    DEFAULT_REPOS_DIR,
    load_cceval_example,
    locate_repo_index,
)
from evaluation.experiment import preflight_check, save_results_jsonl, save_summary_json
from evaluation.metrics import edit_similarity, exact_match, identifier_f1
from generation.generator import CompletionGenerator
from retrieval.bm25_retriever import BM25Retriever
from retrieval.candidate_pipeline import CandidatePipeline
from retrieval.dependency_retriever import DependencyRetriever
from retrieval.symbol_retriever import SymbolRetriever
from selection.llm_selector import LLMSelector

DEFAULT_RESULTS_DIR = "results"
MIN_SELECTED_CANDIDATES = 2

_EMPTY_MIN2_FIELDS = {
    "candidate_count": None,
    "qwen_selected_ids": None,
    "final_selected_ids": None,
    "final_selected_count": None,
    "completion": None,
    "groundtruth": None,
    "exact_match": None,
    "ES": None,
    "ID-F1": None,
    "generation_time": None,
}


def _fill_to_minimum(selected_ids: List[str], candidates: List[Dict], min_count: int) -> List[str]:
    """Fills selected_ids up to min_count using candidates in their existing
    ranked order (most nominating sources first, highest score as tiebreak)
    -- never re-ranks or re-retrieves. Ids already selected are skipped."""
    final_ids = list(selected_ids)
    for candidate in candidates:
        if len(final_ids) >= min_count:
            break
        chunk_id = candidate["chunk_id"]
        if chunk_id not in final_ids:
            final_ids.append(chunk_id)
    return final_ids


def run_one_example_min2(
    selector: LLMSelector,
    generator: CompletionGenerator,
    jsonl_path: str = DEFAULT_JSONL,
    index: int = 0,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Dict:
    """Same retrieval + Qwen selection as run_one_example(), but if Qwen
    selects fewer than MIN_SELECTED_CANDIDATES, the shortfall is filled from
    the original ranked candidate pool before StarCoder generates. Only
    example["prompt"]/["file"] ever reach retrieval, selection, or
    generation -- groundtruth is extracted for evaluation only, never
    passed to any of them.
    """
    example = load_cceval_example(jsonl_path, index)
    chunks = locate_repo_index(example["repository"], repos_dir=repos_dir, index_dir=index_dir)

    pipeline = CandidatePipeline(BM25Retriever(chunks), SymbolRetriever(chunks), DependencyRetriever(chunks))
    candidates = pipeline.nominate(example["prompt"], target_file=example["file"])

    selection = selector.select(example["prompt"], example["file"], candidates)
    qwen_selected_ids = selection["selected_chunk_ids"]

    if len(qwen_selected_ids) >= MIN_SELECTED_CANDIDATES:
        final_selected_ids = list(qwen_selected_ids)
    else:
        final_selected_ids = _fill_to_minimum(qwen_selected_ids, candidates, MIN_SELECTED_CANDIDATES)

    generated = generator.generate(example["prompt"], example["file"], final_selected_ids)

    return {
        "task_id": example["task_id"],
        "repository": example["repository"],
        "target_file": example["file"],
        "prompt": example["prompt"],
        "groundtruth": example["groundtruth"],
        "num_candidates": len(candidates),
        "candidates": candidates,
        "qwen_selected_ids": qwen_selected_ids,
        "final_selected_ids": final_selected_ids,
        "raw_response": selection["raw_response"],
        "parse_status": selection["parse_status"],
        "completion": generated["completion"],
    }


def _run_single_task_min2(
    selector: LLMSelector,
    generator: CompletionGenerator,
    jsonl_path: str,
    index: int,
    repos_dir: str,
    index_dir: str,
) -> Dict:
    example = load_cceval_example(jsonl_path, index)
    record = {"task_id": example["task_id"]}

    try:
        t0 = time.time()
        result = run_one_example_min2(
            selector, generator, jsonl_path=jsonl_path, index=index, repos_dir=repos_dir, index_dir=index_dir
        )
        generation_time = time.time() - t0

        record.update(
            {
                "candidate_count": result["num_candidates"],
                "qwen_selected_ids": result["qwen_selected_ids"],
                "final_selected_ids": result["final_selected_ids"],
                "final_selected_count": len(result["final_selected_ids"]),
                "completion": result["completion"],
                "groundtruth": result["groundtruth"],
                "exact_match": exact_match(result["completion"], result["groundtruth"]),
                "ES": edit_similarity(result["completion"], result["groundtruth"]),
                "ID-F1": identifier_f1(result["completion"], result["groundtruth"]),
                "generation_time": generation_time,
                "error": None,
            }
        )
    except Exception as e:  # noqa: BLE001 -- one task's failure must not stop the batch
        record.update(_EMPTY_MIN2_FIELDS)
        record["error"] = f"{type(e).__name__}: {e}"

    return record


def summarize_min2(task_results: List[Dict]) -> Dict:
    successful = [r for r in task_results if r["error"] is None]
    failed = [r for r in task_results if r["error"] is not None]

    def avg(key: str) -> Optional[float]:
        values = [r[key] for r in successful if r[key] is not None]
        return sum(values) / len(values) if values else None

    exact_matches = [r["exact_match"] for r in successful]
    num_filled = sum(1 for r in successful if len(r["qwen_selected_ids"]) < MIN_SELECTED_CANDIDATES)

    return {
        "total_tasks": len(task_results),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "failed_task_ids": [r["task_id"] for r in failed],
        "avg_candidate_count": avg("candidate_count"),
        "avg_final_selected_count": avg("final_selected_count"),
        "num_tasks_filled_to_minimum": num_filled,
        "exact_match_rate": (sum(1 for m in exact_matches if m) / len(exact_matches)) if exact_matches else None,
        "avg_ES": avg("ES"),
        "avg_ID_F1": avg("ID-F1"),
        "avg_generation_time": avg("generation_time"),
    }


def run_min2_selection_experiment(
    selector: LLMSelector,
    generator: CompletionGenerator,
    n_tasks: int = 20,
    jsonl_path: str = DEFAULT_JSONL,
    results_dir: str = DEFAULT_RESULTS_DIR,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Dict:
    """Runs the min2_selection ablation across n_tasks. Saves
    results/cceval_{n_tasks}_min2_selection.jsonl and
    results/cceval_{n_tasks}_min2_selection_summary.json; never touches the
    no-selection baseline's or the plain Qwen-selection experiment's own
    result files.
    """
    preflight_check(jsonl_path, n_tasks)

    task_results = [
        _run_single_task_min2(selector, generator, jsonl_path, index, repos_dir, index_dir)
        for index in range(n_tasks)
    ]
    summary = summarize_min2(task_results)

    results_dir_path = Path(results_dir)
    save_results_jsonl(task_results, results_dir_path / f"cceval_{n_tasks}_min2_selection.jsonl")
    save_summary_json(summary, results_dir_path / f"cceval_{n_tasks}_min2_selection_summary.json")

    return {"results": task_results, "summary": summary}


def print_min2_isolation_flags() -> None:
    print("=== Data isolation verification (min2_selection ablation) ===")
    print("groundtruth_used_for_retrieval = False")
    print("groundtruth_used_for_selection = False")
    print("groundtruth_used_for_generation = False")
    print("groundtruth_used_for_evaluation = True")


def _fmt(value, spec: str = "{:.3f}") -> str:
    return spec.format(value) if value is not None else "n/a"


def print_min2_summary_table(summary: Dict) -> None:
    print("=== CCEval min2_selection ablation summary ===")
    print(f"  total tasks:            {summary['total_tasks']}")
    print(f"  successful:             {summary['num_successful']}")
    print(f"  failed:                 {summary['num_failed']}")
    if summary["failed_task_ids"]:
        print(f"  failed task_ids:        {summary['failed_task_ids']}")
    print(f"  avg candidate count:    {_fmt(summary['avg_candidate_count'], '{:.2f}')}")
    print(f"  avg final selected:     {_fmt(summary['avg_final_selected_count'], '{:.2f}')}")
    print(f"  tasks filled to min={MIN_SELECTED_CANDIDATES}:  {summary['num_tasks_filled_to_minimum']}")
    print(f"  exact match rate:       {_fmt(summary['exact_match_rate'])}")
    print(f"  avg ES:                 {_fmt(summary['avg_ES'])}")
    print(f"  avg ID-F1:              {_fmt(summary['avg_ID_F1'])}")
    print(f"  avg generation time:    {_fmt(summary['avg_generation_time'], '{:.2f}')}s")


def print_min2_task_table(task_results: List[Dict]) -> None:
    header = (
        f"{'task_id':<24} | {'cand':>4} | {'qwen_sel':>8} | {'final_sel':>9} | "
        f"{'EM':>5} | {'ES':>6} | {'ID-F1':>6} | {'time':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in task_results:
        if r["error"] is not None:
            print(f"{r['task_id']:<24} | ERROR: {r['error']}")
            continue
        em = "True" if r["exact_match"] else "False"
        print(
            f"{r['task_id']:<24} | {r['candidate_count']:>4} | {len(r['qwen_selected_ids']):>8} | "
            f"{r['final_selected_count']:>9} | {em:>5} | {r['ES']:.3f} | {r['ID-F1']:.3f} | {r['generation_time']:.2f}s"
        )


def print_three_way_comparison(baseline_summary: Dict, qwen_summary: Dict, min2_summary: Dict) -> None:
    """Prints the no-selection baseline, plain Qwen selection, and
    min2_selection ablation summaries side by side for the metrics the
    research question turns on."""
    header = f"{'metric':<20} | {'no selection':>13} | {'qwen selection':>15} | {'min2_selection':>15}"
    print(header)
    print("-" * len(header))

    def row(label, b, q, m, spec="{:.3f}"):
        print(f"{label:<20} | {_fmt(b, spec):>13} | {_fmt(q, spec):>15} | {_fmt(m, spec):>15}")

    row("avg candidates", baseline_summary.get("avg_candidate_count"), qwen_summary.get("avg_candidate_count"), min2_summary.get("avg_candidate_count"), "{:.2f}")
    row("avg selected", None, qwen_summary.get("avg_selected_count"), min2_summary.get("avg_final_selected_count"), "{:.2f}")
    row("exact match", baseline_summary.get("exact_match_rate"), qwen_summary.get("exact_match_rate"), min2_summary.get("exact_match_rate"))
    row("ES", baseline_summary.get("avg_ES"), qwen_summary.get("avg_ES"), min2_summary.get("avg_ES"))
    row("ID-F1", baseline_summary.get("avg_ID_F1"), qwen_summary.get("avg_ID_F1"), min2_summary.get("avg_ID_F1"))
    row("avg time (s)", baseline_summary.get("avg_generation_time"), qwen_summary.get("avg_generation_time"), min2_summary.get("avg_generation_time"), "{:.2f}")
