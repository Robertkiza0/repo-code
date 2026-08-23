"""random2_selection baseline: identical retrieval (BM25 + symbol +
dependency, deduplicated, capped) and generation as the other CCEval
experiments, but selection is a fixed policy -- exactly 2 candidates
chosen uniformly at random from the existing, unmodified candidate pool,
using a fixed seed so the run is reproducible. No LLM is involved at all.

Used alongside top2_selection to determine whether LLM selection's
improvement over the no-selection baseline comes from intelligent
candidate choice, or simply from shrinking the context to ~2 candidates.
"""
import random
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

DEFAULT_RESULTS_DIR = "results"
RANDOM_N = 2
# Reused unchanged for every task -- a fresh random.Random(RANDOM_SEED) is
# constructed per task (see run_one_example_random2) rather than one shared
# generator advancing across tasks, so each task's selection is
# independently reproducible from this single fixed seed.
RANDOM_SEED = 42

_EMPTY_RANDOM2_FIELDS = {
    "candidate_count": None,
    "selected_count": None,
    "selected_candidate_ids": None,
    "completion": None,
    "groundtruth": None,
    "exact_match": None,
    "ES": None,
    "ID-F1": None,
    "generation_time": None,
}


def run_one_example_random2(
    generator: CompletionGenerator,
    jsonl_path: str = DEFAULT_JSONL,
    index: int = 0,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    seed: int = RANDOM_SEED,
) -> Dict:
    """Same retrieval as the other experiments -- selection is exactly
    RANDOM_N candidates chosen uniformly at random (seeded, reproducible)
    from the unmodified candidate pool, no LLM call. If fewer than RANDOM_N
    candidates were nominated, all of them are selected. The candidate pool
    itself is never reordered before sampling, and the chosen candidates
    are returned in their original pool order (only *which* ones are
    picked is randomized). Only example["prompt"]/["file"] ever reach
    retrieval or generation -- groundtruth is extracted for evaluation
    only, never passed to either.
    """
    example = load_cceval_example(jsonl_path, index)
    chunks = locate_repo_index(example["repository"], repos_dir=repos_dir, index_dir=index_dir)

    pipeline = CandidatePipeline(BM25Retriever(chunks), SymbolRetriever(chunks), DependencyRetriever(chunks))
    candidates = pipeline.nominate(example["prompt"], target_file=example["file"])

    rng = random.Random(seed)
    k = min(RANDOM_N, len(candidates))
    chosen_indices = sorted(rng.sample(range(len(candidates)), k))
    selected_ids = [candidates[i]["chunk_id"] for i in chosen_indices]

    generated = generator.generate(example["prompt"], example["file"], selected_ids)

    return {
        "task_id": example["task_id"],
        "repository": example["repository"],
        "target_file": example["file"],
        "prompt": example["prompt"],
        "groundtruth": example["groundtruth"],
        "num_candidates": len(candidates),
        "candidates": candidates,
        "selected_chunk_ids": selected_ids,
        "completion": generated["completion"],
    }


def _run_single_task_random2(
    generator: CompletionGenerator, jsonl_path: str, index: int, repos_dir: str, index_dir: str, seed: int
) -> Dict:
    example = load_cceval_example(jsonl_path, index)
    record = {"task_id": example["task_id"]}

    try:
        t0 = time.time()
        result = run_one_example_random2(
            generator, jsonl_path=jsonl_path, index=index, repos_dir=repos_dir, index_dir=index_dir, seed=seed
        )
        generation_time = time.time() - t0

        record.update(
            {
                "candidate_count": result["num_candidates"],
                "selected_count": len(result["selected_chunk_ids"]),
                "selected_candidate_ids": result["selected_chunk_ids"],
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
        record.update(_EMPTY_RANDOM2_FIELDS)
        record["error"] = f"{type(e).__name__}: {e}"

    return record


def summarize_random2(task_results: List[Dict]) -> Dict:
    successful = [r for r in task_results if r["error"] is None]
    failed = [r for r in task_results if r["error"] is not None]

    def avg(key: str) -> Optional[float]:
        values = [r[key] for r in successful if r[key] is not None]
        return sum(values) / len(values) if values else None

    exact_matches = [r["exact_match"] for r in successful]

    return {
        "total_tasks": len(task_results),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "failed_task_ids": [r["task_id"] for r in failed],
        "avg_candidate_count": avg("candidate_count"),
        "avg_selected_count": avg("selected_count"),
        "exact_match_rate": (sum(1 for m in exact_matches if m) / len(exact_matches)) if exact_matches else None,
        "avg_ES": avg("ES"),
        "avg_ID_F1": avg("ID-F1"),
        "avg_generation_time": avg("generation_time"),
    }


def run_random2_experiment(
    generator: CompletionGenerator,
    n_tasks: int = 20,
    jsonl_path: str = DEFAULT_JSONL,
    results_dir: str = DEFAULT_RESULTS_DIR,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    seed: int = RANDOM_SEED,
) -> Dict:
    """Runs the random2_selection baseline across n_tasks, using the same
    fixed seed for every task. Saves
    results/cceval_{n_tasks}_random2_selection.jsonl and
    results/cceval_{n_tasks}_random2_selection_summary.json; never touches
    any other experiment's own result files.
    """
    preflight_check(jsonl_path, n_tasks)

    task_results = [
        _run_single_task_random2(generator, jsonl_path, index, repos_dir, index_dir, seed) for index in range(n_tasks)
    ]
    summary = summarize_random2(task_results)

    results_dir_path = Path(results_dir)
    save_results_jsonl(task_results, results_dir_path / f"cceval_{n_tasks}_random2_selection.jsonl")
    save_summary_json(summary, results_dir_path / f"cceval_{n_tasks}_random2_selection_summary.json")

    return {"results": task_results, "summary": summary}


def print_random2_isolation_flags() -> None:
    print("=== Data isolation verification (random2_selection) ===")
    print("groundtruth_used_for_retrieval = False")
    print("groundtruth_used_for_selection = False")
    print("groundtruth_used_for_generation = False")
    print("groundtruth_used_for_evaluation = True")


def _fmt(value, spec: str = "{:.3f}") -> str:
    return spec.format(value) if value is not None else "n/a"


def print_random2_summary_table(summary: Dict) -> None:
    print("=== CCEval random2_selection summary ===")
    print(f"  total tasks:          {summary['total_tasks']}")
    print(f"  successful:           {summary['num_successful']}")
    print(f"  failed:               {summary['num_failed']}")
    if summary["failed_task_ids"]:
        print(f"  failed task_ids:      {summary['failed_task_ids']}")
    print(f"  avg candidate count:  {_fmt(summary['avg_candidate_count'], '{:.2f}')}")
    print(f"  avg selected count:   {_fmt(summary['avg_selected_count'], '{:.2f}')}")
    print(f"  exact match rate:     {_fmt(summary['exact_match_rate'])}")
    print(f"  avg ES:               {_fmt(summary['avg_ES'])}")
    print(f"  avg ID-F1:            {_fmt(summary['avg_ID_F1'])}")
    print(f"  avg generation time:  {_fmt(summary['avg_generation_time'], '{:.2f}')}s")


def print_random2_task_table(task_results: List[Dict]) -> None:
    header = (
        f"{'task_id':<24} | {'candidates':>10} | {'selected':>8} | "
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
            f"{r['task_id']:<24} | {r['candidate_count']:>10} | {r['selected_count']:>8} | "
            f"{em:>5} | {r['ES']:.3f} | {r['ID-F1']:.3f} | {r['generation_time']:.2f}s"
        )
