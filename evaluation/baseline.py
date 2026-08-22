"""No-selection baseline: identical retrieval (BM25 + symbol + dependency,
deduplicated, capped) and generation as the Qwen-selection experiment, but
skips selection entirely and hands the generator EVERY nominated candidate.
LLMSelector is never constructed or called here.

Used to establish whether Qwen's selection actually helps completion quality
compared to just giving StarCoder everything retrieval found -- nothing here
modifies retrieval, candidate nomination, generation, or evaluation; this
module only orchestrates them differently (no selection step in between).
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

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

_EMPTY_BASELINE_FIELDS = {
    "candidate_count": None,
    "completion": None,
    "groundtruth": None,
    "exact_match": None,
    "ES": None,
    "ID-F1": None,
    "generation_time": None,
}


def run_one_example_baseline(
    generator: CompletionGenerator,
    jsonl_path: str = DEFAULT_JSONL,
    index: int = 0,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Dict:
    """Same retrieval as run_one_example() (BM25 + symbol + dependency,
    deduplicated, capped at 12), but every nominated candidate is passed to
    the generator -- no selection step, LLMSelector is never involved. Only
    example["prompt"]/["file"] ever reach retrieval or generation --
    groundtruth is extracted for evaluation only, never passed to either.

    `generator` must be passed explicitly (e.g. CompletionGenerator with
    HuggingFaceGenerationBackend on Colab) -- there is no local default
    backend, since real generation only ever runs on Colab.
    """
    example = load_cceval_example(jsonl_path, index)
    chunks = locate_repo_index(example["repository"], repos_dir=repos_dir, index_dir=index_dir)

    pipeline = CandidatePipeline(BM25Retriever(chunks), SymbolRetriever(chunks), DependencyRetriever(chunks))
    candidates = pipeline.nominate(example["prompt"], target_file=example["file"])
    all_candidate_ids = [c["chunk_id"] for c in candidates]

    generated = generator.generate(example["prompt"], example["file"], all_candidate_ids)

    return {
        "task_id": example["task_id"],
        "repository": example["repository"],
        "target_file": example["file"],
        "prompt": example["prompt"],
        "groundtruth": example["groundtruth"],
        "num_candidates": len(candidates),
        "candidates": candidates,
        "completion": generated["completion"],
    }


def _run_single_task_baseline(
    generator: CompletionGenerator, jsonl_path: str, index: int, repos_dir: str, index_dir: str
) -> Dict:
    example = load_cceval_example(jsonl_path, index)
    record = {"task_id": example["task_id"]}

    try:
        t0 = time.time()
        result = run_one_example_baseline(
            generator, jsonl_path=jsonl_path, index=index, repos_dir=repos_dir, index_dir=index_dir
        )
        generation_time = time.time() - t0

        record.update(
            {
                "candidate_count": result["num_candidates"],
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
        record.update(_EMPTY_BASELINE_FIELDS)
        record["error"] = f"{type(e).__name__}: {e}"

    return record


def summarize_baseline(task_results: List[Dict]) -> Dict:
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
        "exact_match_rate": (sum(1 for m in exact_matches if m) / len(exact_matches)) if exact_matches else None,
        "avg_ES": avg("ES"),
        "avg_ID_F1": avg("ID-F1"),
        "avg_generation_time": avg("generation_time"),
    }


def run_baseline_experiment(
    generator: CompletionGenerator,
    n_tasks: int = 20,
    jsonl_path: str = DEFAULT_JSONL,
    results_dir: str = DEFAULT_RESULTS_DIR,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Dict:
    """Runs the no-selection baseline across n_tasks: same retrieval as the
    Qwen-selection experiment, but every nominated candidate goes straight
    to the generator -- LLMSelector is never called. Saves
    results/cceval_{n_tasks}_baseline.jsonl and
    results/cceval_{n_tasks}_baseline_summary.json; never touches the
    selection experiment's own result files.

    `generator` must be passed explicitly -- no local/Ollama default, since
    real generation only ever runs on Colab.
    """
    preflight_check(jsonl_path, n_tasks)

    task_results = [
        _run_single_task_baseline(generator, jsonl_path, index, repos_dir, index_dir) for index in range(n_tasks)
    ]
    summary = summarize_baseline(task_results)

    results_dir_path = Path(results_dir)
    save_results_jsonl(task_results, results_dir_path / f"cceval_{n_tasks}_baseline.jsonl")
    save_summary_json(summary, results_dir_path / f"cceval_{n_tasks}_baseline_summary.json")

    return {"results": task_results, "summary": summary}


def print_baseline_isolation_flags() -> None:
    print("=== Data isolation verification (no-selection baseline) ===")
    print("groundtruth_used_for_retrieval = False")
    print("groundtruth_used_for_selection = N/A (no selection stage in the baseline)")
    print("groundtruth_used_for_generation = False")
    print("groundtruth_used_for_evaluation = True")


def _fmt(value, spec: str = "{:.3f}") -> str:
    return spec.format(value) if value is not None else "n/a"


def print_baseline_summary_table(summary: Dict) -> None:
    print("=== CCEval no-selection baseline summary ===")
    print(f"  total tasks:          {summary['total_tasks']}")
    print(f"  successful:           {summary['num_successful']}")
    print(f"  failed:               {summary['num_failed']}")
    if summary["failed_task_ids"]:
        print(f"  failed task_ids:      {summary['failed_task_ids']}")
    print(f"  avg candidate count:  {_fmt(summary['avg_candidate_count'], '{:.2f}')}")
    print(f"  exact match rate:     {_fmt(summary['exact_match_rate'])}")
    print(f"  avg ES:               {_fmt(summary['avg_ES'])}")
    print(f"  avg ID-F1:            {_fmt(summary['avg_ID_F1'])}")
    print(f"  avg generation time:  {_fmt(summary['avg_generation_time'], '{:.2f}')}s")


def print_baseline_task_table(task_results: List[Dict]) -> None:
    header = f"{'task_id':<24} | {'candidates':>10} | {'EM':>5} | {'ES':>6} | {'ID-F1':>6} | {'time':>7}"
    print(header)
    print("-" * len(header))
    for r in task_results:
        if r["error"] is not None:
            print(f"{r['task_id']:<24} | ERROR: {r['error']}")
            continue
        em = "True" if r["exact_match"] else "False"
        print(
            f"{r['task_id']:<24} | {r['candidate_count']:>10} | "
            f"{em:>5} | {r['ES']:.3f} | {r['ID-F1']:.3f} | {r['generation_time']:.2f}s"
        )
