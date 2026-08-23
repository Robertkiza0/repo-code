"""llm_selection_with_memory experiment: identical retrieval, candidate
nomination, candidate ordering, Qwen selection reasoning, StarCoder
generation, and evaluation as llm_selection (evaluation.experiment's
run_experiment_v2) -- the ONLY difference is that the selector is also
given structured repository memory (memory.build_repository_memory),
built once per repository from the SAME chunks retrieval already uses,
entirely offline and never touching groundtruth. Never calls
run_one_example()/run_experiment_v2, so llm_selection's own saved results
are never touched.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.cceval_adapter import (
    DEFAULT_INDEX_DIR,
    DEFAULT_JSONL,
    DEFAULT_REPOS_DIR,
    assign_candidate_labels,
    load_cceval_example,
    locate_repo_index,
)
from evaluation.experiment import preflight_check, save_results_jsonl, save_summary_json, summarize
from evaluation.metrics import edit_similarity, exact_match, identifier_f1
from generation.generator import CompletionGenerator
from memory.repository_memory import build_repository_memory
from retrieval.bm25_retriever import BM25Retriever
from retrieval.candidate_pipeline import CandidatePipeline
from retrieval.dependency_retriever import DependencyRetriever
from retrieval.symbol_retriever import SymbolRetriever
from selection.llm_selector import MAX_MEMORY_SELECTED_HINT, LLMSelector

DEFAULT_RESULTS_DIR = "results"

_EMPTY_TASK_FIELDS = {
    "candidate_count": None,
    "candidate_ids": None,
    "selected_candidate_ids": None,
    "selected_count": None,
    "memory_symbols_found_count": None,
    "memory_relationships_found_count": None,
    "memory_augmented_candidate_count": None,
    "raw_llm_response": None,
    "parse_status": None,
    "completion": None,
    "groundtruth": None,
    "exact_match": None,
    "ES": None,
    "ID-F1": None,
    "generation_time": None,
}


def run_one_example_llm_selection_with_memory(
    selector: LLMSelector,
    generator: CompletionGenerator,
    jsonl_path: str = DEFAULT_JSONL,
    index: int = 0,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    max_selected: int = MAX_MEMORY_SELECTED_HINT,
    memory_cache: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """Same retrieval as llm_selection (BM25 + symbol + dependency,
    deduplicated, capped at 12, unmodified ordering); the only difference
    is that `selector.select()` is called with `memory=` (structured
    repository memory built from the SAME chunks, offline, no groundtruth)
    and `max_selected=`. `memory_cache`, if given, is a dict this function
    reads/writes keyed by repository -- so a batch run only builds each
    repository's memory once instead of once per task, even though a repo
    is nominated by many CCEval tasks. Only example["prompt"]/["file"]
    ever reach retrieval, memory construction/query, or selection --
    groundtruth is extracted for evaluation only, never passed to any of
    them.
    """
    example = load_cceval_example(jsonl_path, index)
    chunks = locate_repo_index(example["repository"], repos_dir=repos_dir, index_dir=index_dir)

    if memory_cache is not None and example["repository"] in memory_cache:
        memory = memory_cache[example["repository"]]
    else:
        memory = build_repository_memory(chunks)
        if memory_cache is not None:
            memory_cache[example["repository"]] = memory

    pipeline = CandidatePipeline(BM25Retriever(chunks), SymbolRetriever(chunks), DependencyRetriever(chunks))
    candidates = pipeline.nominate(example["prompt"], target_file=example["file"])

    selection = selector.select(
        example["prompt"], example["file"], candidates, memory=memory, max_selected=max_selected
    )
    generated = generator.generate(example["prompt"], example["file"], selection["selected_chunk_ids"])

    return {
        "task_id": example["task_id"],
        "repository": example["repository"],
        "target_file": example["file"],
        "prompt": example["prompt"],
        "groundtruth": example["groundtruth"],
        "num_candidates": len(candidates),
        "candidates": candidates,
        "selected_chunk_ids": selection["selected_chunk_ids"],
        "raw_response": selection["raw_response"],
        "parse_status": selection["parse_status"],
        "memory_symbols_found": selection.get("memory_symbols_found", []),
        "memory_relationships_found": selection.get("memory_relationships_found", []),
        "memory_augmented_candidate_count": selection.get("memory_augmented_candidate_count", 0),
        "completion": generated["completion"],
    }


def _run_single_task(
    selector: LLMSelector,
    generator: CompletionGenerator,
    jsonl_path: str,
    index: int,
    repos_dir: str,
    index_dir: str,
    max_selected: int,
    memory_cache: Dict[str, Dict],
) -> Dict:
    example = load_cceval_example(jsonl_path, index)
    record = {"task_id": example["task_id"]}

    try:
        t0 = time.time()
        result = run_one_example_llm_selection_with_memory(
            selector,
            generator,
            jsonl_path=jsonl_path,
            index=index,
            repos_dir=repos_dir,
            index_dir=index_dir,
            max_selected=max_selected,
            memory_cache=memory_cache,
        )
        generation_time = time.time() - t0

        labels = assign_candidate_labels(result["candidates"])
        selected_ids = [cid for cid in result["selected_chunk_ids"] if cid in labels]

        record.update(
            {
                "candidate_count": result["num_candidates"],
                "candidate_ids": [labels[c["chunk_id"]] for c in result["candidates"]],
                "selected_candidate_ids": [labels[cid] for cid in selected_ids],
                "selected_count": len(selected_ids),
                "memory_symbols_found_count": len(result["memory_symbols_found"]),
                "memory_relationships_found_count": len(result["memory_relationships_found"]),
                "memory_augmented_candidate_count": result["memory_augmented_candidate_count"],
                "raw_llm_response": result["raw_response"],
                "parse_status": result["parse_status"],
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
        record.update(_EMPTY_TASK_FIELDS)
        record["error"] = f"{type(e).__name__}: {e}"

    return record


def summarize_llm_selection_with_memory(task_results: List[Dict]) -> Dict:
    base_summary = summarize(task_results)  # reused, unmodified -- same fields as every other experiment

    successful = [r for r in task_results if r["error"] is None]

    def avg(key: str) -> Optional[float]:
        values = [r[key] for r in successful if r[key] is not None]
        return sum(values) / len(values) if values else None

    memory_assisted = sum(1 for r in successful if (r["memory_augmented_candidate_count"] or 0) > 0)
    distribution = {"0": 0, "1": 0, "2": 0, ">=3": 0}
    for r in successful:
        n = r["selected_count"]
        key = str(n) if n in (0, 1, 2) else ">=3"
        distribution[key] += 1

    base_summary.update(
        {
            "memory_assisted_selections": memory_assisted,
            "avg_memory_relationships_found": avg("memory_relationships_found_count"),
            "selection_count_distribution": distribution,
        }
    )
    return base_summary


def run_llm_selection_with_memory_experiment(
    selector: LLMSelector,
    generator: CompletionGenerator,
    n_tasks: int = 20,
    jsonl_path: str = DEFAULT_JSONL,
    results_dir: str = DEFAULT_RESULTS_DIR,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    max_selected: int = MAX_MEMORY_SELECTED_HINT,
) -> Dict:
    """Runs the llm_selection_with_memory experiment across n_tasks. Saves
    results/cceval_{n_tasks}_llm_selection_with_memory.jsonl and
    results/cceval_{n_tasks}_llm_selection_with_memory_summary.json; never
    touches llm_selection's (or any other experiment's) own result files.
    """
    preflight_check(jsonl_path, n_tasks)

    memory_cache: Dict[str, Dict] = {}
    task_results = [
        _run_single_task(selector, generator, jsonl_path, index, repos_dir, index_dir, max_selected, memory_cache)
        for index in range(n_tasks)
    ]
    summary = summarize_llm_selection_with_memory(task_results)

    results_dir_path = Path(results_dir)
    save_results_jsonl(task_results, results_dir_path / f"cceval_{n_tasks}_llm_selection_with_memory.jsonl")
    save_summary_json(summary, results_dir_path / f"cceval_{n_tasks}_llm_selection_with_memory_summary.json")

    return {"results": task_results, "summary": summary}


def print_llm_selection_with_memory_isolation_flags() -> None:
    print("=== Data isolation verification (llm_selection_with_memory) ===")
    print("groundtruth_used_for_retrieval = False")
    print("groundtruth_used_for_selection = False")
    print("groundtruth_used_for_generation = False")
    print("groundtruth_used_for_evaluation = True")


def _fmt(value, spec: str = "{:.3f}") -> str:
    return spec.format(value) if value is not None else "n/a"


def print_llm_selection_with_memory_summary_table(summary: Dict) -> None:
    print("=== CCEval llm_selection_with_memory summary ===")
    print(f"  total tasks:                {summary['total_tasks']}")
    print(f"  successful:                 {summary['num_successful']}")
    print(f"  failed:                     {summary['num_failed']}")
    if summary["failed_task_ids"]:
        print(f"  failed task_ids:            {summary['failed_task_ids']}")
    print(f"  avg candidate count:        {_fmt(summary['avg_candidate_count'], '{:.2f}')}")
    print(f"  avg selected count:         {_fmt(summary['avg_selected_count'], '{:.2f}')}")
    print(f"  exact match rate:           {_fmt(summary['exact_match_rate'])}")
    print(f"  avg ES:                     {_fmt(summary['avg_ES'])}")
    print(f"  avg ID-F1:                  {_fmt(summary['avg_ID_F1'])}")
    print(f"  avg generation time:        {_fmt(summary['avg_generation_time'], '{:.2f}')}s")
    print(f"  memory-assisted selections: {summary['memory_assisted_selections']}")
    print(f"  avg memory relationships:   {_fmt(summary['avg_memory_relationships_found'], '{:.2f}')}")
    dist = summary["selection_count_distribution"]
    print(f"  selection count dist:       0={dist['0']}  1={dist['1']}  2={dist['2']}  >=3={dist['>=3']}")


def print_llm_selection_with_memory_task_table(task_results: List[Dict]) -> None:
    header = (
        f"{'task_id':<24} | {'candidates':>10} | {'selected':>8} | {'mem_rels':>8} | "
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
            f"{r['memory_relationships_found_count']:>8} | {em:>5} | {r['ES']:.3f} | "
            f"{r['ID-F1']:.3f} | {r['generation_time']:.2f}s"
        )
