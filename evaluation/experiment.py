"""Runs the CCEval experiment across multiple tasks, reusing the unmodified
single-task pipeline (evaluation.cceval_adapter.run_one_example -- retrieval,
selection, and generation are untouched) for each one. Adds per-task
evaluation metrics, continues past individual task failures instead of
aborting the whole run, and saves aggregate results.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from evaluation.cceval_adapter import (
    DEFAULT_INDEX_DIR,
    DEFAULT_JSONL,
    DEFAULT_REPOS_DIR,
    assign_candidate_labels,
    load_cceval_example,
    run_one_example,
)
from evaluation.metrics import edit_similarity, exact_match, identifier_f1
from generation.generator import CompletionGenerator
from selection.llm_selector import LLMSelector

DEFAULT_RESULTS_DIR = "results"

_EMPTY_TASK_FIELDS = {
    "candidate_count": None,
    "candidate_ids": None,
    "candidate_sources": None,
    "selected_candidate_ids": None,
    "selected_count": None,
    "selected_sources": None,
    "completion": None,
    "groundtruth": None,
    "exact_match": None,
    "ES": None,
    "ID-F1": None,
    "generation_time": None,
}


def _count_jsonl_lines(jsonl_path: str) -> int:
    with open(jsonl_path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def preflight_check(jsonl_path: str, n_tasks: int) -> List[Dict]:
    """Verifies n_tasks examples exist in jsonl_path and returns their
    task_id/repository/file mapping, without running anything through the
    pipeline yet."""
    available = _count_jsonl_lines(jsonl_path)
    if n_tasks > available:
        raise ValueError(f"requested {n_tasks} tasks but {jsonl_path} only has {available} line(s)")

    mapping = []
    for index in range(n_tasks):
        example = load_cceval_example(jsonl_path, index)
        mapping.append(
            {"index": index, "task_id": example["task_id"], "repository": example["repository"], "file": example["file"]}
        )
    return mapping


def _run_single_task(jsonl_path: str, index: int, selector, generator, repos_dir: str, index_dir: str) -> Dict:
    example = load_cceval_example(jsonl_path, index)
    record = {"task_id": example["task_id"], "repository": example["repository"], "target_file": example["file"]}

    try:
        t0 = time.time()
        result = run_one_example(
            jsonl_path, index, selector=selector, generator=generator, repos_dir=repos_dir, index_dir=index_dir
        )
        generation_time = time.time() - t0

        labels = assign_candidate_labels(result["candidates"])
        candidates_by_id = {c["chunk_id"]: c for c in result["candidates"]}
        selected_ids = [cid for cid in result["selected_chunk_ids"] if cid in labels]

        record.update(
            {
                "candidate_count": result["num_candidates"],
                "candidate_ids": [labels[c["chunk_id"]] for c in result["candidates"]],
                "candidate_sources": [c["sources"] for c in result["candidates"]],
                "selected_candidate_ids": [labels[cid] for cid in selected_ids],
                "selected_count": len(selected_ids),
                "selected_sources": [candidates_by_id[cid]["sources"] for cid in selected_ids],
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


def summarize(task_results: List[Dict]) -> Dict:
    successful = [r for r in task_results if r["error"] is None]
    failed = [r for r in task_results if r["error"] is not None]

    def avg(key: str) -> Optional[float]:
        values = [r[key] for r in successful if r[key] is not None]
        return sum(values) / len(values) if values else None

    ratios = [r["selected_count"] / r["candidate_count"] for r in successful if r["candidate_count"]]
    exact_matches = [r["exact_match"] for r in successful]

    return {
        "total_tasks": len(task_results),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "failed_task_ids": [r["task_id"] for r in failed],
        "avg_candidate_count": avg("candidate_count"),
        "avg_selected_count": avg("selected_count"),
        "avg_selection_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        "exact_match_rate": (sum(1 for m in exact_matches if m) / len(exact_matches)) if exact_matches else None,
        "avg_ES": avg("ES"),
        "avg_ID_F1": avg("ID-F1"),
        "avg_generation_time": avg("generation_time"),
    }


def run_experiment(
    n_tasks: int = 20,
    jsonl_path: str = DEFAULT_JSONL,
    selector: Optional[LLMSelector] = None,
    generator: Optional[CompletionGenerator] = None,
    results_dir: str = DEFAULT_RESULTS_DIR,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Dict:
    """Runs the first n_tasks examples from jsonl_path through the unmodified
    pipeline, recording per-task fields and metrics. A failing task is
    recorded with its error (candidate/completion/metric fields left None)
    and does not stop the run. Saves results/cceval_{n_tasks}_results.jsonl
    and results/cceval_{n_tasks}_summary.json. Returns {"results": [...], "summary": {...}}.
    """
    preflight_check(jsonl_path, n_tasks)  # raises early if the data isn't there, before any pipeline work

    task_results = [
        _run_single_task(jsonl_path, index, selector, generator, repos_dir, index_dir) for index in range(n_tasks)
    ]
    summary = summarize(task_results)

    results_dir_path = Path(results_dir)
    save_results_jsonl(task_results, results_dir_path / f"cceval_{n_tasks}_results.jsonl")
    save_summary_json(summary, results_dir_path / f"cceval_{n_tasks}_summary.json")

    return {"results": task_results, "summary": summary}


def save_results_jsonl(task_results: List[Dict], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in task_results:
            f.write(json.dumps(record) + "\n")


def save_summary_json(summary: Dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _fmt(value, spec: str = "{:.3f}") -> str:
    return spec.format(value) if value is not None else "n/a"


def print_summary_table(summary: Dict) -> None:
    print("=== CCEval experiment summary ===")
    print(f"  total tasks:          {summary['total_tasks']}")
    print(f"  successful:           {summary['num_successful']}")
    print(f"  failed:               {summary['num_failed']}")
    if summary["failed_task_ids"]:
        print(f"  failed task_ids:      {summary['failed_task_ids']}")
    print(f"  avg candidate count:  {_fmt(summary['avg_candidate_count'], '{:.2f}')}")
    print(f"  avg selected count:   {_fmt(summary['avg_selected_count'], '{:.2f}')}")
    print(f"  avg selection ratio:  {_fmt(summary['avg_selection_ratio'])}")
    print(f"  exact match rate:     {_fmt(summary['exact_match_rate'])}")
    print(f"  avg ES:               {_fmt(summary['avg_ES'])}")
    print(f"  avg ID-F1:            {_fmt(summary['avg_ID_F1'])}")
    print(f"  avg generation time:  {_fmt(summary['avg_generation_time'], '{:.2f}')}s")


def print_task_table(task_results: List[Dict]) -> None:
    header = f"{'task_id':<24} | {'candidates':>10} | {'selected':>8} | {'EM':>5} | {'ES':>6} | {'ID-F1':>6} | {'time':>7}"
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
