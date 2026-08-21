"""Runs one CrossCodeEval line_completion example through the existing
retrieval -> selection -> generation pipeline, unmodified.

Only `prompt` (the left-context) is ever passed to retrieval/selection/
generation -- `groundtruth` and `right_context` are extracted for reference
but never fed into any pipeline stage, matching a real completion setting
where the model can't see the answer or what comes after the cursor.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from indexer.repo_parser import RepoParser
from retrieval.bm25_retriever import BM25Retriever
from retrieval.symbol_retriever import SymbolRetriever
from retrieval.dependency_retriever import DependencyRetriever
from retrieval.candidate_pipeline import CandidatePipeline
from selection.llm_selector import LLMSelector
from generation.generator import CompletionGenerator
from generation.backends import OllamaGenerationBackend
from scripts.fetch_raw_repos import clone_and_checkout, resolve_owner_repo

DEFAULT_JSONL = "data/cceval/samples/line_completion_20.jsonl"
DEFAULT_REPOS_DIR = "data/cceval/repos"
DEFAULT_INDEX_DIR = "data/cceval/repo_indexes"


def load_cceval_example(jsonl_path: str = DEFAULT_JSONL, index: int = 0) -> Dict:
    """Extract task_id, repository, file, prompt, and groundtruth from one
    CrossCodeEval example. right_context is intentionally never read here."""
    path = Path(jsonl_path)
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                raw = json.loads(line)
                break
        else:
            raise IndexError(f"{path} has fewer than {index + 1} line(s)")

    return {
        "task_id": raw["metadata"]["task_id"],
        "repository": raw["metadata"]["repository"],
        "file": raw["metadata"]["file"],
        "prompt": raw["prompt"],
        "groundtruth": raw["groundtruth"],
    }


def locate_repo_index(
    repository: str,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> List[Dict]:
    """Find the chunks for a CrossCodeEval repository id, building them if needed:
    resolve owner/repo/commit and clone (reusing scripts.fetch_raw_repos' logic,
    not duplicating it) if not already fetched, then parse with the existing
    indexer and cache the result. Cached results are reused on later calls."""
    index_path = Path(index_dir) / f"{repository}.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))

    repo_dir = Path(repos_dir) / repository
    if not repo_dir.exists():
        print(f"[cceval_adapter] {repository} not fetched yet -- resolving and cloning...")
        owner, repo, commit = resolve_owner_repo(repository)
        clone_and_checkout(owner, repo, commit, repo_dir)
        print(f"[cceval_adapter] cloned {owner}/{repo}@{commit} -> {repo_dir}")

    chunks = [c.to_dict() for c in RepoParser(str(repo_dir)).parse_repo()]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"[cceval_adapter] indexed {len(chunks)} chunk(s) -> {index_path}")
    return chunks


def run_one_example(
    jsonl_path: str = DEFAULT_JSONL,
    index: int = 0,
    selector: Optional[LLMSelector] = None,
    generator: Optional[CompletionGenerator] = None,
    repos_dir: str = DEFAULT_REPOS_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Dict:
    """End-to-end: load one example, locate its repo's chunks, nominate
    candidates (BM25 + symbol + dependency), let `selector` (default: Qwen via
    Ollama) pick useful ones, then have `generator` (default: StarCoder via
    Ollama) complete the code. Only example["prompt"] and example["file"] are
    ever passed downstream -- groundtruth/right_context never are.
    """
    example = load_cceval_example(jsonl_path, index)
    chunks = locate_repo_index(example["repository"], repos_dir=repos_dir, index_dir=index_dir)

    pipeline = CandidatePipeline(BM25Retriever(chunks), SymbolRetriever(chunks), DependencyRetriever(chunks))
    candidates = pipeline.nominate(example["prompt"], target_file=example["file"])

    if selector is None:
        selector = LLMSelector(chunks)
    selection = selector.select(example["prompt"], example["file"], candidates)

    if generator is None:
        generator = CompletionGenerator(chunks, OllamaGenerationBackend())
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
        "completion": generated["completion"],
    }


def _preview(source_code: str, num_lines: int) -> str:
    lines = source_code.splitlines()
    preview = "\n".join(f"    {line}" for line in lines[:num_lines])
    if len(lines) > num_lines:
        preview += "\n    ..."
    return preview


def print_result(chunks: List[Dict], result: Dict, preview_lines: int = 3, prompt_tail_lines: int = 8) -> None:
    """Prints task_id/repository/target_file, the tail of the incomplete
    prompt, all candidates (chunk_id + a short source preview), Qwen's
    selected chunk_ids with their full source, the StarCoder completion, and
    -- for comparison only, never fed into the pipeline -- the groundtruth.
    """
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    print("task_id:    ", result["task_id"])
    print("repository: ", result["repository"])
    print("target_file:", result["target_file"])
    print()

    prompt_lines = result["prompt"].splitlines()
    tail = prompt_lines[-prompt_tail_lines:]
    print(f"Incomplete code (last {len(tail)} line(s) of the prompt):")
    print("\n".join(tail))
    print()

    print(f"Candidates ({result['num_candidates']}):")
    for candidate in result["candidates"]:
        chunk = chunk_lookup.get(candidate["chunk_id"], {})
        print(f"  [{candidate['chunk_id']}]  sources={candidate['sources']}")
        print(_preview(chunk.get("source_code", ""), preview_lines))
    print()

    print(f"Qwen selected chunk_ids ({len(result['selected_chunk_ids'])}):")
    for chunk_id in result["selected_chunk_ids"]:
        chunk = chunk_lookup.get(chunk_id, {})
        print(f"  [{chunk_id}]")
        for line in chunk.get("source_code", "").splitlines():
            print(f"    {line}")
    print()

    print("StarCoder completion:")
    print(result["completion"])
    print()

    print("Ground truth (for comparison only -- never passed to retrieval/selection/generation):")
    print(result["groundtruth"])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one CrossCodeEval example through retrieval -> selection -> generation."
    )
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("-n", "--index", type=int, default=0)
    args = parser.parse_args()

    example = load_cceval_example(args.jsonl, args.index)
    chunks = locate_repo_index(example["repository"])  # cached after first call
    result = run_one_example(args.jsonl, args.index)
    print_result(chunks, result)


if __name__ == "__main__":
    main()
