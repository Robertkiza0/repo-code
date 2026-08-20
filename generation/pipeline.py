import json
from pathlib import Path
from typing import Dict, List, Union

from retrieval.bm25_retriever import BM25Retriever
from retrieval.candidate_pipeline import DEFAULT_MAX_CANDIDATES, CandidatePipeline
from retrieval.dependency_retriever import DependencyRetriever
from retrieval.symbol_retriever import SymbolRetriever
from selection.llm_selector import LLMSelector
from generation.generator import CompletionGenerator


def run_examples(
    chunks: List[Dict],
    examples: List[Dict],
    selector: LLMSelector,
    generator: CompletionGenerator,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> List[Dict]:
    """Run the full pipeline for each completion example.

    For every example ({"target_file": ..., "code_before_cursor": ...}):
    nominate candidates (BM25 + symbol + dependency), let `selector` (e.g. a
    Qwen-backed LLMSelector) pick the useful chunk_ids from that pool, then
    have `generator` (e.g. a StarCoder-backed CompletionGenerator) produce a
    completion using those chunks as repository context.

    Returns one result dict per example: target_file, selected_chunk_ids,
    candidate_chunk_ids, context (the full assembled prompt), and completion.
    """
    pipeline = CandidatePipeline(
        BM25Retriever(chunks), SymbolRetriever(chunks), DependencyRetriever(chunks), max_candidates=max_candidates
    )

    results = []
    for example in examples:
        target_file = example["target_file"]
        code_before_cursor = example["code_before_cursor"]

        candidates = pipeline.nominate(code_before_cursor, target_file=target_file)
        selection = selector.select(code_before_cursor, target_file, candidates)
        result = generator.generate(code_before_cursor, target_file, selection["selected_chunk_ids"])
        result["candidate_chunk_ids"] = selection["candidate_chunk_ids"]
        results.append(result)

    return results


def save_results(results: List[Dict], output_path: Union[str, Path]) -> None:
    Path(output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
