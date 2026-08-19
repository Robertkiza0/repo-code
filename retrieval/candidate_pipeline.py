import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dependency_retriever import DependencyRetriever
from retrieval.symbol_retriever import SymbolRetriever

DEFAULT_MAX_CANDIDATES = 12
DEFAULT_PER_SOURCE_TOP_K = 8


@dataclass
class Candidate:
    chunk_id: str
    file_path: str
    name: str
    sources: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "name": self.name,
            "sources": sorted(self.sources),
            "scores": dict(self.scores),
        }


class CandidatePipeline:
    """Combines BM25, symbol, and dependency retrieval into one deduplicated candidate pool.

    Each retriever nominates chunks independently; a chunk nominated by more
    than one source is merged into a single Candidate that records every
    source that nominated it (and that source's score), rather than appearing
    once per source. The pool is capped at max_candidates, ranked by (most
    nominating sources first, then highest individual score as a tiebreak).
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        symbol: SymbolRetriever,
        dependency: DependencyRetriever,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ):
        self.bm25 = bm25
        self.symbol = symbol
        self.dependency = dependency
        self.max_candidates = max_candidates

    @classmethod
    def from_index_file(
        cls, index_path: Union[str, Path], max_candidates: int = DEFAULT_MAX_CANDIDATES
    ) -> "CandidatePipeline":
        chunks = json.loads(Path(index_path).read_text(encoding="utf-8"))
        return cls(
            BM25Retriever(chunks),
            SymbolRetriever(chunks),
            DependencyRetriever(chunks),
            max_candidates=max_candidates,
        )

    def nominate(
        self,
        code_before_cursor: str,
        target_file: str,
        per_source_top_k: int = DEFAULT_PER_SOURCE_TOP_K,
    ) -> List[Dict]:
        candidates: Dict[str, Candidate] = {}

        def add(results: List[Dict], source: str) -> None:
            for r in results:
                chunk_id = r["chunk_id"]
                candidate = candidates.setdefault(
                    chunk_id, Candidate(chunk_id=chunk_id, file_path=r["file_path"], name=r["name"])
                )
                candidate.sources.append(source)
                candidate.scores[source] = r["score"]

        add(self.bm25.search(code_before_cursor, top_k=per_source_top_k), "bm25")
        add(self.symbol.search(code_before_cursor, top_k=per_source_top_k), "symbol")
        add(self.dependency.search(target_file, top_k=per_source_top_k), "dependency")

        ranked = sorted(
            candidates.values(),
            key=lambda c: (-len(c.sources), -max(c.scores.values()), c.file_path, c.name),
        )
        return [c.to_dict() for c in ranked[: self.max_candidates]]


def nominate_index(
    index_path: Union[str, Path],
    code_before_cursor: str,
    target_file: str,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    per_source_top_k: int = DEFAULT_PER_SOURCE_TOP_K,
) -> List[Dict]:
    """Convenience one-shot call: load a repo_index.json and run the full candidate pipeline."""
    pipeline = CandidatePipeline.from_index_file(index_path, max_candidates=max_candidates)
    return pipeline.nominate(code_before_cursor, target_file, per_source_top_k=per_source_top_k)
