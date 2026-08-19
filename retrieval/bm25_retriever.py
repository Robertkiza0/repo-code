import json
import re
from pathlib import Path
from typing import Dict, List, Union

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase identifier/word tokenizer, used identically for indexing and queries."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _chunk_text(chunk: Dict) -> str:
    return "\n".join(
        [
            chunk.get("name") or "",
            chunk.get("signature") or "",
            chunk.get("docstring") or "",
            chunk.get("file_path") or "",
            chunk.get("source_code") or "",
        ]
    )


class BM25Retriever:
    """BM25 retrieval over the chunks produced by indexer.RepoParser (repo_index.json)."""

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        corpus_tokens = [tokenize(_chunk_text(c)) for c in chunks]
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    @classmethod
    def from_index_file(cls, index_path: Union[str, Path]) -> "BM25Retriever":
        chunks = json.loads(Path(index_path).read_text(encoding="utf-8"))
        return cls(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return the top_k chunks for an (incomplete) code query, ranked by BM25 score.

        Scores are normalized to [0, 10] by dividing by the top raw BM25 score for
        this query (negative scores, a rare BM25 IDF artifact, are clipped to 0
        first) -- so 10.0 always means "best match for this query", not an
        absolute/cross-query-comparable relevance value.
        """
        if self._bm25 is None:
            return []
        raw_scores = [max(s, 0.0) for s in self._bm25.get_scores(tokenize(query))]
        max_score = max(raw_scores) if raw_scores else 0.0
        normalized_scores = [float(10.0 * s / max_score) if max_score > 0 else 0.0 for s in raw_scores]
        ranked_indices = sorted(range(len(self.chunks)), key=lambda i: normalized_scores[i], reverse=True)[:top_k]
        return [
            {
                "chunk_id": self.chunks[i]["chunk_id"],
                "file_path": self.chunks[i]["file_path"],
                "name": self.chunks[i]["name"],
                "score": normalized_scores[i],
            }
            for i in ranked_indices
        ]


def search_index(index_path: Union[str, Path], query: str, top_k: int = 5) -> List[Dict]:
    """Convenience one-shot search: load a repo_index.json and BM25-rank it against query."""
    return BM25Retriever.from_index_file(index_path).search(query, top_k)
