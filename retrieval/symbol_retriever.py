import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Matches a trailing (possibly dotted) identifier right before the cursor,
# tolerating an in-progress call's opening paren and surrounding whitespace.
# e.g. "result = calculator.add(" -> "calculator.add", "multiply(" -> "multiply".
_TRAILING_SYMBOL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(?\s*$")

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def extract_symbol(code_before_cursor: str) -> Optional[str]:
    """Extract the identifier or dotted identifier being typed at the cursor.

    Only looks at the tail of code_before_cursor -- this is a lightweight
    regex heuristic, not a Python tokenizer, so it only handles the common
    "name" / "obj.name" / "obj.name(" shapes.
    """
    match = _TRAILING_SYMBOL_RE.search(code_before_cursor)
    return match.group(1) if match else None


def split_symbol(symbol: str) -> Tuple[Optional[str], str]:
    """Split a possibly-dotted symbol into (owner_prefix, name).

    "add" -> (None, "add"); "calculator.add" -> ("calculator", "add");
    for multi-dot symbols only the last component before the final dot is
    used as the owner hint (e.g. "a.b.c" -> ("a.b", "c")).
    """
    if "." in symbol:
        prefix, name = symbol.rsplit(".", 1)
        return prefix, name
    return None, symbol


class SymbolRetriever:
    """Exact/prefix symbol matching over repo_index.json chunks.

    Unlike BM25 (bag-of-words relevance over free text), this ranks chunks by
    how directly they match the identifier being typed at the cursor, against
    each chunk's name/class_name/signature -- exact symbol matches first.
    """

    # Same 0-10 scale as BM25Retriever's normalized score, so results from both
    # retrievers can be compared/merged directly.
    EXACT_DOTTED = 10.0  # class_name.method matches the typed "prefix.name" exactly
    EXACT_NAME = 7.5  # name matches exactly, no owner/class context available or needed
    PREFIX_NAME = 5.0  # name starts with what's been typed so far (still-typing case)
    SIGNATURE_MATCH = 2.5  # the symbol shows up as a word in the signature (e.g. a param name)

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks

    @classmethod
    def from_index_file(cls, index_path: Union[str, Path]) -> "SymbolRetriever":
        chunks = json.loads(Path(index_path).read_text(encoding="utf-8"))
        return cls(chunks)

    def search(self, code_before_cursor: str, top_k: int = 5) -> List[Dict]:
        """Rank chunks for the identifier being typed right before the cursor."""
        symbol = extract_symbol(code_before_cursor)
        if not symbol:
            return []

        prefix, name = split_symbol(symbol)
        name_lower = name.lower()
        prefix_lower = prefix.lower() if prefix else None

        scored: List[Tuple[float, Dict]] = []
        for chunk in self.chunks:
            score = self._score(chunk, name_lower, prefix_lower)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: (-pair[0], pair[1].get("file_path", ""), pair[1].get("name", "")))
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "file_path": chunk["file_path"],
                "name": chunk["name"],
                "class_name": chunk.get("class_name"),
                "score": score,
            }
            for score, chunk in scored[:top_k]
        ]

    @classmethod
    def _score(cls, chunk: Dict, name_lower: str, prefix_lower: Optional[str]) -> float:
        chunk_name = (chunk.get("name") or "").lower()
        chunk_class = (chunk.get("class_name") or "").lower()
        signature = (chunk.get("signature") or "").lower()

        if prefix_lower and chunk_name == name_lower and chunk_class == prefix_lower:
            return cls.EXACT_DOTTED
        if chunk_name == name_lower:
            return cls.EXACT_NAME
        if chunk_name.startswith(name_lower):
            return cls.PREFIX_NAME
        if name_lower and name_lower in _WORD_RE.findall(signature):
            return cls.SIGNATURE_MATCH
        return 0.0


def search_index(index_path: Union[str, Path], code_before_cursor: str, top_k: int = 5) -> List[Dict]:
    """Convenience one-shot search: load a repo_index.json and symbol-rank it against the cursor context."""
    return SymbolRetriever.from_index_file(index_path).search(code_before_cursor, top_k)
