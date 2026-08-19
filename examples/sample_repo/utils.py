"""Generic helper utilities used across the sample project."""
import re
from typing import Iterable, List


def is_even(n: int) -> bool:
    """Return True if n is even."""
    return n % 2 == 0


def chunk_list(items: List, size: int) -> List[List]:
    """Split items into consecutive chunks of at most size elements."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def slugify(text: str) -> str:
    """Convert text into a lowercase, hyphen-separated slug."""
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_]+", "-", text)


def flatten(nested: Iterable[Iterable]) -> List:
    """Flatten one level of nesting."""
    result = []
    for group in nested:
        result.extend(group)
    return result
