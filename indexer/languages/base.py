from abc import ABC, abstractmethod
from typing import List

from tree_sitter import Tree

from indexer.models import Chunk


class BaseExtractor(ABC):
    """A language-specific extractor turns a parsed tree-sitter Tree into Chunks.

    Implement this for each new language (e.g. JavaExtractor) and register it
    with a LanguageConfig in indexer/languages/__init__.py.
    """

    language_name: str

    @abstractmethod
    def extract_chunks(self, tree: Tree, source_bytes: bytes, file_path: str) -> List[Chunk]:
        """Return chunks (classes/functions/methods) found in one parsed file."""
        raise NotImplementedError
