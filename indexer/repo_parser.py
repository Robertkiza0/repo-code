import json
from pathlib import Path
from typing import Dict, Iterator, List

from tree_sitter import Parser

from indexer.languages import LANGUAGES, get_language_for_extension
from indexer.models import Chunk

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    ".idea",
    ".vscode",
}


class RepoParser:
    """Recursively parses a repository into Chunks (classes/functions/methods).

    Language support is driven entirely by indexer.languages.LANGUAGES, so
    adding a new language (e.g. Java) requires no changes here.
    """

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).resolve()
        self._parsers: Dict[str, Parser] = {}

    def _get_parser(self, language_name: str) -> Parser:
        if language_name not in self._parsers:
            config = LANGUAGES[language_name]
            self._parsers[language_name] = Parser(config.language_factory())
        return self._parsers[language_name]

    def _is_excluded(self, path: Path) -> bool:
        return any(part in EXCLUDED_DIRS for part in path.relative_to(self.repo_root).parts)

    def find_source_files(self) -> Iterator[Path]:
        extensions = {ext for config in LANGUAGES.values() for ext in config.extensions}
        for path in self.repo_root.rglob("*"):
            if path.is_file() and path.suffix in extensions and not self._is_excluded(path):
                yield path

    def parse_file(self, file_path: Path) -> List[Chunk]:
        config = get_language_for_extension(file_path.suffix)
        if config is None:
            return []

        source_bytes = file_path.read_bytes()
        parser = self._get_parser(config.name)
        tree = parser.parse(source_bytes)
        relative_path = file_path.relative_to(self.repo_root).as_posix()
        return config.extractor.extract_chunks(tree, source_bytes, relative_path)

    def parse_repo(self) -> List[Chunk]:
        chunks: List[Chunk] = []
        for file_path in sorted(self.find_source_files()):
            chunks.extend(self.parse_file(file_path))
        return chunks

    @staticmethod
    def save_index(chunks: List[Chunk], output_path: str) -> None:
        data = [chunk.to_dict() for chunk in chunks]
        Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
