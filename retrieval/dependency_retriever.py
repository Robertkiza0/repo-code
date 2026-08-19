import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, List, Set, Union

_IMPORT_RE = re.compile(r"^import\s+(.+)$")
_FROM_IMPORT_RE = re.compile(r"^from\s+(\.*)([\w.]*)\s*import\s+(.+)$")


def parse_import_modules(import_line: str, importing_file: str) -> List[str]:
    """Best-effort extraction of the dotted module path(s) referenced by an import line.

    Handles "import a, b as c" and "from a.b import c", plus simple relative
    imports ("from . import x", "from .a import b") resolved against the
    importing file's own package directory. This is a text heuristic, not a
    real import resolver: it doesn't know sys.path, and for "from . import
    name" it can't tell whether `name` is a submodule or a re-exported symbol,
    so it just tries `name` as a candidate module (the DependencyRetriever
    silently drops candidates that don't match a file in the index anyway).
    """
    line = import_line.strip().rstrip(";")

    match = _IMPORT_RE.match(line)
    if match:
        modules = []
        for part in match.group(1).split(","):
            token = part.strip().split(" as ")[0].strip()
            if token:
                modules.append(token)
        return modules

    match = _FROM_IMPORT_RE.match(line)
    if match:
        dots, module_part, names_part = match.groups()
        if not dots:
            return [module_part] if module_part else []

        base_parts = list(PurePosixPath(importing_file).parent.parts)
        level = len(dots)
        if level > 1:
            base_parts = base_parts[: max(len(base_parts) - (level - 1), 0)]

        if module_part:
            return [".".join(base_parts + module_part.split("."))]

        # "from . import name[, name2, ...]" -- try each name as a submodule.
        candidates = []
        for part in names_part.split(","):
            token = part.strip().split(" as ")[0].strip()
            if token:
                candidates.append(".".join(base_parts + [token]))
        return candidates

    return []


class DependencyRetriever:
    """Retrieves chunks from files that a target file locally imports.

    Uses only the "imports" text already recorded per chunk in repo_index.json
    -- no re-parsing of source. External/stdlib imports (typing, os, ...) simply
    don't resolve to any file in the index and are silently skipped.
    """

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self._chunks_by_file: Dict[str, List[Dict]] = defaultdict(list)
        self._imports_by_file: Dict[str, List[str]] = {}
        for chunk in chunks:
            file_path = chunk["file_path"]
            self._chunks_by_file[file_path].append(chunk)
            self._imports_by_file.setdefault(file_path, chunk.get("imports") or [])
        self._module_to_file = self._build_module_index(self._chunks_by_file.keys())

    @classmethod
    def from_index_file(cls, index_path: Union[str, Path]) -> "DependencyRetriever":
        chunks = json.loads(Path(index_path).read_text(encoding="utf-8"))
        return cls(chunks)

    @staticmethod
    def _build_module_index(file_paths) -> Dict[str, str]:
        """Map both a bare stem ("module_a") and a dotted path ("pkg.module_b")
        to the file that defines it, so "from module_a import X" resolves the
        same as "from pkg.module_b import Y" would for a nested file."""
        module_to_file: Dict[str, str] = {}
        for file_path in file_paths:
            path = PurePosixPath(file_path)
            parent_parts = path.parent.parts
            if path.stem == "__init__":
                dotted = ".".join(parent_parts)
                bare = parent_parts[-1] if parent_parts else dotted
            else:
                dotted = ".".join(parent_parts + (path.stem,))
                bare = path.stem
            for key in {dotted, bare}:
                if key:
                    module_to_file.setdefault(key, file_path)
        return module_to_file

    def _resolve_import_files(self, importing_file: str, import_line: str) -> Set[str]:
        resolved = set()
        for module in parse_import_modules(import_line, importing_file):
            file_path = self._module_to_file.get(module)
            if file_path and file_path != importing_file:
                resolved.add(file_path)
        return resolved

    def get_dependency_files(self, target_file: str) -> Set[str]:
        """The set of in-repo files that target_file locally imports."""
        files: Set[str] = set()
        for import_line in self._imports_by_file.get(target_file, []):
            files |= self._resolve_import_files(target_file, import_line)
        return files

    def search(self, target_file: str, top_k: int = None) -> List[Dict]:
        """All chunks defined in files that target_file locally imports."""
        candidates = [
            chunk
            for file_path in sorted(self.get_dependency_files(target_file))
            for chunk in self._chunks_by_file.get(file_path, [])
        ]
        if top_k is not None:
            candidates = candidates[:top_k]
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "file_path": chunk["file_path"],
                "name": chunk["name"],
                # Presence signal, not a ranked relevance score: every chunk from
                # an imported file gets the same top-tier score.
                "score": 10.0,
            }
            for chunk in candidates
        ]


def search_index(index_path: Union[str, Path], target_file: str, top_k: int = None) -> List[Dict]:
    """Convenience one-shot search: load a repo_index.json and find target_file's local-import chunks."""
    return DependencyRetriever.from_index_file(index_path).search(target_file, top_k)
