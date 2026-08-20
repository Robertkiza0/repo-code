import json
from pathlib import Path
from typing import Dict, List, Union

from generation.backends import GenerationBackend


class CompletionGenerator:
    """Builds a repo-context completion prompt from selected chunk_ids and
    generates a completion via a pluggable GenerationBackend (e.g. StarCoder).

    Only ever consumes chunk_ids that some upstream selector (e.g.
    selection.LLMSelector) already validated against the real candidate pool
    -- an id with no matching chunk in the index is silently skipped rather
    than crashing, since it isn't this class's job to re-validate selection.
    """

    def __init__(self, chunks: List[Dict], backend: GenerationBackend):
        self.backend = backend
        self._chunk_lookup = {c["chunk_id"]: c for c in chunks}

    @classmethod
    def from_index_file(cls, index_path: Union[str, Path], backend: GenerationBackend) -> "CompletionGenerator":
        chunks = json.loads(Path(index_path).read_text(encoding="utf-8"))
        return cls(chunks, backend)

    def build_prompt(self, code_before_cursor: str, target_file: str, selected_chunk_ids: List[str]) -> str:
        context_blocks = []
        for chunk_id in selected_chunk_ids:
            chunk = self._chunk_lookup.get(chunk_id)
            if chunk is None:
                continue
            context_blocks.append(f"# File: {chunk['file_path']}\n{chunk['source_code']}")

        parts = []
        if context_blocks:
            parts.append("# Repository context:\n\n" + "\n\n".join(context_blocks))
        parts.append(f"# File: {target_file}\n{code_before_cursor}")
        return "\n\n".join(parts)

    def generate(self, code_before_cursor: str, target_file: str, selected_chunk_ids: List[str]) -> Dict:
        context = self.build_prompt(code_before_cursor, target_file, selected_chunk_ids)
        completion = self.backend.generate(context)
        return {
            "target_file": target_file,
            "selected_chunk_ids": selected_chunk_ids,
            "context": context,
            "completion": completion,
        }
