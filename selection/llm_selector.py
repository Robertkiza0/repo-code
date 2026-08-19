import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Union

import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 60.0

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_DOCSTRING_PREVIEW_LEN = 200


def parse_selected_ids(raw_response: str) -> List[str]:
    """Extract a list of chunk_id strings from the model's raw text response.

    Accepts either {"selected_chunk_ids": [...]} or a bare [...] list, and
    falls back to pulling the first "[...]" block out of a noisier response
    (e.g. the model added stray prose despite being asked for JSON only).
    Returns [] if nothing parseable is found.
    """
    text = raw_response.strip()
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_ARRAY_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    if isinstance(data, dict):
        data = data.get("selected_chunk_ids", [])
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if isinstance(item, (str, int))]


class LLMSelector:
    """Asks a local Ollama model which candidate chunks are useful context for
    completing the code at the cursor -- it only ever picks ids, it is never
    asked to write the completion itself.

    Any chunk_id the model returns that isn't actually in the candidate pool
    (a hallucination) is dropped rather than trusted; both the full candidate
    pool and the validated selection are logged for every call.
    """

    def __init__(
        self,
        chunks: List[Dict],
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._chunk_lookup = {c["chunk_id"]: c for c in chunks}

    @classmethod
    def from_index_file(cls, index_path: Union[str, Path], **kwargs) -> "LLMSelector":
        chunks = json.loads(Path(index_path).read_text(encoding="utf-8"))
        return cls(chunks, **kwargs)

    def select(self, code_before_cursor: str, target_file: str, candidates: List[Dict]) -> Dict:
        candidate_ids = [c["chunk_id"] for c in candidates]

        if not candidates:
            logger.info(
                "llm_selector: target_file=%s candidate_ids=[] selected_ids=[] (no candidates offered)",
                target_file,
            )
            return {"selected_chunk_ids": [], "candidate_chunk_ids": [], "rejected_hallucinated_ids": [], "raw_response": ""}

        prompt = self._build_prompt(code_before_cursor, target_file, candidates)
        raw_response = self._call_ollama(prompt)
        proposed_ids = parse_selected_ids(raw_response)

        candidate_id_set = set(candidate_ids)
        selected_ids: List[str] = []
        seen = set()
        for chunk_id in proposed_ids:
            if chunk_id in candidate_id_set and chunk_id not in seen:
                seen.add(chunk_id)
                selected_ids.append(chunk_id)
        rejected_ids = [chunk_id for chunk_id in proposed_ids if chunk_id not in candidate_id_set]

        logger.info(
            "llm_selector: target_file=%s candidate_ids=%s selected_ids=%s rejected_hallucinated_ids=%s",
            target_file,
            candidate_ids,
            selected_ids,
            rejected_ids,
        )

        return {
            "selected_chunk_ids": selected_ids,
            "candidate_chunk_ids": candidate_ids,
            "rejected_hallucinated_ids": rejected_ids,
            "raw_response": raw_response,
        }

    def _format_candidate(self, candidate: Dict) -> str:
        chunk = self._chunk_lookup.get(candidate["chunk_id"], {})
        docstring = (chunk.get("docstring") or "").strip()
        if len(docstring) > _DOCSTRING_PREVIEW_LEN:
            docstring = docstring[:_DOCSTRING_PREVIEW_LEN] + "..."
        lines = [
            f'chunk_id: {candidate["chunk_id"]}',
            f'  file: {chunk.get("file_path", candidate.get("file_path", "?"))}',
            f'  type: {chunk.get("type", "?")}',
            f'  signature: {chunk.get("signature", "?")}',
        ]
        if docstring:
            lines.append(f"  docstring: {docstring}")
        return "\n".join(lines)

    def _build_prompt(self, code_before_cursor: str, target_file: str, candidates: List[Dict]) -> str:
        candidate_blocks = "\n\n".join(self._format_candidate(c) for c in candidates)
        return (
            "You are helping select useful context for completing Python code. "
            "You are NOT writing the completion -- only choosing which candidate code "
            "chunks would help.\n\n"
            f"Target file: {target_file}\n\n"
            "Incomplete code (the cursor is at the end):\n"
            f"```\n{code_before_cursor}\n```\n\n"
            f"Candidate chunks:\n{candidate_blocks}\n\n"
            'Return a JSON object with exactly one key, "selected_chunk_ids", whose value '
            "is a list of the chunk_id strings above that would be useful context for "
            "completing the code. Only use chunk_id values exactly as listed above -- do "
            "not invent new ones. If none are useful, return an empty list. Do not write "
            "any code. Do not explain your answer. Respond with JSON only."
        )

    def _call_ollama(self, prompt: str) -> str:
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")
