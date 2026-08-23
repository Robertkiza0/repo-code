import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from selection.backends import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
    OllamaBackend,
    SelectionBackend,
)

logger = logging.getLogger(__name__)

# Kept as aliases for backward compatibility -- these used to live here directly.
DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL
DEFAULT_HOST = DEFAULT_OLLAMA_HOST
DEFAULT_TIMEOUT = DEFAULT_OLLAMA_TIMEOUT

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
_DOCSTRING_PREVIEW_LEN = 200


def _strip_code_fence(text: str) -> str:
    """Strips a ```json ... ``` or ``` ... ``` markdown code fence if the
    whole (stripped) text is wrapped in one, returning the inner text
    unchanged otherwise. Models frequently wrap JSON responses in a fence
    even when explicitly told to respond with JSON only."""
    match = _CODE_FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text


def _try_parse_json(raw_response: str) -> Tuple[object, bool]:
    """Attempts to parse raw_response as JSON -- first the whole thing
    (after stripping a code fence if present), then falling back to the
    first "[...]" block found in it (e.g. the model added stray prose
    despite being asked for JSON only). Returns (value, True) on success,
    or (None, False) if genuinely nothing parseable was found -- this is
    the single source of truth for whether the response was valid JSON at
    all, used both by parse_selected_ids() and parse_selection_response()
    so a real parse failure is never silently indistinguishable from a
    validly-empty response.
    """
    text = _strip_code_fence(raw_response)
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass

    match = _JSON_ARRAY_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0)), True
        except json.JSONDecodeError:
            pass

    return None, False


def parse_selected_ids(raw_response: str) -> List[str]:
    """Extract a list of chunk_id strings from the model's raw text response.

    Accepts {"selected_chunk_ids": [...]} or a bare [...] list, optionally
    wrapped in a ```json ... ``` code fence. Returns [] both when the
    response validly says "nothing selected" and when it's unparseable --
    use parse_selection_response() if you need to tell those two apart.
    """
    return parse_selection_response(raw_response)["selected_chunk_ids"]


def parse_selection_response(raw_response: str) -> Dict:
    """Like parse_selected_ids(), but never silently collapses a genuine
    parse failure into an empty list. Returns:
      {"selected_chunk_ids": [...], "parse_status": "ok", "selection_parse_error": None}
    on success (even if the list is legitimately empty), or
      {"selected_chunk_ids": [], "parse_status": "parse_error", "selection_parse_error": "..."}
    if raw_response wasn't valid JSON at all (after fence-stripping and the
    bracket-extraction fallback), so callers can distinguish "the model chose
    nothing" from "the response was unparseable" instead of both collapsing
    to the same empty list.
    """
    data, ok = _try_parse_json(raw_response)
    if not ok:
        return {
            "selected_chunk_ids": [],
            "parse_status": "parse_error",
            "selection_parse_error": "raw_response is not valid JSON, even after stripping a markdown code fence",
        }

    if isinstance(data, dict):
        data = data.get("selected_chunk_ids", [])
    if not isinstance(data, list):
        return {
            "selected_chunk_ids": [],
            "parse_status": "parse_error",
            "selection_parse_error": (
                f"parsed JSON is a {type(data).__name__}, not a list or a "
                '{"selected_chunk_ids": [...]} object'
            ),
        }

    ids = [str(item) for item in data if isinstance(item, (str, int))]
    return {"selected_chunk_ids": ids, "parse_status": "ok", "selection_parse_error": None}


class LLMSelector:
    """Asks a model which candidate chunks are useful context for completing
    the code at the cursor -- it only ever picks ids, it is never asked to
    write the completion itself.

    Any chunk_id the model returns that isn't actually in the candidate pool
    (a hallucination) is dropped rather than trusted; both the full candidate
    pool and the validated selection are logged for every call.

    Model access is pluggable via `backend` (a SelectionBackend): defaults to
    OllamaBackend for backward compatibility, but e.g. `HuggingFaceBackend()`
    from selection.backends can be passed instead to run a transformers model
    in-process (useful where running a separate Ollama server is impractical,
    like Colab).
    """

    def __init__(
        self,
        chunks: List[Dict],
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT,
        backend: Optional[SelectionBackend] = None,
    ):
        self.backend = backend or OllamaBackend(model=model, host=host, timeout=timeout)
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
            return {
                "selected_chunk_ids": [],
                "candidate_chunk_ids": [],
                "rejected_hallucinated_ids": [],
                "raw_response": "",
                "parse_status": "ok",
                "selection_parse_error": None,
            }

        labels = self._assign_labels(candidates)
        label_to_chunk_id = {label: chunk_id for chunk_id, label in labels.items()}

        prompt = self._build_prompt(code_before_cursor, target_file, candidates, labels)
        raw_response = self.backend.generate(prompt)
        parsed = parse_selection_response(raw_response)
        proposed_labels = parsed["selected_chunk_ids"]

        # The model only ever sees/returns short labels (C1, C2, ...), never
        # real chunk_ids -- map back here so everything downstream of
        # select() (generation, evaluation, ...) still only ever deals in
        # real chunk_ids, unchanged from before this label scheme existed.
        selected_ids: List[str] = []
        seen = set()
        for label in proposed_labels:
            chunk_id = label_to_chunk_id.get(label)
            if chunk_id is not None and chunk_id not in seen:
                seen.add(chunk_id)
                selected_ids.append(chunk_id)
        rejected_ids = [label for label in proposed_labels if label not in label_to_chunk_id]

        logger.info(
            "llm_selector: target_file=%s candidate_ids=%s selected_ids=%s rejected_hallucinated_ids=%s "
            "parse_status=%s",
            target_file,
            candidate_ids,
            selected_ids,
            rejected_ids,
            parsed["parse_status"],
        )
        if parsed["parse_status"] != "ok":
            logger.warning(
                "llm_selector: target_file=%s selection_parse_error=%s raw_response=%r",
                target_file,
                parsed["selection_parse_error"],
                raw_response,
            )

        return {
            "selected_chunk_ids": selected_ids,
            "candidate_chunk_ids": candidate_ids,
            "rejected_hallucinated_ids": rejected_ids,
            "raw_response": raw_response,
            "parse_status": parsed["parse_status"],
            "selection_parse_error": parsed["selection_parse_error"],
        }

    def _assign_labels(self, candidates: List[Dict]) -> Dict[str, str]:
        """Maps each candidate's real chunk_id -> the short label ("C1",
        "C2", ...) shown to the model, in nomination order. This is the
        ONLY id the model ever sees or is asked to return -- select() maps
        labels back to real chunk_ids after parsing, so nothing downstream
        of select() (generation, evaluation, ...) ever sees a label.
        """
        return {candidate["chunk_id"]: f"C{i + 1}" for i, candidate in enumerate(candidates)}

    def _format_candidate(self, candidate: Dict, label: str) -> str:
        chunk = self._chunk_lookup.get(candidate["chunk_id"], {})
        docstring = (chunk.get("docstring") or "").strip()
        if len(docstring) > _DOCSTRING_PREVIEW_LEN:
            docstring = docstring[:_DOCSTRING_PREVIEW_LEN] + "..."
        lines = [
            label,
            f'file: {chunk.get("file_path", candidate.get("file_path", "?"))}',
            f'type: {chunk.get("type", "?")}',
            f'signature: {chunk.get("signature", "?")}',
        ]
        if docstring:
            lines.append(f"docstring: {docstring}")
        return "\n".join(lines)

    def _build_prompt(
        self, code_before_cursor: str, target_file: str, candidates: List[Dict], labels: Dict[str, str]
    ) -> str:
        candidate_blocks = "\n\n".join(self._format_candidate(c, labels[c["chunk_id"]]) for c in candidates)
        return (
            "You are a CONTEXT SELECTOR for repository-level code "
            "completion. You do not write or generate code -- you only "
            "decide which candidate chunks the code generator needs to "
            "see.\n\n"
            f"Target file: {target_file}\n\n"
            f"Incomplete code:\n```\n{code_before_cursor}\n```\n\n"
            f"Candidates:\n{candidate_blocks}\n\n"
            "For each candidate, reason about the completion point: which "
            "symbols, functions, classes, or objects in the incomplete "
            "code need information from another chunk to complete "
            "correctly? Which candidate explains the expected API, "
            "arguments, return value, state, or behavior needed at the "
            "cursor? Which candidate is structurally related to the "
            "current code, even without high lexical overlap? Are there "
            "candidates that only make sense together, providing "
            "complementary context?\n\n"
            "Do not select a candidate merely because it is in the same "
            "file, was retrieved by keyword search, has a name that looks "
            "vaguely similar, or is a dependency candidate without "
            "evidence it is actually useful. Do not rely on lexical "
            "similarity alone -- repository-level dependencies and "
            "structural relationships matter more than shared words.\n\n"
            "Prefer the smallest sufficient set; do not select candidates "
            "just to fill a quota. However, if a candidate is clearly "
            "relevant, select it even if its lexical overlap with the "
            "incomplete code is low. Select multiple candidates when the "
            "completion genuinely needs multiple pieces of repository "
            "context. Evaluate every candidate against the completion "
            "task before deciding.\n\n"
            'Return only the candidate labels shown above (e.g. "C1", '
            '"C4"), never a file name or anything else. Return JSON only '
            "-- no explanations, rankings, markdown, or code: "
            '{"selected_chunk_ids": ["C1", "C4"]}. If none are useful: '
            '{"selected_chunk_ids": []}.'
        )
