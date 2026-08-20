import logging
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from indexer.repo_parser import RepoParser
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dependency_retriever import DependencyRetriever
from retrieval.symbol_retriever import SymbolRetriever
from retrieval.candidate_pipeline import CandidatePipeline
from selection.llm_selector import LLMSelector, parse_selected_ids
from selection.backends import HuggingFaceBackend, OllamaBackend, SelectionBackend

SAMPLE_REPO = Path(__file__).parent / "sample_repo"


def _mock_ollama_response(response_text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"response": response_text}
    return mock_response


class ParseSelectedIdsTest(unittest.TestCase):
    def test_dict_with_key(self):
        raw = '{"selected_chunk_ids": ["a.py::foo::function:1-2", "b.py::Bar::class:1-5"]}'
        self.assertEqual(parse_selected_ids(raw), ["a.py::foo::function:1-2", "b.py::Bar::class:1-5"])

    def test_bare_list(self):
        raw = '["a.py::foo::function:1-2"]'
        self.assertEqual(parse_selected_ids(raw), ["a.py::foo::function:1-2"])

    def test_empty_list(self):
        self.assertEqual(parse_selected_ids('{"selected_chunk_ids": []}'), [])

    def test_list_embedded_in_extra_prose(self):
        raw = 'Sure, here you go:\n["a.py::foo::function:1-2"]\nHope that helps!'
        self.assertEqual(parse_selected_ids(raw), ["a.py::foo::function:1-2"])

    def test_garbage_returns_empty(self):
        self.assertEqual(parse_selected_ids("not json at all"), [])

    def test_non_list_value_returns_empty(self):
        self.assertEqual(parse_selected_ids('{"selected_chunk_ids": "not-a-list"}'), [])


class LLMSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
        pipeline = CandidatePipeline(
            BM25Retriever(cls.chunks), SymbolRetriever(cls.chunks), DependencyRetriever(cls.chunks)
        )
        cls.candidates = pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py")
        cls.candidate_ids = [c["chunk_id"] for c in cls.candidates]

    def test_empty_candidates_skips_the_network_call(self):
        selector = LLMSelector(self.chunks)
        with patch("selection.backends.requests.post") as mock_post:
            result = selector.select("result = Greeter(", "pkg/module_b.py", [])
        mock_post.assert_not_called()
        self.assertEqual(result["selected_chunk_ids"], [])
        self.assertEqual(result["candidate_chunk_ids"], [])

    def test_valid_selection_is_returned(self):
        chosen = [self.candidate_ids[0], self.candidate_ids[1]]
        selector = LLMSelector(self.chunks)
        with patch("selection.backends.requests.post") as mock_post:
            mock_post.return_value = _mock_ollama_response(f'{{"selected_chunk_ids": {chosen!r}}}'.replace("'", '"'))
            result = selector.select("result = Greeter(", "pkg/module_b.py", self.candidates)
        self.assertEqual(result["selected_chunk_ids"], chosen)
        self.assertEqual(result["candidate_chunk_ids"], self.candidate_ids)
        self.assertEqual(result["rejected_hallucinated_ids"], [])

    def test_hallucinated_ids_are_dropped(self):
        chosen = [self.candidate_ids[0], "totally_made_up_chunk_id"]
        selector = LLMSelector(self.chunks)
        with patch("selection.backends.requests.post") as mock_post:
            mock_post.return_value = _mock_ollama_response(f'{{"selected_chunk_ids": {chosen!r}}}'.replace("'", '"'))
            result = selector.select("result = Greeter(", "pkg/module_b.py", self.candidates)
        self.assertEqual(result["selected_chunk_ids"], [self.candidate_ids[0]])
        self.assertEqual(result["rejected_hallucinated_ids"], ["totally_made_up_chunk_id"])

    def test_duplicate_ids_from_model_are_deduped(self):
        chosen = [self.candidate_ids[0], self.candidate_ids[0]]
        selector = LLMSelector(self.chunks)
        with patch("selection.backends.requests.post") as mock_post:
            mock_post.return_value = _mock_ollama_response(f'{{"selected_chunk_ids": {chosen!r}}}'.replace("'", '"'))
            result = selector.select("result = Greeter(", "pkg/module_b.py", self.candidates)
        self.assertEqual(result["selected_chunk_ids"], [self.candidate_ids[0]])

    def test_never_asked_to_write_code_and_never_returns_a_completion(self):
        selector = LLMSelector(self.chunks)
        captured_payload = {}

        def fake_post(url, json, timeout):
            captured_payload.update(json)
            return _mock_ollama_response('{"selected_chunk_ids": []}')

        with patch("selection.backends.requests.post", side_effect=fake_post):
            result = selector.select("result = Greeter(", "pkg/module_b.py", self.candidates)

        self.assertIn("not writing the completion", captured_payload["prompt"].lower())
        self.assertNotIn("completion", result)

    def test_logs_candidate_and_selected_ids(self):
        selector = LLMSelector(self.chunks)
        chosen = [self.candidate_ids[0]]
        with patch("selection.backends.requests.post") as mock_post:
            mock_post.return_value = _mock_ollama_response(f'{{"selected_chunk_ids": {chosen!r}}}'.replace("'", '"'))
            with self.assertLogs("selection.llm_selector", level="INFO") as log_ctx:
                selector.select("result = Greeter(", "pkg/module_b.py", self.candidates)
        logged = "\n".join(log_ctx.output)
        self.assertIn(self.candidate_ids[0], logged)
        for cid in self.candidate_ids:
            self.assertIn(cid, logged)

    def test_ollama_error_propagates(self):
        selector = LLMSelector(self.chunks)
        with patch("selection.backends.requests.post", side_effect=requests.ConnectionError("no server")):
            with self.assertRaises(requests.ConnectionError):
                selector.select("result = Greeter(", "pkg/module_b.py", self.candidates)


class FakeBackend(SelectionBackend):
    """A trivial in-memory backend for testing LLMSelector <-> backend plumbing."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.prompts_received: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts_received.append(prompt)
        return self.response_text


class LLMSelectorCustomBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
        pipeline = CandidatePipeline(
            BM25Retriever(cls.chunks), SymbolRetriever(cls.chunks), DependencyRetriever(cls.chunks)
        )
        cls.candidates = pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py")
        cls.candidate_ids = [c["chunk_id"] for c in cls.candidates]

    def test_default_backend_is_ollama(self):
        selector = LLMSelector(self.chunks)
        self.assertIsInstance(selector.backend, OllamaBackend)

    def test_injected_backend_is_used_instead_of_ollama(self):
        chosen = [self.candidate_ids[0]]
        fake = FakeBackend(f'{{"selected_chunk_ids": {chosen!r}}}'.replace("'", '"'))
        selector = LLMSelector(self.chunks, backend=fake)

        with patch("selection.backends.requests.post") as mock_post:
            result = selector.select("result = Greeter(", "pkg/module_b.py", self.candidates)

        mock_post.assert_not_called()  # never touches Ollama at all
        self.assertEqual(len(fake.prompts_received), 1)
        self.assertEqual(result["selected_chunk_ids"], chosen)


class HuggingFaceBackendTest(unittest.TestCase):
    def test_raises_clear_error_without_torch_transformers_installed(self):
        # This dev environment intentionally doesn't have torch installed
        # (kept the Windows box light) -- so this is a real assertion, not a
        # simulated one: loading by name without injected objects must fail
        # fast with a clear message rather than a confusing stack trace.
        with self.assertRaises(ImportError) as ctx:
            HuggingFaceBackend()
        self.assertIn("torch", str(ctx.exception))

    def test_generate_uses_chat_template_and_decodes_output(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = MagicMock()
        tokenizer.decode.return_value = '{"selected_chunk_ids": ["a.py::foo::function:1-2"]}'
        model = MagicMock()
        model.generate.return_value = MagicMock()

        backend = HuggingFaceBackend(tokenizer=tokenizer, model=model)
        result = backend.generate("some prompt")

        self.assertEqual(result, '{"selected_chunk_ids": ["a.py::foo::function:1-2"]}')
        tokenizer.apply_chat_template.assert_called_once()
        messages = tokenizer.apply_chat_template.call_args[0][0]
        self.assertEqual(messages, [{"role": "user", "content": "some prompt"}])
        model.generate.assert_called_once()

    def test_used_end_to_end_by_llm_selector(self):
        chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
        pipeline = CandidatePipeline(BM25Retriever(chunks), SymbolRetriever(chunks), DependencyRetriever(chunks))
        candidates = pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py")
        chosen_id = candidates[0]["chunk_id"]

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = MagicMock()
        tokenizer.decode.return_value = f'{{"selected_chunk_ids": ["{chosen_id}"]}}'
        model = MagicMock()
        model.generate.return_value = MagicMock()

        backend = HuggingFaceBackend(tokenizer=tokenizer, model=model)
        selector = LLMSelector(chunks, backend=backend)
        result = selector.select("result = Greeter(", "pkg/module_b.py", candidates)

        self.assertEqual(result["selected_chunk_ids"], [chosen_id])


def _is_ollama_reachable() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
        return True
    except requests.RequestException:
        return False


@unittest.skipUnless(_is_ollama_reachable(), "local Ollama server not reachable on localhost:11434")
class LLMSelectorLiveTest(unittest.TestCase):
    """A real end-to-end call against the user's local Ollama + qwen2.5-coder:7b."""

    @classmethod
    def setUpClass(cls):
        cls.chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
        pipeline = CandidatePipeline(
            BM25Retriever(cls.chunks), SymbolRetriever(cls.chunks), DependencyRetriever(cls.chunks)
        )
        cls.candidates = pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py")

    def test_live_selection_returns_only_valid_candidate_ids(self):
        # This machine runs the model on CPU only, so a real call can take
        # minutes -- much longer than the library's fast-failing default.
        selector = LLMSelector(self.chunks, timeout=300.0)
        result = selector.select("result = Greeter(", "pkg/module_b.py", self.candidates)
        candidate_ids = {c["chunk_id"] for c in self.candidates}
        for chunk_id in result["selected_chunk_ids"]:
            self.assertIn(chunk_id, candidate_ids)


if __name__ == "__main__":
    unittest.main()
