import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from indexer.repo_parser import RepoParser
from generation.backends import HuggingFaceGenerationBackend, OllamaGenerationBackend
from generation.generator import CompletionGenerator
from generation.pipeline import run_examples, save_results
from selection.backends import SelectionBackend
from selection.llm_selector import LLMSelector

SAMPLE_REPO = Path(__file__).parent / "sample_repo"  # module_a.py + pkg/module_b.py (real cross-file import)


def _mock_ollama_response(response_text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"response": response_text}
    return mock_response


class FakeSelectionBackend(SelectionBackend):
    """Always selects the first `n` candidates it's shown, in prompt order."""

    def __init__(self, n: int = 1):
        self.n = n
        self.prompts_received = []

    def generate(self, prompt: str) -> str:
        self.prompts_received.append(prompt)
        # Pull "C<n>" label lines back out of the prompt we were just given,
        # so this stays correct regardless of which candidates were offered.
        labels = re.findall(r"^C\d+$", prompt, re.MULTILINE)
        return json.dumps({"selected_chunk_ids": labels[: self.n]})


class CompletionGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
        cls.chunk_by_id = {c["chunk_id"]: c for c in cls.chunks}

    def _some_chunk_id(self, name: str) -> str:
        return next(c["chunk_id"] for c in self.chunks if c["name"] == name)

    def test_build_prompt_includes_selected_chunk_source_and_target_code(self):
        greeter_id = self._some_chunk_id("Greeter")
        generator = CompletionGenerator(self.chunks, backend=MagicMock())
        prompt = generator.build_prompt("result = Greeter(", "pkg/module_b.py", [greeter_id])

        self.assertIn("class Greeter:", prompt)  # the selected chunk's source_code
        self.assertIn("# File: module_a.py", prompt)
        self.assertIn("# File: pkg/module_b.py", prompt)
        self.assertIn("result = Greeter(", prompt)  # the incomplete target code

    def test_build_prompt_with_no_selected_chunks_still_includes_target(self):
        generator = CompletionGenerator(self.chunks, backend=MagicMock())
        prompt = generator.build_prompt("result = Greeter(", "pkg/module_b.py", [])
        self.assertNotIn("Repository context", prompt)
        self.assertIn("result = Greeter(", prompt)

    def test_unknown_chunk_id_is_skipped_not_crashed_on(self):
        generator = CompletionGenerator(self.chunks, backend=MagicMock())
        prompt = generator.build_prompt("result = Greeter(", "pkg/module_b.py", ["not-a-real-id"])
        self.assertNotIn("not-a-real-id", prompt)

    def test_generate_returns_expected_result_shape(self):
        greeter_id = self._some_chunk_id("Greeter")
        fake_backend = MagicMock()
        fake_backend.generate.return_value = "self.default_greeting = default_greeting"

        generator = CompletionGenerator(self.chunks, backend=fake_backend)
        result = generator.generate("result = Greeter(", "pkg/module_b.py", [greeter_id])

        self.assertEqual(
            set(result.keys()), {"target_file", "selected_chunk_ids", "context", "completion"}
        )
        self.assertEqual(result["target_file"], "pkg/module_b.py")
        self.assertEqual(result["selected_chunk_ids"], [greeter_id])
        self.assertEqual(result["completion"], "self.default_greeting = default_greeting")
        fake_backend.generate.assert_called_once_with(result["context"])


class RunExamplesPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]

    def test_end_to_end_with_fake_selection_and_generation_backends(self):
        selection_backend = FakeSelectionBackend(n=2)
        selector = LLMSelector(self.chunks, backend=selection_backend)

        generation_backend = MagicMock()
        generation_backend.generate.return_value = "n.strip().title()"
        generator = CompletionGenerator(self.chunks, backend=generation_backend)

        examples = [{"target_file": "pkg/module_b.py", "code_before_cursor": "result = Greeter("}]
        results = run_examples(self.chunks, examples, selector, generator)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["target_file"], "pkg/module_b.py")
        self.assertEqual(len(result["selected_chunk_ids"]), 2)
        self.assertGreaterEqual(len(result["candidate_chunk_ids"]), len(result["selected_chunk_ids"]))
        self.assertEqual(result["completion"], "n.strip().title()")
        # everything selected must actually have been offered as a candidate
        for chunk_id in result["selected_chunk_ids"]:
            self.assertIn(chunk_id, result["candidate_chunk_ids"])

    def test_multiple_examples_produce_one_result_each(self):
        selector = LLMSelector(self.chunks, backend=FakeSelectionBackend(n=1))
        generator = CompletionGenerator(self.chunks, backend=MagicMock())
        examples = [
            {"target_file": "pkg/module_b.py", "code_before_cursor": "result = Greeter("},
            {"target_file": "module_a.py", "code_before_cursor": "result = greet("},
        ]
        results = run_examples(self.chunks, examples, selector, generator)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["target_file"], "pkg/module_b.py")
        self.assertEqual(results[1]["target_file"], "module_a.py")

    def test_save_results_round_trips_to_json(self):
        selector = LLMSelector(self.chunks, backend=FakeSelectionBackend(n=1))
        generation_backend = MagicMock()
        generation_backend.generate.return_value = "n.strip().title()"
        generator = CompletionGenerator(self.chunks, backend=generation_backend)
        results = run_examples(
            self.chunks, [{"target_file": "pkg/module_b.py", "code_before_cursor": "result = Greeter("}], selector, generator
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "generation_results.json"
            save_results(results, output_path)
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded, results)


class OllamaGenerationBackendTest(unittest.TestCase):
    def test_no_json_format_forced_unlike_selection(self):
        backend = OllamaGenerationBackend()
        with patch("generation.backends.requests.post") as mock_post:
            mock_post.return_value = _mock_ollama_response("def foo():\n    return 1")
            result = backend.generate("some prompt")

        self.assertEqual(result, "def foo():\n    return 1")
        payload = mock_post.call_args.kwargs["json"]
        self.assertNotIn("format", payload)  # selection forces format="json"; generation must not

    def test_default_stop_sequence_is_sent_to_ollama_server_side(self):
        backend = OllamaGenerationBackend()
        with patch("generation.backends.requests.post") as mock_post:
            mock_post.return_value = _mock_ollama_response("some completion")
            backend.generate("some prompt")

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["stop"], ["\n"])

    def test_stop_sequences_can_be_disabled(self):
        backend = OllamaGenerationBackend(stop_sequences=[])
        with patch("generation.backends.requests.post") as mock_post:
            mock_post.return_value = _mock_ollama_response("some completion")
            backend.generate("some prompt")

        payload = mock_post.call_args.kwargs["json"]
        self.assertNotIn("stop", payload["options"])

    def test_error_propagates(self):
        backend = OllamaGenerationBackend()
        with patch("generation.backends.requests.post", side_effect=requests.ConnectionError("no server")):
            with self.assertRaises(requests.ConnectionError):
                backend.generate("some prompt")


class HuggingFaceGenerationBackendTest(unittest.TestCase):
    def test_raises_clear_error_without_torch_transformers_installed(self):
        # Real assertion: this dev machine intentionally has no torch installed.
        with self.assertRaises(ImportError) as ctx:
            HuggingFaceGenerationBackend()
        self.assertIn("torch", str(ctx.exception))

    def test_generate_uses_plain_tokenizer_call_not_chat_template(self):
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "    return 1"  # single line: no default-stop-sequence truncation
        model = MagicMock()
        model.generate.return_value = MagicMock()

        backend = HuggingFaceGenerationBackend(tokenizer=tokenizer, model=model)
        result = backend.generate("# File: a.py\ndef foo():")

        self.assertEqual(result, "    return 1")
        tokenizer.assert_called_once_with("# File: a.py\ndef foo():", return_tensors="pt")
        tokenizer.apply_chat_template.assert_not_called()
        model.generate.assert_called_once()
        self.assertIn("input_ids", model.generate.call_args.kwargs)
        self.assertIn("attention_mask", model.generate.call_args.kwargs)

    def test_default_stop_sequence_truncates_at_first_newline(self):
        # A base completion model with no stop sequence will happily keep
        # generating past the target line, e.g. hallucinating fake extra
        # "# File: ..." blocks that mimic the repo-context prompt shape.
        tokenizer = MagicMock()
        tokenizer.decode.return_value = 'default_greeting="Hello")\n\n# File: pkg/module_b.py\nresult = LoudGreeter('
        model = MagicMock()
        model.generate.return_value = MagicMock()

        backend = HuggingFaceGenerationBackend(tokenizer=tokenizer, model=model)
        result = backend.generate("result = Greeter(")

        self.assertEqual(result, 'default_greeting="Hello")')
        self.assertNotIn("# File:", result)

    def test_stop_sequences_can_be_disabled(self):
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "line one\nline two\nline three"
        model = MagicMock()
        model.generate.return_value = MagicMock()

        backend = HuggingFaceGenerationBackend(tokenizer=tokenizer, model=model, stop_sequences=[])
        result = backend.generate("some prompt")

        self.assertEqual(result, "line one\nline two\nline three")


if __name__ == "__main__":
    unittest.main()
