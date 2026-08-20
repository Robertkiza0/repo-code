from abc import ABC, abstractmethod
from typing import List, Optional

import requests

DEFAULT_OLLAMA_MODEL = "starcoder2:3b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 120.0

DEFAULT_HF_MODEL = "bigcode/starcoder2-3b"

# CrossCodeEval-style tasks are single-*line* completions, and StarCoder is a
# raw base completion model with no notion of "stop here" -- left unchecked,
# it happily keeps going past the target line and starts hallucinating
# entirely new "# File: ..." blocks, echoing the repo-context prompt shape it
# was primed with. Stopping at the first newline matches the task and avoids
# that runaway drift; pass stop_sequences=[] to disable for block-level use.
DEFAULT_STOP_SEQUENCES = ["\n"]


def _truncate_at_stop(text: str, stop_sequences: List[str]) -> str:
    cut = len(text)
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


class GenerationBackend(ABC):
    """Turns a code-completion prompt into generated code text.

    Unlike selection.backends.SelectionBackend (asked for a constrained JSON
    list of ids), this is plain free-form text continuation -- no JSON
    formatting, no chat template: StarCoder-family models are raw
    code-continuation models, not instruction-tuned chat models, so the
    prompt is fed straight in and the continuation is returned as-is.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaGenerationBackend(GenerationBackend):
    """Calls a local (or remote) Ollama server's /api/generate endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
        max_new_tokens: int = 128,
        stop_sequences: Optional[List[str]] = None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = DEFAULT_STOP_SEQUENCES if stop_sequences is None else stop_sequences

    def generate(self, prompt: str) -> str:
        options = {"temperature": 0, "num_predict": self.max_new_tokens}
        if self.stop_sequences:
            options["stop"] = self.stop_sequences  # Ollama stops generation server-side on these

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")


class HuggingFaceGenerationBackend(GenerationBackend):
    """Runs a transformers causal LM in-process (e.g. on a Colab GPU) instead
    of calling out to a separate Ollama server.

    torch/transformers are only imported when actually loading a model from
    its name (lazily, inside __init__) -- so importing this module never
    requires them. For testing, an already-constructed tokenizer/model pair
    can be injected directly.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_HF_MODEL,
        device_map: str = "auto",
        max_new_tokens: int = 128,
        load_in_4bit: bool = True,
        stop_sequences: Optional[List[str]] = None,
        hf_token: Optional[str] = None,
        tokenizer: Optional[object] = None,
        model: Optional[object] = None,
    ):
        self.stop_sequences = DEFAULT_STOP_SEQUENCES if stop_sequences is None else stop_sequences

        if tokenizer is None or model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "HuggingFaceGenerationBackend needs 'torch' and 'transformers' installed "
                    "(pip install torch transformers accelerate)."
                ) from e
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name, token=hf_token)

            model_kwargs = {"device_map": device_map, "token": hf_token}
            if load_in_4bit:
                # Same reasoning as selection.backends.HuggingFaceBackend: fits
                # fully on a free-tier GPU instead of getting silently
                # offloaded to CPU/disk by accelerate.
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                model_kwargs["dtype"] = torch.bfloat16

            model = model or AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        generated = output_ids[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return _truncate_at_stop(text, self.stop_sequences) if self.stop_sequences else text
