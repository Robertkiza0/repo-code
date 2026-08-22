from abc import ABC, abstractmethod
from typing import Optional

import requests

DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 60.0

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


class SelectionBackend(ABC):
    """Turns a selection prompt into the model's raw text response.

    LLMSelector only depends on this interface, not on how/where the model
    actually runs -- swap backends without touching selection logic.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaBackend(SelectionBackend):
    """Calls a local (or remote) Ollama server's /api/generate endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
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


class HuggingFaceBackend(SelectionBackend):
    """Runs a transformers causal LM in-process (e.g. on a Colab GPU) instead
    of calling out to a separate Ollama server.

    torch/transformers are only imported when actually loading a model from
    its name (lazily, inside __init__) -- so importing this module, or using
    OllamaBackend, never requires them to be installed. For testing, an
    already-constructed tokenizer/model pair can be injected directly.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_HF_MODEL,
        device_map: Optional[object] = None,
        max_new_tokens: int = 512,
        load_in_4bit: bool = True,
        hf_token: Optional[str] = None,
        tokenizer: Optional[object] = None,
        model: Optional[object] = None,
    ):
        if tokenizer is None or model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "HuggingFaceBackend needs 'torch' and 'transformers' installed "
                    "(pip install torch transformers accelerate)."
                ) from e
            # hf_token=None is fine -- from_pretrained then falls back to
            # whatever huggingface_hub.login() already set up ambiently.
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name, token=hf_token)

            if device_map is None:
                # "auto" can make unpredictable partial-CPU-offload decisions
                # once a second model is already resident in VRAM (observed:
                # generation silently and massively slower after this backend
                # loaded alongside another). Pin explicitly to GPU 0 when one
                # exists; only fall back to "auto" on CPU-only machines.
                device_map = {"": 0} if torch.cuda.is_available() else "auto"
            model_kwargs = {"device_map": device_map, "token": hf_token}
            if torch.cuda.is_available():
                # PyTorch's built-in memory-efficient attention -- "eager"
                # (the default) allocates full O(seq_len^2) attention score
                # matrices, which can spike VRAM usage on longer prompts even
                # with 4-bit weights. sdpa needs no extra install and has no
                # accuracy tradeoff.
                model_kwargs["attn_implementation"] = "sdpa"
            if load_in_4bit:
                # In bf16 this model needs ~15GB -- basically all of a free-tier
                # T4's VRAM, which forces accelerate to silently offload layers
                # to CPU/disk (slow) even though a GPU is selected. 4-bit
                # quantization shrinks it to ~5GB so it fits with room to spare.
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
        messages = [{"role": "user", "content": prompt}]
        # return_dict=True makes the return type predictable (a BatchEncoding
        # with "input_ids"/"attention_mask" keys) -- without it, some
        # transformers/template versions return a bare tensor and others a
        # dict-like object, and the latter has no .shape attribute of its own.
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        generated = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
