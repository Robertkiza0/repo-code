# repo-context-completion

A repo-level code completion pipeline: parse a repository with tree-sitter, retrieve
relevant cross-file context for an incomplete line of code, have an LLM pick which
retrieved chunks actually matter, then generate the completion with that context. the target is CAN AN LLM SELECT A MINUMUN/USEFUL SUBSET OF REPOSITORY FRAGMENTS FROM A CANDIDATE POOL THAT GIVES TE CODE GENERATOR ENOUGH INFORMATION TO COMPLE THE CODE EFFECTIVELY

```
indexer  ->  retrieval  ->  selection  ->  generation
(parse)      (find candidates)  (Qwen picks useful ones)  (StarCoder completes)
```

Python is supported now; the parser is structured so other languages (Java, etc.)
can be added without touching the retrieval/selection/generation stages.

## Pipeline

### 1. Indexer (`indexer/`)

Recursively parses a repository with tree-sitter and extracts every class, function,
and method as a **chunk**: `chunk_id`, `file_path`, `language`, `type`, `name`,
`class_name`, `signature`, `docstring`, `imports`, exact `source_code`, and line range.

```bash
python -m indexer.cli path/to/repo -o repo_index.json
```

Adding a language means writing a `BaseExtractor` subclass and registering it in
`indexer/languages/__init__.py` — nothing else needs to change.

### 2. Retrieval (`retrieval/`)

Three independent retrieval sources, fused into one deduplicated candidate pool:

- **`BM25Retriever`** — keyword relevance over each chunk's name/signature/docstring/
  file_path/source_code.
- **`SymbolRetriever`** — extracts the identifier being typed at the cursor (e.g.
  `calculator.add(` from `result = calculator.add(`) and ranks exact/prefix symbol
  matches against chunk name/class_name/signature.
- **`DependencyRetriever`** — resolves a target file's local imports (using the
  `imports` text already recorded per chunk) and returns every chunk defined in the
  files it imports.
- **`CandidatePipeline`** — runs all three, deduplicates by `chunk_id`, caps the pool
  at `max_candidates` (default 12), and records which source(s) nominated each chunk
  and their scores.

```python
from retrieval.bm25_retriever import BM25Retriever
from retrieval.symbol_retriever import SymbolRetriever
from retrieval.dependency_retriever import DependencyRetriever
from retrieval.candidate_pipeline import CandidatePipeline

pipeline = CandidatePipeline(BM25Retriever(chunks), SymbolRetriever(chunks), DependencyRetriever(chunks))
candidates = pipeline.nominate("result = calculator.add(", target_file="app.py")
```

### 3. Selection (`selection/`)

`LLMSelector` asks a model which candidate chunks are actually useful context for
completing the code — it only ever picks `chunk_id`s, never writes a completion.
Any id the model returns that isn't in the real candidate pool (a hallucination) is
dropped rather than trusted; the full pool and the validated selection are logged
on every call.

Model access is pluggable via `selection/backends.py`:

- **`OllamaBackend`** (default) — calls a local Ollama server, model `qwen2.5-coder:7b`.
- **`HuggingFaceBackend`** — runs a `transformers` model (default
  `Qwen/Qwen2.5-Coder-7B-Instruct`) in-process, 4-bit quantized by default so it fits
  on a free-tier GPU (e.g. Colab's T4).

```python
from selection.llm_selector import LLMSelector

selector = LLMSelector(chunks)  # Ollama by default
result = selector.select(code_before_cursor, target_file, candidates)
# result["selected_chunk_ids"], result["candidate_chunk_ids"], result["rejected_hallucinated_ids"]
```

### 4. Generation (`generation/`)

`CompletionGenerator` looks up each selected chunk's `source_code` from the index,
assembles a repository-context + target-file prompt, and generates the completion.
Same pluggable-backend shape as selection (`OllamaGenerationBackend` /
`HuggingFaceGenerationBackend`, default `bigcode/starcoder2-3b`), but no chat
template or JSON constraint — StarCoder is a raw code-continuation model. Generation
stops at the first newline by default (`DEFAULT_STOP_SEQUENCES`), matching
single-line completion tasks and preventing the model from drifting into
hallucinated extra file blocks.

`generation/pipeline.py`'s `run_examples()` connects retrieval -> selection ->
generation for a batch of examples in one call:

```python
from generation.backends import OllamaGenerationBackend
from generation.generator import CompletionGenerator
from generation.pipeline import run_examples, save_results

generator = CompletionGenerator(chunks, OllamaGenerationBackend())
results = run_examples(chunks, examples, selector, generator)
save_results(results, "generation_results.json")
```

Each result: `target_file`, `selected_chunk_ids`, `candidate_chunk_ids`, `context`
(the full prompt), `completion`.

```bash
python -m generation.cli path/to/repo -o generation_results.json
```

## Project layout

```
indexer/       tree-sitter parsing -> repo_index.json chunks
retrieval/     BM25 / symbol / dependency retrieval + candidate pipeline
selection/     LLM picks useful candidates (Ollama or Hugging Face backend)
generation/    LLM generates the completion from selected context
data/          CrossCodeEval benchmark (gitignored -- see data/README)
examples/      demo repo (calculator.py/user.py/utils.py) + its repo_index.json
tests/         unit tests, incl. tests/sample_repo fixture with a real cross-file import
notebooks/     colab_demo.ipynb -- run the whole pipeline on a free GPU
scripts/       build_treesitter, colab_setup, raw-repo reconstruction helper
```

## Setup

```bash
pip install -r requirements.txt   # tree-sitter grammars, rank-bm25, transformers, etc.
pip install -e .                  # editable install so indexer/retrieval/selection/generation import from anywhere
```

`requirements.txt` intentionally excludes `torch`/`vllm`/`bitsandbytes` (heavy,
GPU/Linux-oriented) — install those separately (`pip install torch transformers
accelerate bitsandbytes`) only if using the Hugging Face backends.

## Running on GPU (Colab)

`notebooks/colab_demo.ipynb` runs the full pipeline end-to-end: installs Ollama,
gets the repo onto Colab, runs retrieval, selection (Ollama or Hugging Face
backend), and generation, saving `generation_results.json`. Open it at:

```
https://colab.research.google.com/github/<owner>/<repo>/blob/main/notebooks/colab_demo.ipynb
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Network/model-dependent tests (the live Ollama integration test in
`tests/test_llm_selector.py`) are guarded with `@unittest.skipUnless(...)` and skip
automatically if no local model server is reachable. Everything else runs fully
mocked, with no live model calls required.

## Data

`data/` holds the CrossCodeEval benchmark (line-completion tasks across
Python/Java/TypeScript/C#) — see `data/README` for details. It's gitignored: the
archive is large (~900MB extracted) and the dataset's own README asks that raw repo
source not be freely redistributed. `scripts/fetch_raw_repos.py` can reconstruct
referenced repos from their `metadata.repository` field (`{owner}-{repo}-{commit}`)
if needed, but this fetches hundreds of individual GitHub repos and is slow.
