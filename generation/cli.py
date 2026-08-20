import argparse

from indexer.repo_parser import RepoParser
from selection.llm_selector import LLMSelector
from generation.backends import OllamaGenerationBackend
from generation.generator import CompletionGenerator
from generation.pipeline import run_examples, save_results

# Demo completion examples against tests/sample_repo. "target_file" and
# "code_before_cursor" are the same shape CandidatePipeline.nominate() and
# LLMSelector.select() already take -- swap this list for your own examples.
DEFAULT_EXAMPLES = [
    {"target_file": "pkg/module_b.py", "code_before_cursor": "result = Greeter("},
    {"target_file": "module_a.py", "code_before_cursor": "class Greeter:\n    def __init__(self, default_greeting: str = "},
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval -> Qwen selection -> StarCoder generation over a repo's examples."
    )
    parser.add_argument("repo_path", help="Path to the repository to index and generate completions for")
    parser.add_argument("-o", "--output", default="generation_results.json", help="Output JSON path")
    args = parser.parse_args()

    chunks = [c.to_dict() for c in RepoParser(args.repo_path).parse_repo()]
    selector = LLMSelector(chunks)  # defaults to Ollama qwen2.5-coder:7b
    generator = CompletionGenerator(chunks, OllamaGenerationBackend())  # defaults to Ollama starcoder2:3b

    results = run_examples(chunks, DEFAULT_EXAMPLES, selector, generator)
    save_results(results, args.output)
    print(f"Saved {len(results)} completion result(s) to {args.output}")


if __name__ == "__main__":
    main()
