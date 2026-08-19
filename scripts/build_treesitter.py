#!/usr/bin/env python
"""Build/verify tree-sitter parsers for CrossCodeEval languages and record a manifest in data/.

Modern tree-sitter (>=0.22) ships each grammar as a prebuilt Python package
(tree-sitter-python, tree-sitter-java, tree-sitter-typescript, tree-sitter-c-sharp)
instead of compiling .so files from cloned grammar repos, so there is nothing to
compile here. This script just verifies each language loads and parses, then writes
a manifest so the rest of the pipeline (indexer/) can confirm parser availability
without re-importing every language package.
"""
import json
from pathlib import Path

import tree_sitter_python as ts_python
import tree_sitter_java as ts_java
import tree_sitter_typescript as ts_typescript
import tree_sitter_c_sharp as ts_csharp
from tree_sitter import Language, Parser

SAMPLES = {
    "python": (ts_python, ts_python.language(), b"def foo():\n    return 1\n"),
    "java": (ts_java, ts_java.language(), b"class Foo { int bar() { return 1; } }"),
    "typescript": (ts_typescript, ts_typescript.language_typescript(), b"function foo(): number { return 1; }"),
    "csharp": (ts_csharp, ts_csharp.language(), b"class Foo { int Bar() { return 1; } }"),
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST_PATH = DATA_DIR / "treesitter_languages.json"


def build_and_verify():
    manifest = {}
    for lang, (module, capsule, sample) in SAMPLES.items():
        language = Language(capsule)
        parser = Parser(language)
        tree = parser.parse(sample)
        assert tree.root_node.child_count > 0, f"{lang} parser produced an empty tree"
        manifest[lang] = {
            "package": module.__name__,
            "version": getattr(module, "__version__", "unknown"),
        }
        print(f"[ok] {lang}: parsed sample via {module.__name__}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    build_and_verify()
