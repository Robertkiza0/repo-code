#!/usr/bin/env bash
# Build/verify tree-sitter parsers for Python, Java, TypeScript, and C#.
#
# Note: unlike the original cceval script, this does not clone grammar repos
# or compile .so files via Language.build_library() -- that API was removed in
# tree-sitter >=0.22 and requires a C compiler. Instead it relies on the
# prebuilt tree-sitter-<lang> packages from requirements.txt and just verifies
# they load correctly, writing a manifest to data/treesitter_languages.json.
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/build_treesitter.py
