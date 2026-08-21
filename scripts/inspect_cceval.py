#!/usr/bin/env python
"""Inspect one example from a CrossCodeEval line_completion jsonl file.

Prints top-level field names, types, and string lengths -- never the actual
prompt/groundtruth/right_context text, which can be very large.
"""
import argparse
import json
from pathlib import Path

# Resolved relative to this script's own location (the repo root, one level
# up from scripts/), not the current working directory -- so the default
# works whether run as `python scripts/inspect_cceval.py` from the repo root,
# via VS Code's "Run Python File" (which uses the file's own folder as cwd),
# or from anywhere else.
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data/cceval/samples/line_completion_20.jsonl"


def describe(value) -> str:
    """A short type/size description of value -- string content itself is
    never included, only its length; short nested fields still show shape."""
    if isinstance(value, str):
        return f"str, length={len(value)}"
    if isinstance(value, dict):
        return f"dict, {len(value)} field(s): {list(value.keys())}"
    if isinstance(value, list):
        return f"list, {len(value)} item(s)"
    return f"{type(value).__name__}, value={value!r}"


def inspect_example(example: dict) -> None:
    print(f"Top-level fields ({len(example)}):\n")
    for key, value in example.items():
        print(f"  {key}: {describe(value)}")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"      {sub_key}: {describe(sub_value)}")


def load_example(path: Path, index: int) -> dict:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"{path} has fewer than {index + 1} line(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        default=DEFAULT_PATH,
        help=f"Path to a jsonl file (default: {DEFAULT_PATH})",
    )
    parser.add_argument("-n", "--index", type=int, default=0, help="Which example (0-based line) to inspect")
    args = parser.parse_args()

    path = Path(args.jsonl_path)
    example = load_example(path, args.index)

    print(f"File: {path}  (example #{args.index})\n")
    inspect_example(example)


if __name__ == "__main__":
    main()
