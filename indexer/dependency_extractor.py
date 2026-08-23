"""Attaches per-chunk `dependencies` (calls/uses/imports/inherits) to the
chunks the existing indexer already produces, so they're inspectable
directly in a persisted repo_index.json.

Reuses memory.repository_memory's already-derived, already-tested
relationship graph (calls, references, uses_attribute, depends_on,
inherits) rather than re-deriving any of it -- this module only reshapes
that graph into the requested per-chunk schema. The one genuinely new
piece of resolution here is import-line -> specific-symbol-chunk matching
("from model import Settings" -> the chunk_id defining Settings in
model.py), which reuses retrieval.dependency_retriever's module-to-file
resolution (parse_import_modules(), DependencyRetriever's module index)
rather than re-deriving that either.

Deliberately does NOT modify indexer.repo_parser.RepoParser or
evaluation.cceval_adapter.locate_repo_index -- attach_dependencies() is a
separate, explicit, opt-in step: call it on a chunks list (from RepoParser
or an already-loaded repo_index.json) to get back chunk dicts with a
"dependencies" key added, then save/use that yourself. Every existing
consumer of the current index format (retrieval, selection, generation)
is unaffected either way, since this only ever adds a new key -- nothing
existing is renamed or removed. selection/llm_selector.py is untouched.

Never uses groundtruth/right_context/evaluation information -- only the
same chunks retrieval already operates on.
"""
import re
from typing import Dict, List, Tuple

from memory.repository_memory import build_repository_memory
from retrieval.dependency_retriever import DependencyRetriever, parse_import_modules

_FROM_IMPORT_SYMBOLS_RE = re.compile(r"^from\s+\.*[\w.]*\s*import\s+(.+)$")

_EMPTY_DEPENDENCIES = {"calls": [], "uses": [], "imports": [], "inherits": []}


def _extract_imported_symbol_names(import_line: str) -> List[str]:
    """"from X import a, b as c" -> ["a", "b"] (the real bound name, not
    its alias). Plain "import X" has no specific symbol to point at and is
    intentionally not included -- see module docstring."""
    match = _FROM_IMPORT_SYMBOLS_RE.match(import_line.strip().rstrip(";"))
    if not match:
        return []
    names = []
    for part in match.group(1).split(","):
        token = part.strip().split(" as ")[0].strip()
        if token and token != "*" and token.isidentifier():
            names.append(token)
    return names


def _resolve_imports_for_file(
    file_path: str,
    import_lines: List[str],
    chunks_by_file_and_name: Dict[Tuple[str, str], str],
    module_to_file: Dict[str, str],
) -> List[Dict]:
    resolved = []
    for import_line in import_lines:
        symbols = _extract_imported_symbol_names(import_line)
        if not symbols:
            continue  # plain "import X" -- no specific symbol to resolve
        for module in parse_import_modules(import_line, file_path):
            target_file = module_to_file.get(module)
            for symbol in symbols:
                resolved_target = (
                    chunks_by_file_and_name.get((target_file, symbol)) if target_file is not None else None
                )
                resolved.append(
                    {
                        "module": target_file if target_file is not None else module,
                        "symbol": symbol,
                        # Preserve the unresolved case rather than inventing a
                        # target: external/stdlib imports (numpy, torch, ...)
                        # or a repo file where that exact symbol isn't
                        # actually defined both leave this None.
                        "resolved_target": resolved_target,
                    }
                )
    return resolved


def extract_chunk_dependencies(chunks: List[Dict]) -> Dict[str, Dict]:
    """Returns {chunk_id: {"calls": [...], "uses": [...], "imports": [...],
    "inherits": [...]}}:

      calls:    [{"target": chunk_id, "symbol": name}]
      uses:     [{"target": chunk_id_or_dotted_attr, "symbol": name,
                  "kind": "object"|"attribute", "owner_chunk_id": chunk_id}]
                 (owner_chunk_id is only present for kind="attribute",
                  since "target" there is a synthetic "<owner>.<attr>"
                  string, not itself a real chunk_id)
      imports:  [{"module": file_path_or_raw_module, "symbol": name,
                  "resolved_target": chunk_id|None}]
      inherits: [{"target": chunk_id, "symbol": name}]

    Every "target"/"resolved_target" that IS a chunk_id is guaranteed to
    exist in `chunks` -- nothing here is invented; unresolved references
    keep their raw module/symbol text with resolved_target=None instead.
    """
    memory = build_repository_memory(chunks)
    dependencies: Dict[str, Dict] = {c["chunk_id"]: {"calls": [], "uses": [], "imports": [], "inherits": []} for c in chunks}

    for r in memory["relationships"]:
        source_chunk_id = r.get("source_chunk_id")
        if source_chunk_id is None or source_chunk_id not in dependencies:
            continue

        if r["relation"] == "calls":
            dependencies[source_chunk_id]["calls"].append({"target": r["target_chunk_id"], "symbol": r["target"]})
        elif r["relation"] in ("references", "depends_on"):
            dependencies[source_chunk_id]["uses"].append(
                {"target": r["target_chunk_id"], "symbol": r["target"], "kind": "object"}
            )
        elif r["relation"] == "uses_attribute":
            owner_chunk_id = r["target_chunk_id"]
            dependencies[source_chunk_id]["uses"].append(
                {
                    "target": f"{owner_chunk_id}.{r['target']}",
                    "symbol": r["target"],
                    "kind": "attribute",
                    "owner_chunk_id": owner_chunk_id,
                }
            )
        elif r["relation"] == "inherits":
            dependencies[source_chunk_id]["inherits"].append({"target": r["target_chunk_id"], "symbol": r["target"]})

    # imports: per-file, symbol-level resolution -- reuses
    # DependencyRetriever's module index rather than re-deriving it.
    dependency_retriever = DependencyRetriever(chunks)
    module_to_file: Dict[str, str] = dependency_retriever._module_to_file
    chunks_by_file_and_name = {(c["file_path"], c["name"]): c["chunk_id"] for c in chunks}

    imports_by_file: Dict[str, List[Dict]] = {}
    for file_path, info in memory["files"].items():
        imports_by_file[file_path] = _resolve_imports_for_file(
            file_path, info["imports"], chunks_by_file_and_name, module_to_file
        )
    for chunk in chunks:
        dependencies[chunk["chunk_id"]]["imports"] = imports_by_file.get(chunk["file_path"], [])

    return dependencies


def attach_dependencies(chunks: List[Dict]) -> List[Dict]:
    """Returns NEW chunk dicts (shallow copies of the input, in the same
    order) each with a "dependencies" key added -- the existing chunk
    representation (chunk_id, file_path, type, name, signature, docstring,
    imports, source_code, ...) is otherwise completely untouched, and the
    input list/dicts are not mutated.
    """
    dependencies = extract_chunk_dependencies(chunks)
    return [{**chunk, "dependencies": dependencies[chunk["chunk_id"]]} for chunk in chunks]


def print_chunk_dependencies(chunk: Dict) -> None:
    deps = chunk.get("dependencies") or _EMPTY_DEPENDENCIES
    print(chunk["chunk_id"])
    for category in ("calls", "uses", "imports", "inherits"):
        items = deps.get(category) or []
        if not items:
            continue
        print(f"  {category}:")
        for item in items:
            if category == "imports":
                target = item.get("resolved_target") or f"{item['module']} (unresolved)"
                print(f"    -> {target}  (symbol={item['symbol']})")
            else:
                kind = f"  [{item['kind']}]" if "kind" in item else ""
                print(f"    -> {item['target']}{kind}")


def print_repository_dependencies(chunks: List[Dict]) -> None:
    """Debug/validation view: prints every chunk that has at least one
    dependency of any kind, in the format:

        generator.py::ExLlamaGenerator
          calls:
            -> generator.py::ExLlamaGenerator.sample_current
          ...

    Chunks with dependencies attached (via attach_dependencies()) show
    their real chunk_ids; chunks without a "dependencies" key are silently
    skipped rather than erroring, so this also works on a plain chunks
    list that hasn't been through attach_dependencies() yet (it'll just
    print nothing).
    """
    shown = 0
    for chunk in chunks:
        deps = chunk.get("dependencies")
        if deps and any(deps.values()):
            print_chunk_dependencies(chunk)
            print()
            shown += 1
    if shown == 0:
        print("(no dependencies found -- did you run attach_dependencies() first?)")
