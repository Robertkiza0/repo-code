"""Structured repository memory: a compact, deterministic graph of symbols
(classes/functions/methods), their attributes, and the static structural
relationships between them (containment, attribute definitions, calls,
references, imports) -- built entirely offline from the chunks the existing
indexer already produces (indexer.repo_parser.RepoParser /
indexer.models.Chunk). No new parser is introduced: every field used here
(name, type, class_name, signature, docstring, imports, source_code,
file_path) already exists on a Chunk. Import resolution reuses
retrieval.dependency_retriever.DependencyRetriever rather than re-deriving
it.

Never touches groundtruth/right_context/evaluation information -- this
module only ever sees the same `chunks` list retrieval already operates on,
plus (at query time) the same `code_before_cursor` prompt text every other
retriever already receives.

Relationships are only ever derived from things that can be checked
statically (chunk boundaries already given by the parser, `self.X =`
assignments in a class's own source, identifier tokens that literally
appear in a chunk's source and match another known symbol's name, and
import-line resolution) -- nothing is inferred semantically.
"""
import keyword
import random as _random_module
import re
from typing import Dict, List, Optional, Set, Tuple

from retrieval.bm25_retriever import tokenize
from retrieval.dependency_retriever import DependencyRetriever

_ATTRIBUTE_ASSIGN_RE = re.compile(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)")
_CLASS_BASES_RE = re.compile(r"class\s+\w+\s*\(([^)]*)\)")
_STOPWORDS = set(keyword.kwlist) | {"self", "cls"}

# Relations pointing the "reverse" direction of the ones stored in
# memory["relationships"], used only when formatting a candidate block from
# that candidate's own point of view (e.g. a method shown as "part_of" its
# class, rather than only ever seeing "ClassX contains MethodY"). Symmetric
# relations (same_class, same_file) aren't listed -- absent keys default to
# their own name, which is already correct for a symmetric relation.
_INVERSE_RELATION = {
    "contains": "part_of",
    "defines_attribute": "attribute_of",
    "depends_on": "depended_on_by",
    "inherits": "inherited_by",
    "calls": "called_by",
    "references": "referenced_by",
    "imports": "imported_by",
}


def _extract_base_class_names(signature: str) -> List[str]:
    """Best-effort extraction of a class's declared base names from its own
    signature text (e.g. "class LoudGreeter(Greeter):" -> ["Greeter"]) --
    already-available data, no new parsing. Dotted bases (e.g. "nn.Module")
    are reduced to their last component; keyword args like "metaclass=..."
    are skipped."""
    match = _CLASS_BASES_RE.search(signature or "")
    if not match:
        return []
    bases = []
    for part in match.group(1).split(","):
        token = part.strip()
        if not token or "=" in token:
            continue
        name = token.split(".")[-1].strip()
        if name.isidentifier():
            bases.append(name)
    return bases

# Bounds how much memory context ever reaches a prompt -- this is a
# selection aid, not a full graph dump (see module docstring / requirement
# "keep this compact").
DEFAULT_MAX_RELATIONSHIPS = 20
DEFAULT_MAX_RELATIONSHIP_LINES_PER_CANDIDATE = 5


def _qualified_name(chunk: Dict) -> str:
    class_name = chunk.get("class_name")
    return f"{class_name}.{chunk['name']}" if class_name else chunk["name"]


def build_repository_memory(chunks: List[Dict]) -> Dict:
    """Derives structured memory from the SAME chunks retrieval already
    uses -- purely a read-only transformation, no new parsing, no
    groundtruth. Deterministic: the same chunks always produce the same
    memory.

    Returns:
      {
        "symbols": {chunk_id: {name, qualified_name, type, file, signature,
                                class_name, methods, attributes}},
        "files": {file_path: {imports, classes, functions}},
        "name_index": {lowercased token: [chunk_id, ...]},
        "relationships": [{"source": name, "relation": str, "target": name}],
      }
    """
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    symbols: Dict[str, Dict] = {}
    for chunk in chunks:
        symbols[chunk["chunk_id"]] = {
            "name": chunk["name"],
            "qualified_name": _qualified_name(chunk),
            "type": chunk["type"],
            "file": chunk["file_path"],
            "signature": chunk.get("signature") or "",
            "class_name": chunk.get("class_name"),
            "methods": [],
            "attributes": [],
        }

    # Every relationship also carries the real chunk_id(s) of whichever
    # endpoint(s) are actual chunks (None for endpoints that aren't -- a
    # file path, or a bare attribute name) alongside the human-readable
    # name. The name is what gets displayed/matched against typed
    # identifiers; the chunk_id is what format_candidate_memory_block()
    # uses to pick out exactly this candidate's own edges, so two
    # same-named methods on different classes (e.g. two "reset" methods)
    # never bleed into each other's candidate block even though they'd
    # otherwise be indistinguishable by name alone.
    def _edge(source_name, source_chunk_id, relation, target_name, target_chunk_id=None):
        return {
            "source": source_name,
            "source_chunk_id": source_chunk_id,
            "relation": relation,
            "target": target_name,
            "target_chunk_id": target_chunk_id,
        }

    # class -> contains -> method (from the parser's own class_name field,
    # no new derivation needed) and class -> defines_attribute -> attribute
    # (from a lightweight "self.X =" scan of the class's own source, which
    # already includes all of its nested methods' bodies).
    relationships: List[Dict] = []
    for chunk in chunks:
        if chunk["type"] != "method" or not chunk.get("class_name"):
            continue
        owner = next((c for c in chunks if c["type"] == "class" and c["name"] == chunk["class_name"]), None)
        if owner is None:
            continue
        symbols[owner["chunk_id"]]["methods"].append(chunk["name"])
        relationships.append(
            _edge(owner["name"], owner["chunk_id"], "contains", chunk["name"], chunk["chunk_id"])
        )

    for chunk in chunks:
        if chunk["type"] != "class":
            continue
        attributes = sorted(set(_ATTRIBUTE_ASSIGN_RE.findall(chunk.get("source_code") or "")))
        symbols[chunk["chunk_id"]]["attributes"] = attributes
        for attr in attributes:
            relationships.append(_edge(chunk["name"], chunk["chunk_id"], "defines_attribute", attr))

    # class -> inherits -> base class, parsed directly from the class's own
    # signature (e.g. "class LoudGreeter(Greeter):") -- more precise than
    # the generic token-based "depends_on" below for this specific case, so
    # a base-class pair is recorded as "inherits" only, not both.
    inherits_pairs: Set[Tuple[str, str]] = set()
    class_chunks_by_name = {c["name"]: c for c in chunks if c["type"] == "class"}
    for chunk in chunks:
        if chunk["type"] != "class":
            continue
        for base_name in _extract_base_class_names(chunk.get("signature") or ""):
            base_chunk = class_chunks_by_name.get(base_name)
            if base_chunk is None or base_chunk["chunk_id"] == chunk["chunk_id"]:
                continue
            inherits_pairs.add((chunk["chunk_id"], base_chunk["chunk_id"]))
            relationships.append(
                _edge(chunk["name"], chunk["chunk_id"], "inherits", base_name, base_chunk["chunk_id"])
            )

    # files: imports/classes/functions, reusing DependencyRetriever for
    # import-line -> in-repo-file resolution rather than re-deriving it.
    files: Dict[str, Dict] = {}
    dependency_retriever = DependencyRetriever(chunks)
    for chunk in chunks:
        file_path = chunk["file_path"]
        entry = files.setdefault(file_path, {"imports": [], "classes": [], "functions": []})
        if not entry["imports"]:
            entry["imports"] = list(chunk.get("imports") or [])
        if chunk["type"] == "class" and chunk["name"] not in entry["classes"]:
            entry["classes"].append(chunk["name"])
        elif chunk["type"] == "function" and chunk["name"] not in entry["functions"]:
            entry["functions"].append(chunk["name"])

    for file_path in files:
        for target_file in sorted(dependency_retriever.get_dependency_files(file_path)):
            relationships.append(_edge(file_path, None, "imports", target_file))

    # symbol -> defined_in -> file (every top-level class/function; methods
    # are already reachable via their class's "contains" edge).
    for chunk in chunks:
        if chunk["type"] in ("class", "function"):
            relationships.append(_edge(chunk["name"], chunk["chunk_id"], "defined_in", chunk["file_path"]))

    # name_index: every symbol's own name/qualified_name/class_name, plus
    # every discovered attribute name -> its owning class's chunk_id, so a
    # bare identifier typed at the cursor (e.g. "token_repetition_penalty_max"
    # or the instance-variable-cased "settings") can resolve back to the
    # right symbol at query time.
    name_index: Dict[str, List[str]] = {}

    def _index(token: str, chunk_id: str) -> None:
        if not token:
            return
        key = token.lower()
        bucket = name_index.setdefault(key, [])
        if chunk_id not in bucket:
            bucket.append(chunk_id)

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        _index(chunk["name"], chunk_id)
        _index(_qualified_name(chunk), chunk_id)
        if chunk.get("class_name"):
            _index(chunk["class_name"], chunk_id)
    for chunk_id, info in symbols.items():
        for attr in info["attributes"]:
            _index(attr, chunk_id)

    # calls / references: for each chunk, find OTHER known symbol names that
    # literally appear as identifier tokens in its own source -- "calls" if
    # immediately followed by "(", "depends_on" if the referenced symbol is
    # a class, "references" otherwise. Case-insensitive so an instance
    # variable like "settings" links to a class named "Settings".
    #
    # Displayed names are bare (matching the requested schema), but every
    # edge below is still tagged with the exact chunk_id(s) involved -- so
    # even though "reset" might match several unrelated methods by name,
    # format_candidate_memory_block() only ever shows a candidate the edges
    # that really came from its own chunk_id, never another chunk's.
    name_to_chunk_ids: Dict[str, List[str]] = {}
    for chunk in chunks:
        name_to_chunk_ids.setdefault(chunk["name"].lower(), []).append(chunk["chunk_id"])

    call_re_cache: Dict[str, re.Pattern] = {}

    def _is_called(name: str, source: str) -> bool:
        pattern = call_re_cache.get(name)
        if pattern is None:
            pattern = re.compile(r"\b" + re.escape(name) + r"\s*\(")
            call_re_cache[name] = pattern
        return bool(pattern.search(source))

    seen_edges: Set[str] = set()
    for chunk in chunks:
        source_code = chunk.get("source_code") or ""
        tokens = {t for t in tokenize(source_code) if t not in _STOPWORDS}
        own_name_lower = chunk["name"].lower()
        # A class's own methods always appear as tokens in the class's full
        # source (it contains their `def` statements) -- that's already
        # the "contains" edge, not a separate call/reference from the class
        # to itself. Excluded by name (not just by matching class_name) so
        # a homonymous method on an unrelated class isn't spuriously linked
        # just because this class also happens to define a same-named one.
        own_method_names_lower = (
            {m.lower() for m in symbols[chunk["chunk_id"]]["methods"]} if chunk["type"] == "class" else set()
        )
        for token in tokens:
            if token == own_name_lower or token in own_method_names_lower or token not in name_to_chunk_ids:
                continue
            for target_chunk_id in name_to_chunk_ids[token]:
                target_chunk = chunk_by_id[target_chunk_id]
                if target_chunk_id == chunk["chunk_id"]:
                    continue
                if target_chunk["type"] == "class":
                    if (chunk["chunk_id"], target_chunk_id) in inherits_pairs:
                        continue  # already recorded, more precisely, as "inherits"
                    relation = "depends_on"
                elif _is_called(target_chunk["name"], source_code):
                    relation = "calls"
                else:
                    relation = "references"
                edge_key = f"{chunk['chunk_id']}|{relation}|{target_chunk_id}"
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                relationships.append(
                    _edge(chunk["name"], chunk["chunk_id"], relation, target_chunk["name"], target_chunk_id)
                )

    return {
        "symbols": symbols,
        "files": files,
        "name_index": name_index,
        "relationships": relationships,
    }


def pool_relationships(memory: Dict, candidate_chunk_ids: List[str]) -> List[Dict]:
    """Relationships from the full graph where at least one endpoint is a
    chunk_id in the given candidate pool -- i.e. "only show relationships
    involving the current candidate pool", not the whole repository graph,
    and independent of cursor text. Complements query_memory()'s
    cursor-driven expansion with the structural links between candidates
    the pool itself already contains -- e.g. "C2 uses C3" surfaces even if
    the incomplete code never literally mentions either name.
    """
    pool_set = set(candidate_chunk_ids)
    return [
        r
        for r in memory["relationships"]
        if r.get("source_chunk_id") in pool_set or r.get("target_chunk_id") in pool_set
    ]


def pool_structural_relationships(memory: Dict, candidate_chunk_ids: List[str]) -> List[Dict]:
    """same_class / same_file relationships computed directly between
    members of the CURRENT candidate pool only -- deliberately not
    precomputed globally in build_repository_memory() (same_file especially
    would be close to quadratic over a whole file's chunks); bounded by the
    pool size (typically 8-12), so this stays cheap and compact. Symmetric:
    emitted once per unordered pair.
    """
    relationships = []
    ids = list(dict.fromkeys(candidate_chunk_ids))
    for i, a_id in enumerate(ids):
        a = memory["symbols"].get(a_id)
        if a is None:
            continue
        for b_id in ids[i + 1 :]:
            b = memory["symbols"].get(b_id)
            if b is None:
                continue
            if a.get("class_name") and a["class_name"] == b.get("class_name"):
                relationships.append(
                    {
                        "source": a["name"],
                        "source_chunk_id": a_id,
                        "relation": "same_class",
                        "target": b["name"],
                        "target_chunk_id": b_id,
                    }
                )
            if a["file"] == b["file"]:
                relationships.append(
                    {
                        "source": a["name"],
                        "source_chunk_id": a_id,
                        "relation": "same_file",
                        "target": b["name"],
                        "target_chunk_id": b_id,
                    }
                )
    return relationships


def merge_relationships(*relationship_lists: List[Dict]) -> List[Dict]:
    """Concatenates relationship lists while dropping exact duplicates
    (same source/relation/target chunk_ids), preserving first-seen order."""
    seen = set()
    merged = []
    for relationships in relationship_lists:
        for r in relationships:
            key = (r["source"], r.get("source_chunk_id"), r["relation"], r["target"], r.get("target_chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
    return merged


def randomize_memory(memory: Dict, seed: int = 0) -> Dict:
    """Structural control condition for ablation ("LLM + random/noisy
    memory"): same number of relationships and the same relation label on
    each, but every target is reassigned to a uniformly random chunk_id
    from the graph -- so the prompt gets a graph-shaped but non-informative
    "structural relationships" section. Used to test whether real
    structural evidence helps selection, or whether any graph-like text
    changes behavior regardless of truthfulness. Deterministic for a given
    seed; only ever reshuffles a memory object already built without
    groundtruth -- never touches groundtruth itself.
    """
    rng = _random_module.Random(seed)
    chunk_ids = list(memory["symbols"].keys())
    if not chunk_ids:
        return {**memory, "relationships": []}

    shuffled = []
    for r in memory["relationships"]:
        random_target_id = rng.choice(chunk_ids)
        target_info = memory["symbols"][random_target_id]
        shuffled.append(
            {
                "source": r["source"],
                "source_chunk_id": r.get("source_chunk_id"),
                "relation": r["relation"],
                "target": target_info["name"],
                "target_chunk_id": random_target_id,
            }
        )
    return {**memory, "relationships": shuffled}


def query_memory(
    code_before_cursor: str, memory: Dict, max_relationships: int = DEFAULT_MAX_RELATIONSHIPS
) -> Dict:
    """Extracts identifier tokens from the incomplete code, matches them
    against memory's name_index (case-insensitive), and pulls in a bounded
    2-hop neighborhood of relationships around those matches: first the
    relationships touching a directly matched symbol, then the
    relationships touching whatever those introduced -- e.g. typing
    "settings.token_repetition_penalty_max" matches the Settings class
    (via the "settings"/attribute match), pulls in "X depends_on Settings",
    then a second hop surfaces X's own methods too. Never sees groundtruth
    -- only code_before_cursor, exactly like every other retriever.
    """
    tokens = {t for t in tokenize(code_before_cursor) if t not in _STOPWORDS}

    matched_chunk_ids: Set[str] = set()
    for token in tokens:
        matched_chunk_ids.update(memory["name_index"].get(token, []))

    matched_names: Set[str] = set()
    for chunk_id in matched_chunk_ids:
        info = memory["symbols"].get(chunk_id)
        if info is None:
            continue
        matched_names.add(info["name"])
        if info.get("class_name"):
            matched_names.add(info["class_name"])

    def _relationships_touching(names: Set[str]) -> List[Dict]:
        return [r for r in memory["relationships"] if r["source"] in names or r["target"] in names]

    first_hop = _relationships_touching(matched_names)

    expanded_names = set(matched_names)
    for r in first_hop:
        expanded_names.add(r["source"])
        expanded_names.add(r["target"])

    second_hop = [r for r in _relationships_touching(expanded_names) if r not in first_hop]

    all_relationships = (first_hop + second_hop)[:max_relationships]
    symbols_found = sorted(
        expanded_names | {r["source"] for r in all_relationships} | {r["target"] for r in all_relationships}
    )

    return {
        "matched_tokens": sorted(tokens),
        "symbols_found": symbols_found,
        "relationships": all_relationships,
    }


def format_candidate_memory_block(
    chunk_id: Optional[str],
    query_result: Dict,
    max_lines: int = DEFAULT_MAX_RELATIONSHIP_LINES_PER_CANDIDATE,
) -> Optional[str]:
    """Compact "structural relationships" block for one candidate, using
    only the relationships query_memory() already decided were relevant to
    the current incomplete code -- never the candidate's full relationship
    set (that could be large for a heavily-used class) and never anything
    beyond what's already in query_result. Returns None if the candidate's
    symbol has no relevant relationships to show, so callers can skip the
    section entirely for that candidate.

    Filters by the candidate's own chunk_id (not by name) so that two
    same-named symbols elsewhere in the repo (e.g. two different classes
    each with a "reset" method) never bleed into each other's block --
    every relationship's source_chunk_id/target_chunk_id is the real chunk
    it came from, set by build_repository_memory().
    """
    if not chunk_id:
        return None

    by_relation: Dict[str, List[str]] = {}
    for r in query_result["relationships"]:
        if r.get("source_chunk_id") == chunk_id:
            by_relation.setdefault(r["relation"], []).append(r["target"])
        elif r.get("target_chunk_id") == chunk_id:
            inverse = _INVERSE_RELATION.get(r["relation"], r["relation"])
            by_relation.setdefault(inverse, []).append(r["source"])

    if not by_relation:
        return None

    lines = []
    for relation, targets in by_relation.items():
        unique_targets = sorted(set(targets))
        lines.append(f"  {relation}: {', '.join(unique_targets)}")
        if len(lines) >= max_lines:
            break

    return "structural relationships:\n" + "\n".join(lines)
