import unittest
from pathlib import Path

from indexer.repo_parser import RepoParser
from memory.repository_memory import (
    build_repository_memory,
    format_candidate_memory_block,
    merge_relationships,
    pool_relationships,
    pool_structural_relationships,
    query_memory,
    randomize_memory,
)

SAMPLE_REPO = Path(__file__).parent / "sample_repo"


def _chunks():
    return [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]


def _chunk_id(chunks, name, class_name=None):
    for c in chunks:
        if c["name"] == name and c.get("class_name") == class_name:
            return c["chunk_id"]
    raise LookupError(f"no chunk named {name!r} with class_name={class_name!r}")


class BuildRepositoryMemoryTest(unittest.TestCase):
    def test_symbols_capture_type_file_signature_and_class_name(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        info = memory["symbols"][greeter_id]
        self.assertEqual(info["type"], "class")
        self.assertEqual(info["file"], "module_a.py")
        self.assertIn("class Greeter", info["signature"])
        self.assertIsNone(info["class_name"])

    def test_class_captures_its_own_methods_and_attributes(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        info = memory["symbols"][greeter_id]
        self.assertEqual(set(info["methods"]), {"__init__", "greet", "shout"})
        self.assertEqual(info["attributes"], ["default_greeting"])

    def test_contains_relationship_for_every_method(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        contains = [r for r in memory["relationships"] if r["relation"] == "contains" and r["source"] == "Greeter"]
        targets = {r["target"] for r in contains}
        self.assertEqual(targets, {"__init__", "greet", "shout"})

    def test_defines_attribute_relationship(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        edges = [r for r in memory["relationships"] if r["relation"] == "defines_attribute"]
        self.assertIn({"source": "Greeter", "relation": "defines_attribute", "target": "default_greeting",
                        "source_chunk_id": _chunk_id(chunks, "Greeter"), "target_chunk_id": None}, edges)

    def test_import_relationship_resolves_in_repo_file(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        imports = [r for r in memory["relationships"] if r["relation"] == "imports"]
        self.assertTrue(any(r["source"] == "pkg/module_b.py" and r["target"] == "module_a.py" for r in imports))

    def test_subclass_relationship_to_its_base_class(self):
        # A base-class reference is now the more precise "inherits" (see
        # InheritsRelationTest below), not the generic "depends_on".
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        inherits = [r for r in memory["relationships"] if r["relation"] == "inherits"]
        self.assertTrue(any(r["source"] == "LoudGreeter" and r["target"] == "Greeter" for r in inherits))

    def test_class_does_not_spuriously_call_its_own_methods(self):
        # Greeter's own chunk source contains the text of its methods' def
        # statements (e.g. "def greet(...)") -- that must not produce a
        # "Greeter calls greet" edge; it's already the "contains" edge.
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        bad_edges = [
            r for r in memory["relationships"]
            if r["source"] == "Greeter" and r["relation"] in ("calls", "references") and r["target"] in {"__init__", "greet", "shout"}
        ]
        self.assertEqual(bad_edges, [])

    def test_calls_relationship_detected_for_actual_call(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        calls = [r for r in memory["relationships"] if r["relation"] == "calls" and r["target"] == "_format"]
        self.assertTrue(len(calls) >= 1)

    def test_uses_attribute_relationship_for_attribute_access_outside_the_owning_class(self):
        # "settings.token_repetition_penalty_max"-style access: a chunk
        # elsewhere in the repo references an attribute by name, even
        # though that name is never a class/function/method name on its own.
        from indexer.models import Chunk

        chunks = _chunks()
        extra = Chunk(
            chunk_id="extra.py::uses_it::function:1-2", file_path="extra.py", language="python",
            type="function", name="uses_it", class_name=None,
            signature="def uses_it(g):", docstring=None, imports=[],
            source_code="def uses_it(g):\n    return g.default_greeting", start_line=1, end_line=2,
        ).to_dict()
        chunks = chunks + [extra]

        memory = build_repository_memory(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        edges = [
            r for r in memory["relationships"]
            if r["relation"] == "uses_attribute" and r["source_chunk_id"] == "extra.py::uses_it::function:1-2"
        ]
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["target"], "default_greeting")
        self.assertEqual(edges[0]["target_chunk_id"], greeter_id)

    def test_class_does_not_get_a_uses_attribute_edge_to_its_own_attribute(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        self_edges = [
            r for r in memory["relationships"]
            if r["relation"] == "uses_attribute" and r["source_chunk_id"] == greeter_id and r["target_chunk_id"] == greeter_id
        ]
        self.assertEqual(self_edges, [])

    def test_relationships_carry_exact_chunk_id_provenance(self):
        # Two different "greet" chunks (module-level function vs.
        # Greeter.greet method) must not share chunk_id-tagged edges.
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        module_greet_id = _chunk_id(chunks, "greet", class_name=None)
        method_greet_id = _chunk_id(chunks, "greet", class_name="Greeter")
        self.assertNotEqual(module_greet_id, method_greet_id)

        module_greet_edges = [
            r for r in memory["relationships"]
            if r.get("source_chunk_id") == module_greet_id or r.get("target_chunk_id") == module_greet_id
        ]
        # the module-level function is never "contained" by any class
        self.assertFalse(any(r["relation"] == "contains" and r["target_chunk_id"] == module_greet_id for r in module_greet_edges))

        method_greet_edges = [
            r for r in memory["relationships"]
            if r.get("source_chunk_id") == method_greet_id or r.get("target_chunk_id") == method_greet_id
        ]
        self.assertTrue(any(r["relation"] == "contains" and r["source"] == "Greeter" for r in method_greet_edges))

    def test_name_index_resolves_attribute_name_to_owning_class(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        self.assertIn(greeter_id, memory["name_index"].get("default_greeting", []))

    def test_files_capture_imports_classes_and_functions(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        module_b = memory["files"]["pkg/module_b.py"]
        self.assertEqual(module_b["classes"], ["LoudGreeter"])
        self.assertTrue(any("import" in line for line in module_b["imports"]))

    def test_never_needs_or_references_groundtruth(self):
        # Structural test: build_repository_memory's signature only ever
        # accepts chunks -- there is no parameter through which groundtruth
        # could reach it.
        import inspect

        sig = inspect.signature(build_repository_memory)
        self.assertEqual(list(sig.parameters), ["chunks"])


class QueryMemoryTest(unittest.TestCase):
    def test_matches_directly_typed_symbol(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        result = query_memory("result = Greeter(", memory)
        self.assertIn("Greeter", result["symbols_found"])

    def test_matches_attribute_and_expands_to_owning_class_and_its_relationships(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        result = query_memory("x = obj.default_greeting", memory)
        self.assertIn("Greeter", result["symbols_found"])
        # second hop: Greeter's own contains-edges should surface too
        self.assertTrue(any(r["source"] == "Greeter" and r["relation"] == "contains" for r in result["relationships"]))

    def test_no_match_gives_empty_result_not_a_crash(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        result = query_memory("totally_unrelated_xyz123", memory)
        self.assertEqual(result["relationships"], [])

    def test_respects_max_relationships_cap(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        result = query_memory("Greeter", memory, max_relationships=2)
        self.assertLessEqual(len(result["relationships"]), 2)

    def test_never_receives_groundtruth_parameter(self):
        import inspect

        sig = inspect.signature(query_memory)
        self.assertEqual(list(sig.parameters), ["code_before_cursor", "memory", "max_relationships"])


class FormatCandidateMemoryBlockTest(unittest.TestCase):
    def test_returns_none_for_missing_chunk_id(self):
        self.assertIsNone(format_candidate_memory_block(None, {"relationships": []}))

    def test_returns_none_when_no_relationships_touch_this_chunk(self):
        block = format_candidate_memory_block("some_chunk_id", {"relationships": []})
        self.assertIsNone(block)

    def test_disambiguates_same_named_chunks_by_chunk_id(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        module_greet_id = _chunk_id(chunks, "greet", class_name=None)
        method_greet_id = _chunk_id(chunks, "greet", class_name="Greeter")

        query_result = query_memory("result = Greeter(", memory)
        module_block = format_candidate_memory_block(module_greet_id, query_result)
        method_block = format_candidate_memory_block(method_greet_id, query_result)

        # the module-level function must never claim to be "part_of" a class
        if module_block:
            self.assertNotIn("part_of", module_block)
        self.assertIsNotNone(method_block)
        self.assertIn("part_of: Greeter", method_block)

    def test_output_is_compact_and_capped(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        query_result = query_memory("result = Greeter(", memory)
        block = format_candidate_memory_block(greeter_id, query_result, max_lines=1)
        self.assertEqual(len(block.splitlines()), 2)  # header + exactly 1 relation line


class InheritsRelationTest(unittest.TestCase):
    def test_subclass_inherits_base_class(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        inherits = [r for r in memory["relationships"] if r["relation"] == "inherits"]
        self.assertTrue(any(r["source"] == "LoudGreeter" and r["target"] == "Greeter" for r in inherits))

    def test_inherited_pair_is_not_also_recorded_as_generic_depends_on(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        depends_on = [
            r for r in memory["relationships"]
            if r["relation"] == "depends_on" and r["source"] == "LoudGreeter" and r["target"] == "Greeter"
        ]
        self.assertEqual(depends_on, [])


class PoolRelationshipsTest(unittest.TestCase):
    def test_only_includes_edges_touching_the_pool(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        loud_greeter_id = _chunk_id(chunks, "LoudGreeter")

        edges = pool_relationships(memory, [greeter_id, loud_greeter_id])
        for r in edges:
            self.assertTrue(r.get("source_chunk_id") in {greeter_id, loud_greeter_id} or r.get("target_chunk_id") in {greeter_id, loud_greeter_id})
        self.assertTrue(any(r["relation"] == "inherits" for r in edges))

    def test_empty_pool_gives_no_edges(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        self.assertEqual(pool_relationships(memory, []), [])


class PoolStructuralRelationshipsTest(unittest.TestCase):
    def test_same_class_for_sibling_methods(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        init_id = _chunk_id(chunks, "__init__", class_name="Greeter")
        greet_id = _chunk_id(chunks, "greet", class_name="Greeter")

        edges = pool_structural_relationships(memory, [init_id, greet_id])
        same_class = [r for r in edges if r["relation"] == "same_class"]
        self.assertEqual(len(same_class), 1)

    def test_same_file_for_chunks_in_the_same_file(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        format_id = _chunk_id(chunks, "_format")

        edges = pool_structural_relationships(memory, [greeter_id, format_id])
        same_file = [r for r in edges if r["relation"] == "same_file"]
        self.assertEqual(len(same_file), 1)

    def test_no_relation_for_unrelated_pair(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        loud_greeter_id = _chunk_id(chunks, "LoudGreeter")  # different file, different class

        edges = pool_structural_relationships(memory, [greeter_id, loud_greeter_id])
        self.assertEqual(edges, [])


class MergeRelationshipsTest(unittest.TestCase):
    def test_drops_exact_duplicates_preserving_order(self):
        a = {"source": "X", "source_chunk_id": "x1", "relation": "calls", "target": "Y", "target_chunk_id": "y1"}
        b = {"source": "X", "source_chunk_id": "x1", "relation": "calls", "target": "Z", "target_chunk_id": "z1"}
        merged = merge_relationships([a], [a, b])
        self.assertEqual(merged, [a, b])


class RandomizeMemoryTest(unittest.TestCase):
    def test_same_edge_count_and_relation_labels_different_targets(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        randomized = randomize_memory(memory, seed=7)

        self.assertEqual(len(randomized["relationships"]), len(memory["relationships"]))
        original_relations = [r["relation"] for r in memory["relationships"]]
        randomized_relations = [r["relation"] for r in randomized["relationships"]]
        self.assertEqual(original_relations, randomized_relations)  # same shape/labels

    def test_deterministic_for_a_given_seed(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        r1 = randomize_memory(memory, seed=42)
        r2 = randomize_memory(memory, seed=42)
        self.assertEqual(r1["relationships"], r2["relationships"])

    def test_different_seeds_can_give_different_targets(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        r1 = randomize_memory(memory, seed=1)
        r2 = randomize_memory(memory, seed=2)
        targets_1 = [r["target_chunk_id"] for r in r1["relationships"]]
        targets_2 = [r["target_chunk_id"] for r in r2["relationships"]]
        self.assertNotEqual(targets_1, targets_2)

    def test_never_needs_groundtruth(self):
        import inspect

        sig = inspect.signature(randomize_memory)
        self.assertEqual(list(sig.parameters), ["memory", "seed"])


if __name__ == "__main__":
    unittest.main()
