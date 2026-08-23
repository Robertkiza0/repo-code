import unittest
from pathlib import Path

from indexer.repo_parser import RepoParser
from memory.repository_memory import (
    build_repository_memory,
    format_candidate_memory_block,
    query_memory,
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

    def test_depends_on_relationship_for_subclass(self):
        chunks = _chunks()
        memory = build_repository_memory(chunks)
        depends_on = [r for r in memory["relationships"] if r["relation"] == "depends_on"]
        self.assertTrue(any(r["source"] == "LoudGreeter" and r["target"] == "Greeter" for r in depends_on))

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


if __name__ == "__main__":
    unittest.main()
