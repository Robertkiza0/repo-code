import unittest
from pathlib import Path

from indexer.dependency_extractor import (
    _DEPENDENCY_CATEGORIES,
    attach_dependencies,
    extract_chunk_dependencies,
    print_chunk_dependencies,
    print_repository_dependencies,
)
from indexer.models import Chunk
from indexer.repo_parser import RepoParser

SAMPLE_REPO = Path(__file__).parent / "sample_repo"


def _chunks():
    return [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]


def _chunk_id(chunks, name, class_name=None):
    for c in chunks:
        if c["name"] == name and c.get("class_name") == class_name:
            return c["chunk_id"]
    raise LookupError(f"no chunk named {name!r} with class_name={class_name!r}")


class ExtractChunkDependenciesTest(unittest.TestCase):
    def test_every_chunk_gets_an_entry_with_all_categories(self):
        chunks = _chunks()
        dependencies = extract_chunk_dependencies(chunks)
        self.assertEqual(set(dependencies.keys()), {c["chunk_id"] for c in chunks})
        for entry in dependencies.values():
            self.assertEqual(set(entry.keys()), set(_DEPENDENCY_CATEGORIES))

    def test_contains_includes_methods_and_nested_classes(self):
        chunks = _chunks()
        dependencies = extract_chunk_dependencies(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        greet_id = _chunk_id(chunks, "greet", class_name="Greeter")

        contains = dependencies[greeter_id]["contains"]
        self.assertTrue(any(c["target"] == greet_id and c["symbol"] == "greet" for c in contains))

    def test_contains_finds_nested_class(self):
        chunks = _chunks()
        outer = Chunk(
            chunk_id="nested.py::Outer::class:1-20", file_path="nested.py", language="python",
            type="class", name="Outer", class_name=None, signature="class Outer:", docstring=None,
            imports=[], source_code="class Outer:\n    class Inner:\n        pass",
            start_line=1, end_line=20,
        ).to_dict()
        inner = Chunk(
            chunk_id="nested.py::Inner::class:2-3", file_path="nested.py", language="python",
            type="class", name="Inner", class_name=None, signature="class Inner:", docstring=None,
            imports=[], source_code="class Inner:\n    pass",
            start_line=2, end_line=3,
        ).to_dict()
        chunks = chunks + [outer, inner]
        dependencies = extract_chunk_dependencies(chunks)

        contains = dependencies["nested.py::Outer::class:1-20"]["contains"]
        self.assertTrue(any(c["target"] == "nested.py::Inner::class:2-3" for c in contains))

    def test_called_by_is_the_reverse_of_calls(self):
        chunks = _chunks()
        dependencies = extract_chunk_dependencies(chunks)
        format_id = _chunk_id(chunks, "_format")
        greet_method_id = _chunk_id(chunks, "greet", class_name="Greeter")

        called_by = dependencies[format_id]["called_by"]
        self.assertTrue(any(c["source"] == greet_method_id for c in called_by))

    def test_subclasses_is_the_reverse_of_inherits(self):
        chunks = _chunks()
        dependencies = extract_chunk_dependencies(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        loud_greeter_id = _chunk_id(chunks, "LoudGreeter")

        subclasses = dependencies[greeter_id]["subclasses"]
        self.assertEqual(len(subclasses), 1)
        self.assertEqual(subclasses[0]["source"], loud_greeter_id)

    def test_imported_by_is_the_reverse_of_imports(self):
        chunks = _chunks()
        dependencies = extract_chunk_dependencies(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")

        imported_by = dependencies[greeter_id]["imported_by"]
        self.assertTrue(any(i["importing_file"] == "pkg/module_b.py" and i["symbol"] == "Greeter" for i in imported_by))

    def test_imported_by_has_no_duplicate_entries_across_chunks_in_the_same_importing_file(self):
        # imports are file-scoped, so multiple chunks in the same importing
        # file would otherwise each contribute a duplicate imported_by entry.
        chunks = _chunks()
        dependencies = extract_chunk_dependencies(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")

        imported_by = dependencies[greeter_id]["imported_by"]
        keys = [(i["importing_file"], i["symbol"]) for i in imported_by]
        self.assertEqual(len(keys), len(set(keys)))

    def test_inherits_points_at_the_real_base_class_chunk(self):
        chunks = _chunks()
        dependencies = extract_chunk_dependencies(chunks)
        loud_greeter_id = _chunk_id(chunks, "LoudGreeter")
        greeter_id = _chunk_id(chunks, "Greeter")

        inherits = dependencies[loud_greeter_id]["inherits"]
        self.assertEqual(len(inherits), 1)
        self.assertEqual(inherits[0], {"target": greeter_id, "symbol": "Greeter"})

    def test_calls_points_at_the_real_target_chunk(self):
        chunks = _chunks()
        dependencies = extract_chunk_dependencies(chunks)
        format_id = _chunk_id(chunks, "_format")

        # Greeter.greet's own chunk calls _format()
        greet_method_id = _chunk_id(chunks, "greet", class_name="Greeter")
        calls = dependencies[greet_method_id]["calls"]
        self.assertTrue(any(c["target"] == format_id and c["symbol"] == "_format" for c in calls))

    def test_uses_object_kind_for_class_reference(self):
        chunks = _chunks()
        dependencies = extract_chunk_dependencies(chunks)
        loud_greeter_id = _chunk_id(chunks, "LoudGreeter")
        # LoudGreeter's base-class reference is "inherits", not duplicated in "uses"
        uses_targets = {(u["symbol"], u["kind"]) for u in dependencies[loud_greeter_id]["uses"]}
        self.assertNotIn(("Greeter", "object"), uses_targets)

    def test_no_attribute_kind_uses_without_a_pool(self):
        # Repo-wide (no pool_chunk_ids given) never produces kind="attribute"
        # entries -- see module docstring for why (common attribute names
        # collide across many unrelated classes at real-repo scale).
        chunks = _chunks()
        extra = Chunk(
            chunk_id="extra.py::uses_it::function:1-2", file_path="extra.py", language="python",
            type="function", name="uses_it", class_name=None,
            signature="def uses_it(g):", docstring=None, imports=[],
            source_code="def uses_it(g):\n    return g.default_greeting", start_line=1, end_line=2,
        ).to_dict()
        chunks = chunks + [extra]
        dependencies = extract_chunk_dependencies(chunks)

        for entry in dependencies.values():
            self.assertFalse(any(u["kind"] == "attribute" for u in entry["uses"]))

    def test_uses_attribute_kind_when_both_chunks_are_in_the_given_pool(self):
        chunks = _chunks()
        extra = Chunk(
            chunk_id="extra.py::uses_it::function:1-2", file_path="extra.py", language="python",
            type="function", name="uses_it", class_name=None,
            signature="def uses_it(g):", docstring=None, imports=[],
            source_code="def uses_it(g):\n    return g.default_greeting", start_line=1, end_line=2,
        ).to_dict()
        chunks = chunks + [extra]
        greeter_id = _chunk_id(chunks, "Greeter")
        pool = [greeter_id, "extra.py::uses_it::function:1-2"]

        dependencies = extract_chunk_dependencies(chunks, pool_chunk_ids=pool)

        uses = dependencies["extra.py::uses_it::function:1-2"]["uses"]
        attr_uses = [u for u in uses if u["kind"] == "attribute"]
        self.assertEqual(len(attr_uses), 1)
        self.assertEqual(attr_uses[0]["symbol"], "default_greeting")
        self.assertEqual(attr_uses[0]["owner_chunk_id"], greeter_id)
        self.assertEqual(attr_uses[0]["target"], f"{greeter_id}.default_greeting")

    def test_uses_attribute_kind_absent_when_owner_is_outside_the_pool(self):
        chunks = _chunks()
        extra = Chunk(
            chunk_id="extra.py::uses_it::function:1-2", file_path="extra.py", language="python",
            type="function", name="uses_it", class_name=None,
            signature="def uses_it(g):", docstring=None, imports=[],
            source_code="def uses_it(g):\n    return g.default_greeting", start_line=1, end_line=2,
        ).to_dict()
        chunks = chunks + [extra]
        pool = ["extra.py::uses_it::function:1-2"]  # Greeter (the owner) is NOT in the pool

        dependencies = extract_chunk_dependencies(chunks, pool_chunk_ids=pool)
        uses = dependencies["extra.py::uses_it::function:1-2"]["uses"]
        self.assertFalse(any(u["kind"] == "attribute" for u in uses))

    def test_imports_resolves_symbol_to_the_defining_chunk(self):
        chunks = _chunks()
        dependencies = extract_chunk_dependencies(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")
        loud_greeter_id = _chunk_id(chunks, "LoudGreeter")

        # pkg/module_b.py imports Greeter from module_a -- every chunk in
        # that file gets the file's resolved imports.
        imports = dependencies[loud_greeter_id]["imports"]
        self.assertTrue(any(i["symbol"] == "Greeter" and i["resolved_target"] == greeter_id for i in imports))

    def test_unresolvable_import_keeps_raw_info_with_none_target(self):
        chunks = _chunks()
        extra = Chunk(
            chunk_id="extra2.py::f::function:1-2", file_path="extra2.py", language="python",
            type="function", name="f", class_name=None,
            signature="def f():", docstring=None,
            imports=["from numpy import array"],
            source_code="def f():\n    return array([1])", start_line=1, end_line=2,
        ).to_dict()
        chunks = chunks + [extra]
        dependencies = extract_chunk_dependencies(chunks)

        imports = dependencies["extra2.py::f::function:1-2"]["imports"]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0]["symbol"], "array")
        self.assertIsNone(imports[0]["resolved_target"])
        # module must be preserved (not invented/dropped), even though unresolved
        self.assertEqual(imports[0]["module"], "numpy")

    def test_never_needs_groundtruth(self):
        import inspect

        sig = inspect.signature(extract_chunk_dependencies)
        self.assertEqual(list(sig.parameters), ["chunks", "pool_chunk_ids"])


class AttachDependenciesTest(unittest.TestCase):
    def test_preserves_existing_chunk_fields_and_adds_dependencies_key(self):
        chunks = _chunks()
        enriched = attach_dependencies(chunks)

        self.assertEqual(len(enriched), len(chunks))
        for original, updated in zip(chunks, enriched):
            for key, value in original.items():
                self.assertEqual(updated[key], value)
            self.assertIn("dependencies", updated)

    def test_does_not_mutate_the_input_chunks(self):
        chunks = _chunks()
        original_keys = [set(c.keys()) for c in chunks]
        attach_dependencies(chunks)
        self.assertEqual([set(c.keys()) for c in chunks], original_keys)


class PrintFunctionsTest(unittest.TestCase):
    def test_print_chunk_dependencies_matches_the_requested_format(self):
        from io import StringIO
        from unittest.mock import patch

        chunk = {
            "chunk_id": "generator.py::ExLlamaGenerator::class:1-10",
            "dependencies": {
                "calls": [{"target": "generator.py::ExLlamaGenerator.sample_current::method:1-2", "symbol": "sample_current"}],
                "uses": [
                    {"target": "generator.py::Settings::class:1-5", "symbol": "Settings", "kind": "object"},
                    {"target": "generator.py::Settings::class:1-5.token_repetition_penalty_max", "symbol": "token_repetition_penalty_max", "kind": "attribute"},
                ],
                "imports": [{"module": "model.py", "symbol": "Settings", "resolved_target": "model.py::Settings::class:1-5"}],
                "inherits": [{"target": "base.py::BaseGenerator::class:1-5", "symbol": "BaseGenerator"}],
            },
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_chunk_dependencies(chunk)
        output = buf.getvalue()

        self.assertIn("generator.py::ExLlamaGenerator::class:1-10", output)
        self.assertIn("calls:", output)
        self.assertIn("uses:", output)
        self.assertIn("imports:", output)
        self.assertIn("inherits:", output)
        self.assertIn("sample_current", output)
        self.assertIn("[attribute]", output)
        self.assertIn("[object]", output)

    def test_print_repository_dependencies_runs_on_a_real_enriched_repo(self):
        from io import StringIO
        from unittest.mock import patch

        chunks = _chunks()
        enriched = attach_dependencies(chunks)
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_repository_dependencies(enriched)
        self.assertIn("Greeter", buf.getvalue())

    def test_print_repository_dependencies_handles_chunks_without_dependencies(self):
        from io import StringIO
        from unittest.mock import patch

        buf = StringIO()
        with patch("sys.stdout", buf):
            print_repository_dependencies(_chunks())  # never ran attach_dependencies()
        self.assertIn("no dependencies found", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
