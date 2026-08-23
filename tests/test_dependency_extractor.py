import unittest
from pathlib import Path

from indexer.dependency_extractor import (
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
    def test_every_chunk_gets_an_entry_with_all_four_categories(self):
        chunks = _chunks()
        dependencies = extract_chunk_dependencies(chunks)
        self.assertEqual(set(dependencies.keys()), {c["chunk_id"] for c in chunks})
        for entry in dependencies.values():
            self.assertEqual(set(entry.keys()), {"calls", "uses", "imports", "inherits"})

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

    def test_uses_attribute_kind_for_cross_chunk_attribute_access(self):
        chunks = _chunks()
        extra = Chunk(
            chunk_id="extra.py::uses_it::function:1-2", file_path="extra.py", language="python",
            type="function", name="uses_it", class_name=None,
            signature="def uses_it(g):", docstring=None, imports=[],
            source_code="def uses_it(g):\n    return g.default_greeting", start_line=1, end_line=2,
        ).to_dict()
        chunks = chunks + [extra]
        dependencies = extract_chunk_dependencies(chunks)
        greeter_id = _chunk_id(chunks, "Greeter")

        uses = dependencies["extra.py::uses_it::function:1-2"]["uses"]
        attr_uses = [u for u in uses if u["kind"] == "attribute"]
        self.assertEqual(len(attr_uses), 1)
        self.assertEqual(attr_uses[0]["symbol"], "default_greeting")
        self.assertEqual(attr_uses[0]["owner_chunk_id"], greeter_id)
        self.assertEqual(attr_uses[0]["target"], f"{greeter_id}.default_greeting")

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
        self.assertEqual(list(sig.parameters), ["chunks"])


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
