import json
import tempfile
import unittest
from pathlib import Path

from indexer.repo_parser import RepoParser

SAMPLE_REPO = Path(__file__).parent / "sample_repo"


class RepoParserPythonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = RepoParser(str(SAMPLE_REPO))
        cls.chunks = cls.parser.parse_repo()
        cls.by_key = {(c.file_path, c.type, c.class_name, c.name): c for c in cls.chunks}

    def _get(self, file_path, chunk_type, class_name, name):
        chunk = self.by_key.get((file_path, chunk_type, class_name, name))
        self.assertIsNotNone(
            chunk, f"missing chunk {file_path=} {chunk_type=} {class_name=} {name=} in {list(self.by_key)}"
        )
        return chunk

    def test_finds_expected_chunk_count(self):
        # module_a.py: greet(fn), Greeter(class), __init__, greet(method), _format(nested fn), shout
        # pkg/__init__.py: none
        # pkg/module_b.py: LoudGreeter(class), greet(method)
        self.assertEqual(len(self.chunks), 8)

    def test_module_level_function(self):
        chunk = self._get("module_a.py", "function", None, "greet")
        self.assertEqual(chunk.language, "python")
        self.assertEqual(chunk.class_name, None)
        self.assertIn('def greet(name: str, greeting: str = "Hello") -> str:', chunk.signature)
        self.assertEqual(chunk.docstring, "Return a greeting for name.")
        self.assertIn("import os", chunk.imports)
        self.assertIn("from typing import List, Optional", chunk.imports)
        self.assertTrue(chunk.source_code.startswith("def greet("))
        self.assertIn('return f"{greeting}, {name}!"', chunk.source_code)

    def test_class_chunk(self):
        chunk = self._get("module_a.py", "class", None, "Greeter")
        self.assertEqual(chunk.signature, "class Greeter:")
        self.assertEqual(chunk.docstring, "A simple greeter class.")
        self.assertTrue(chunk.source_code.startswith("class Greeter:"))

    def test_method_vs_module_function_with_same_name(self):
        method = self._get("module_a.py", "method", "Greeter", "greet")
        self.assertEqual(method.class_name, "Greeter")
        self.assertIn("def greet(self, name: str) -> str:", method.signature)
        self.assertEqual(method.docstring, "Greet name using the default greeting.")
        # the module-level function with the same name must be a distinct chunk
        function = self._get("module_a.py", "function", None, "greet")
        self.assertNotEqual(method.chunk_id, function.chunk_id)

    def test_nested_function_is_a_function_not_a_method(self):
        chunk = self._get("module_a.py", "function", None, "_format")
        self.assertEqual(chunk.type, "function")
        self.assertIsNone(chunk.class_name)

    def test_decorated_method_keeps_decorator_in_source(self):
        chunk = self._get("module_a.py", "method", "Greeter", "shout")
        self.assertIn("@staticmethod", chunk.source_code)
        self.assertNotIn("@staticmethod", chunk.signature)

    def test_cross_file_inherited_class(self):
        chunk = self._get("pkg/module_b.py", "class", None, "LoudGreeter")
        self.assertEqual(chunk.signature, "class LoudGreeter(Greeter):")
        self.assertIn("from module_a import Greeter", chunk.imports)

    def test_chunk_ids_are_unique(self):
        ids = [c.chunk_id for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_source_code_is_preserved_exactly(self):
        file_path = SAMPLE_REPO / "module_a.py"
        original_lines = file_path.read_text(encoding="utf-8").splitlines()
        chunk = self._get("module_a.py", "function", None, "greet")
        expected = "\n".join(original_lines[chunk.start_line - 1 : chunk.end_line])
        self.assertEqual(chunk.source_code, expected)

    def test_save_index_round_trips_to_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "repo_index.json"
            self.parser.save_index(self.chunks, str(output_path))
            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data), len(self.chunks))
            self.assertEqual(data[0]["chunk_id"], self.chunks[0].chunk_id)
            self.assertIn("source_code", data[0])


if __name__ == "__main__":
    unittest.main()
