import unittest
from pathlib import Path

from indexer.repo_parser import RepoParser
from retrieval.dependency_retriever import DependencyRetriever, parse_import_modules

SAMPLE_REPO = Path(__file__).parent / "sample_repo"  # has a real cross-file import: pkg/module_b.py -> module_a.py


class ParseImportModulesTest(unittest.TestCase):
    def test_plain_import(self):
        self.assertEqual(parse_import_modules("import os", "app.py"), ["os"])

    def test_plain_import_multiple(self):
        self.assertEqual(parse_import_modules("import os, sys as s", "app.py"), ["os", "sys"])

    def test_from_import(self):
        self.assertEqual(parse_import_modules("from module_a import Greeter", "pkg/module_b.py"), ["module_a"])

    def test_from_import_dotted_module(self):
        self.assertEqual(parse_import_modules("from pkg.module_b import Foo", "app.py"), ["pkg.module_b"])

    def test_relative_from_import_with_module(self):
        self.assertEqual(parse_import_modules("from .module_b import Foo", "pkg/module_a.py"), ["pkg.module_b"])

    def test_relative_bare_import(self):
        self.assertEqual(parse_import_modules("from . import module_b", "pkg/module_a.py"), ["pkg.module_b"])

    def test_non_import_line_returns_empty(self):
        self.assertEqual(parse_import_modules("x = 1", "app.py"), [])


class DependencyRetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chunks = RepoParser(str(SAMPLE_REPO)).parse_repo()
        cls.retriever = DependencyRetriever([c.to_dict() for c in chunks])

    def test_resolves_local_module_import(self):
        deps = self.retriever.get_dependency_files("pkg/module_b.py")
        self.assertEqual(deps, {"module_a.py"})

    def test_search_returns_all_chunks_from_the_imported_file(self):
        results = self.retriever.search("pkg/module_b.py")
        file_paths = {r["file_path"] for r in results}
        self.assertEqual(file_paths, {"module_a.py"})
        names = {r["name"] for r in results}
        self.assertEqual(names, {"greet", "Greeter", "__init__", "_format", "shout"})

    def test_result_shape(self):
        results = self.retriever.search("pkg/module_b.py")
        self.assertGreater(len(results), 0)
        self.assertEqual(set(results[0].keys()), {"chunk_id", "file_path", "name", "score"})

    def test_file_with_only_external_imports_has_no_dependencies(self):
        # module_a.py only defines things, imports nothing local.
        self.assertEqual(self.retriever.get_dependency_files("module_a.py"), set())
        self.assertEqual(self.retriever.search("module_a.py"), [])

    def test_unknown_target_file_returns_empty(self):
        self.assertEqual(self.retriever.search("does/not/exist.py"), [])

    def test_top_k_is_respected(self):
        results = self.retriever.search("pkg/module_b.py", top_k=2)
        self.assertLessEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
