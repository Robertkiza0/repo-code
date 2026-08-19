import unittest
from pathlib import Path

from indexer.repo_parser import RepoParser
from retrieval.symbol_retriever import SymbolRetriever, extract_symbol, split_symbol

SAMPLE_REPO = Path(__file__).parent.parent / "examples" / "sample_repo"


class ExtractSymbolTest(unittest.TestCase):
    """extract_symbol() in isolation, for simple and dotted identifiers."""

    def test_simple_identifier_with_open_call_paren(self):
        self.assertEqual(extract_symbol("multiply("), "multiply")

    def test_simple_identifier_without_call(self):
        self.assertEqual(extract_symbol("result = average"), "average")

    def test_dotted_identifier_with_open_call_paren(self):
        self.assertEqual(extract_symbol("result = calculator.add("), "calculator.add")

    def test_dotted_identifier_without_call(self):
        self.assertEqual(extract_symbol("x = user.first_name"), "user.first_name")

    def test_multi_dot_identifier(self):
        self.assertEqual(extract_symbol("self.repo.users.get("), "self.repo.users.get")

    def test_partial_identifier_mid_typing(self):
        self.assertEqual(extract_symbol("result = slug"), "slug")

    def test_no_trailing_identifier_returns_none(self):
        self.assertIsNone(extract_symbol("x = 1 + "))

    def test_empty_string_returns_none(self):
        self.assertIsNone(extract_symbol(""))

    def test_split_symbol_simple(self):
        self.assertEqual(split_symbol("add"), (None, "add"))

    def test_split_symbol_dotted(self):
        self.assertEqual(split_symbol("calculator.add"), ("calculator", "add"))

    def test_split_symbol_multi_dot_uses_last_component_as_name(self):
        self.assertEqual(split_symbol("self.repo.get"), ("self.repo", "get"))


class SymbolRetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chunks = RepoParser(str(SAMPLE_REPO)).parse_repo()
        cls.retriever = SymbolRetriever([c.to_dict() for c in chunks])

    def test_simple_symbol_exact_match(self):
        results = self.retriever.search("result = average(", top_k=5)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["name"], "average")
        self.assertEqual(results[0]["score"], SymbolRetriever.EXACT_NAME)

    def test_dotted_symbol_ranks_matching_class_method_first(self):
        # Both Calculator.add and UserRepository.add exist; "calculator.add"
        # must rank the class-qualified exact match above the bare name match.
        results = self.retriever.search("result = calculator.add(", top_k=5)
        self.assertEqual(results[0]["name"], "add")
        self.assertEqual(results[0]["class_name"], "Calculator")
        self.assertEqual(results[0]["score"], SymbolRetriever.EXACT_DOTTED)

        names_and_classes = [(r["name"], r["class_name"]) for r in results]
        self.assertIn(("add", "UserRepository"), names_and_classes)
        # the dotted exact match must outrank the same-name-wrong-class one
        rank_of_calculator_add = next(i for i, r in enumerate(results) if r["class_name"] == "Calculator")
        rank_of_user_add = next(i for i, r in enumerate(results) if r["class_name"] == "UserRepository")
        self.assertLess(rank_of_calculator_add, rank_of_user_add)

    def test_partial_typing_prefix_match(self):
        results = self.retriever.search("result = slug", top_k=5)
        self.assertEqual(results[0]["name"], "slugify")
        self.assertEqual(results[0]["score"], SymbolRetriever.PREFIX_NAME)

    def test_unmatched_symbol_returns_empty(self):
        results = self.retriever.search("result = totally_unknown_symbol(", top_k=5)
        self.assertEqual(results, [])

    def test_no_symbol_in_code_returns_empty(self):
        results = self.retriever.search("x = 1 + ", top_k=5)
        self.assertEqual(results, [])

    def test_top_k_is_respected(self):
        results = self.retriever.search("result = add(", top_k=1)
        self.assertLessEqual(len(results), 1)

    def test_result_shape(self):
        results = self.retriever.search("result = divide(", top_k=1)
        self.assertEqual(set(results[0].keys()), {"chunk_id", "file_path", "name", "class_name", "score"})


if __name__ == "__main__":
    print("=== Symbol search demo (examples/sample_repo) ===", flush=True)
    demo_chunks = RepoParser(str(SAMPLE_REPO)).parse_repo()
    demo_retriever = SymbolRetriever([c.to_dict() for c in demo_chunks])
    for code in [
        "result = calculator.add(",
        "result = divide(",
        "result = slug",
    ]:
        print(f"\ncode_before_cursor: {code!r}  (symbol={extract_symbol(code)!r})", flush=True)
        for r in demo_retriever.search(code, top_k=3):
            print(
                f"  score={r['score']:.1f}  name={r['name']:12s} class={r['class_name']!s:16s} "
                f"chunk_id={r['chunk_id']}",
                flush=True,
            )

    print("\n=== Running unit tests ===", flush=True)
    unittest.main()
