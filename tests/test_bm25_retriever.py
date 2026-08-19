import unittest
from pathlib import Path

from indexer.repo_parser import RepoParser
from retrieval.bm25_retriever import BM25Retriever

SAMPLE_REPO = Path(__file__).parent.parent / "examples" / "sample_repo"


class BM25RetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chunks = RepoParser(str(SAMPLE_REPO)).parse_repo()
        cls.retriever = BM25Retriever([c.to_dict() for c in chunks])

    def test_search_returns_ranked_results_with_expected_fields(self):
        results = self.retriever.search("divide two numbers", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)
        for r in results:
            self.assertEqual(set(r.keys()), {"chunk_id", "file_path", "name", "score"})
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_query_matching_docstring_ranks_target_chunk_first(self):
        results = self.retriever.search("divide a by b and record the operation", top_k=5)
        self.assertEqual(results[0]["name"], "divide")
        self.assertEqual(results[0]["file_path"], "calculator.py")

    def test_query_matching_function_name_ranks_it_first(self):
        results = self.retriever.search("slugify text into a slug", top_k=5)
        self.assertEqual(results[0]["name"], "slugify")

    def test_incomplete_code_snippet_as_query(self):
        # A realistic "incomplete code" completion-style query.
        query = "class UserRepository:\n    def __init__(self):\n        self._users = {}\n    def get(self"
        results = self.retriever.search(query, top_k=3)
        names = [r["name"] for r in results]
        self.assertIn("get", names)

    def test_top_k_is_respected(self):
        results = self.retriever.search("class", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_empty_index_returns_no_results(self):
        empty_retriever = BM25Retriever([])
        self.assertEqual(empty_retriever.search("anything"), [])


if __name__ == "__main__":
    print("=== BM25 search demo (examples/sample_repo) ===", flush=True)
    demo_chunks = RepoParser(str(SAMPLE_REPO)).parse_repo()
    demo_retriever = BM25Retriever([c.to_dict() for c in demo_chunks])
    for query in [
        "divide two numbers",
        "slugify text into a slug",
        "get user by id",
    ]:
        print(f"\nQuery: {query!r}", flush=True)
        for r in demo_retriever.search(query, top_k=3):
            print(
                f"  score={r['score']:.3f}  {r['file_path']:16s} {r['name']:12s} chunk_id={r['chunk_id']}",
                flush=True,
            )

    print("\n=== Running unit tests ===", flush=True)
    unittest.main()
