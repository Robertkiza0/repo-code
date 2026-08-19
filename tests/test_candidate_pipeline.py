import json
import unittest
from pathlib import Path

from indexer.repo_parser import RepoParser
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dependency_retriever import DependencyRetriever
from retrieval.symbol_retriever import SymbolRetriever
from retrieval.candidate_pipeline import CandidatePipeline, DEFAULT_MAX_CANDIDATES

SAMPLE_REPO = Path(__file__).parent / "sample_repo"  # module_a.py + pkg/module_b.py (real cross-file import)


class CandidatePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = [c.to_dict() for c in RepoParser(str(SAMPLE_REPO)).parse_repo()]
        cls.pipeline = CandidatePipeline(
            BM25Retriever(cls.chunks),
            SymbolRetriever(cls.chunks),
            DependencyRetriever(cls.chunks),
        )

    def test_chunk_nominated_by_all_three_sources_is_merged_not_duplicated(self):
        # module_b.py imports Greeter from module_a.py: "Greeter(" as a query
        # should hit it via bm25 (text match), symbol (exact name match), and
        # dependency (module_b.py imports module_a.py).
        results = self.pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py")
        greeter_ids = [r["chunk_id"] for r in results if r["name"] == "Greeter"]
        self.assertEqual(len(greeter_ids), 1, "Greeter must appear exactly once, not once per source")

        greeter = next(r for r in results if r["name"] == "Greeter")
        self.assertEqual(set(greeter["sources"]), {"bm25", "symbol", "dependency"})
        self.assertEqual(set(greeter["scores"].keys()), {"bm25", "symbol", "dependency"})

    def test_original_chunk_ids_are_preserved(self):
        results = self.pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py")
        real_ids = {c["chunk_id"] for c in self.chunks}
        for r in results:
            self.assertIn(r["chunk_id"], real_ids)

    def test_pool_is_capped_at_max_candidates(self):
        small_pipeline = CandidatePipeline(
            BM25Retriever(self.chunks),
            SymbolRetriever(self.chunks),
            DependencyRetriever(self.chunks),
            max_candidates=2,
        )
        results = small_pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py", per_source_top_k=10)
        self.assertLessEqual(len(results), 2)

    def test_default_max_candidates_is_twelve(self):
        self.assertEqual(DEFAULT_MAX_CANDIDATES, 12)
        results = self.pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py", per_source_top_k=10)
        self.assertLessEqual(len(results), 12)

    def test_multi_source_chunks_rank_above_single_source_chunks(self):
        results = self.pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py", per_source_top_k=10)
        source_counts = [len(r["sources"]) for r in results]
        self.assertEqual(source_counts, sorted(source_counts, reverse=True))

    def test_dependency_only_chunk_is_recorded_with_dependency_source(self):
        # _format is nested inside Greeter.greet in module_a.py; it won't match
        # the "Greeter(" query via bm25/symbol text, only via dependency (it's
        # defined in module_a.py, which module_b.py imports).
        results = self.pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py", per_source_top_k=10)
        format_chunk = next(r for r in results if r["name"] == "_format")
        self.assertIn("dependency", format_chunk["sources"])

    def test_results_are_json_serializable(self):
        results = self.pipeline.nominate("result = Greeter(", target_file="pkg/module_b.py")
        json.dumps(results)  # must not raise (e.g. numpy float64 leaking through)

    def test_no_target_file_dependency_still_returns_bm25_and_symbol_hits(self):
        results = self.pipeline.nominate("result = Greeter(", target_file="module_a.py")
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertNotIn("dependency", r["sources"])


if __name__ == "__main__":
    unittest.main()
