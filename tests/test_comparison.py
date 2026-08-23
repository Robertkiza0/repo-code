import unittest
from io import StringIO
from unittest.mock import patch

from evaluation.comparison import print_four_way_comparison, print_memory_ablation_comparison


class PrintFourWayComparisonTest(unittest.TestCase):
    def test_runs_and_shows_all_four_methods(self):
        no_selection = {"avg_candidate_count": 10.89, "exact_match_rate": 0.389, "avg_ES": 0.641, "avg_ID_F1": 0.655, "avg_generation_time": 37.01}
        llm_selection = {"avg_candidate_count": 10.89, "avg_selected_count": 1.89, "exact_match_rate": 0.444, "avg_ES": 0.611, "avg_ID_F1": 0.648, "avg_generation_time": 34.65}
        top2 = {"avg_candidate_count": 10.89, "avg_selected_count": 2.0, "exact_match_rate": 0.4, "avg_ES": 0.6, "avg_ID_F1": 0.62, "avg_generation_time": 30.0}
        random2 = {"avg_candidate_count": 10.89, "avg_selected_count": 2.0, "exact_match_rate": 0.3, "avg_ES": 0.5, "avg_ID_F1": 0.52, "avg_generation_time": 30.0}

        buf = StringIO()
        with patch("sys.stdout", buf):
            print_four_way_comparison(no_selection, llm_selection, top2, random2)
        output = buf.getvalue()

        self.assertIn("no selection", output)
        self.assertIn("LLM selection", output)
        self.assertIn("top2", output)
        self.assertIn("random2", output)
        self.assertIn("avg candidates", output)
        self.assertIn("exact match", output)

    def test_handles_missing_keys_as_na(self):
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_four_way_comparison({}, {}, {}, {})
        self.assertIn("n/a", buf.getvalue())


class PrintMemoryAblationComparisonTest(unittest.TestCase):
    def test_runs_and_shows_memory_specific_stats(self):
        no_selection = {"avg_candidate_count": 10.89, "exact_match_rate": 0.389, "avg_ES": 0.641, "avg_ID_F1": 0.655, "avg_generation_time": 37.01}
        llm_selection = {"avg_candidate_count": 10.89, "avg_selected_count": 1.28, "exact_match_rate": 0.444, "avg_ES": 0.611, "avg_ID_F1": 0.648, "avg_generation_time": 34.65}
        llm_memory = {
            "avg_candidate_count": 10.89, "avg_selected_count": 2.1, "exact_match_rate": 0.5,
            "avg_ES": 0.65, "avg_ID_F1": 0.66, "avg_generation_time": 36.0,
            "memory_assisted_selections": 15, "avg_memory_relationships_found": 8.3,
            "selection_count_distribution": {"0": 2, "1": 5, "2": 8, ">=3": 5},
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_memory_ablation_comparison(no_selection, llm_selection, llm_memory)
        output = buf.getvalue()

        self.assertIn("LLM + memory", output)
        self.assertIn("memory-assisted selections:", output)
        self.assertIn("15", output)
        self.assertIn("avg memory relationships found:", output)
        self.assertIn("selection count distribution:", output)

    def test_handles_missing_keys_as_na(self):
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_memory_ablation_comparison({}, {}, {})
        self.assertIn("n/a", buf.getvalue())

    def test_includes_fourth_column_when_random_memory_summary_given(self):
        no_selection = {"avg_candidate_count": 10.89, "exact_match_rate": 0.389, "avg_ES": 0.641, "avg_ID_F1": 0.655, "avg_generation_time": 37.01}
        llm_selection = {"avg_candidate_count": 10.89, "avg_selected_count": 1.28, "exact_match_rate": 0.444, "avg_ES": 0.611, "avg_ID_F1": 0.648, "avg_generation_time": 34.65}
        llm_memory = {
            "avg_candidate_count": 10.89, "avg_selected_count": 2.1, "exact_match_rate": 0.5,
            "avg_ES": 0.65, "avg_ID_F1": 0.66, "avg_generation_time": 36.0,
            "memory_assisted_selections": 15, "avg_memory_relationships_found": 8.3,
            "selection_count_distribution": {"0": 2, "1": 5, "2": 8, ">=3": 5},
        }
        random_memory = {
            "avg_candidate_count": 10.89, "avg_selected_count": 1.9, "exact_match_rate": 0.35,
            "avg_ES": 0.58, "avg_ID_F1": 0.6, "avg_generation_time": 35.0,
            "memory_assisted_selections": 10, "avg_memory_relationships_found": 8.0,
            "selection_count_distribution": {"0": 3, "1": 6, "2": 6, ">=3": 5},
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_memory_ablation_comparison(no_selection, llm_selection, llm_memory, random_memory)
        output = buf.getvalue()

        self.assertIn("LLM + random memory", output)
        self.assertIn("LLM + random memory)", output)


if __name__ == "__main__":
    unittest.main()
