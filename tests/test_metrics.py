import unittest

from evaluation.metrics import edit_similarity, exact_match, extract_identifiers, identifier_f1


class ExactMatchTest(unittest.TestCase):
    def test_identical_strings_match(self):
        self.assertTrue(exact_match("return x + 1", "return x + 1"))

    def test_whitespace_is_stripped_before_comparing(self):
        self.assertTrue(exact_match("  return x + 1  ", "return x + 1"))

    def test_different_strings_do_not_match(self):
        self.assertFalse(exact_match("return x + 1", "return x + 2"))


class EditSimilarityTest(unittest.TestCase):
    def test_identical_strings_score_one(self):
        self.assertEqual(edit_similarity("return x", "return x"), 1.0)

    def test_both_empty_scores_one(self):
        self.assertEqual(edit_similarity("", ""), 1.0)

    def test_completely_different_scores_low(self):
        score = edit_similarity("abc", "xyz")
        self.assertLess(score, 0.5)

    def test_one_char_diff_scores_high(self):
        score = edit_similarity("return x", "return y")
        self.assertGreater(score, 0.8)

    def test_score_is_between_zero_and_one(self):
        score = edit_similarity("something completely different here", "x")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class IdentifierF1Test(unittest.TestCase):
    def test_extract_identifiers_ignores_punctuation_and_numbers(self):
        ids = extract_identifiers("result = foo(bar, 42) + baz_qux")
        self.assertEqual(ids, ["result", "foo", "bar", "baz_qux"])

    def test_identical_identifiers_score_one(self):
        self.assertEqual(identifier_f1("return foo(bar)", "return foo(bar)"), 1.0)

    def test_both_empty_of_identifiers_scores_one(self):
        self.assertEqual(identifier_f1("1 + 2", "3 * 4"), 1.0)

    def test_no_overlap_scores_zero(self):
        self.assertEqual(identifier_f1("foo(bar)", "baz(qux)"), 0.0)

    def test_partial_overlap_scores_between_zero_and_one(self):
        score = identifier_f1("foo(bar, extra)", "foo(bar)")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_one_side_empty_scores_zero(self):
        self.assertEqual(identifier_f1("foo(bar)", "1 + 2"), 0.0)


if __name__ == "__main__":
    unittest.main()
