import unittest

from pm_synthesis.matching import match_panels


class MatchingTests(unittest.TestCase):
    def test_order_independent_match(self):
        target = [[0, 0, 0, 1, 1, 1], [2, 0, 0, 3, 1, 1]]
        generated = [target[1], target[0]]
        result = match_panels(target, generated)
        self.assertTrue(result["match"])

    def test_bipartite_augmenting_path_avoids_greedy_failure(self):
        target = [[0, 0, 0, 1, 1, 1], [0.03, 0, 0, 1.03, 1, 1]]
        generated = [[0.015, 0, 0, 1.015, 1, 1], [-0.019, 0, 0, 0.981, 1, 1]]
        result = match_panels(target, generated, tolerance=0.02)
        self.assertTrue(result["match"])

    def test_reports_count_mismatch(self):
        result = match_panels([[0, 0, 0, 1, 1, 1]], [], tolerance=0.01)
        self.assertFalse(result["match"])
        self.assertEqual(result["failure_type"], "topology-count")
        self.assertEqual(result["missing_count"], 1)

    def test_reports_signed_coordinate_delta(self):
        result = match_panels(
            [[0, 0, 0, 1, 1, 1]], [[0.1, 0, 0, 1.1, 1, 1]], tolerance=0.01
        )
        self.assertFalse(result["match"])
        self.assertAlmostEqual(result["nearest_pairs"][0]["delta"][0], 0.1)


if __name__ == "__main__":
    unittest.main()
