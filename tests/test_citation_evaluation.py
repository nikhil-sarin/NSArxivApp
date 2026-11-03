import unittest

from app.citation_evaluation import evaluate


class CitationEvaluationTests(unittest.TestCase):
    def test_precision_first_launch_metrics(self):
        cases = [
            {"id": "s", "label": "strong_citation_opportunity"},
            {"id": "n", "label": "not_relevant"},
            {"id": "c", "label": "not_relevant", "canonical_citation_present": True},
        ]
        predictions = [
            {"id": "s", "classification": "strong_citation_opportunity", "evidence": [{"quote_valid": True}], "latency_seconds": 1},
            {"id": "n", "classification": "not_relevant", "evidence": [], "latency_seconds": 2},
            {"id": "c", "classification": "not_relevant", "reference_check_correct": True, "evidence": [], "latency_seconds": 3},
        ]
        metrics = evaluate(cases, predictions)
        self.assertEqual(metrics["strong_precision"], 1.0)
        self.assertEqual(metrics["reference_check_accuracy"], 1.0)
        self.assertTrue(metrics["launch_ready"])


if __name__ == "__main__":
    unittest.main()
