import unittest

from app.summarizer import PaperSummarizer
from app.summary_workflow import completeness_issues, summarize_with_provenance


class SummaryCompletenessTests(unittest.TestCase):
    def test_truncated_and_short_summaries_are_flagged(self):
        self.assertIn("too short", completeness_issues("A short result."))
        self.assertIn("appears truncated", completeness_issues("word " * 80))

    def test_section_aware_excerpt_keeps_middle_methods_and_results(self):
        summarizer = PaperSummarizer.__new__(PaperSummarizer)
        summarizer.CONTEXT_LIMIT = 30000
        paragraphs = [f"Introduction paragraph {index}. " + "context " * 80 for index in range(12)]
        paragraphs += ["Methods\nBayesian hierarchical inference with calibrated likelihoods. " * 20]
        paragraphs += [f"Body paragraph {index}. " + "details " * 80 for index in range(25)]
        paragraphs += ["Results\nThe posterior constrains ejecta opacity and mass. " * 20]
        paragraphs += [f"Appendix paragraph {index}. " + "appendix " * 80 for index in range(8)]
        excerpt = summarizer._build_quick_summary_text("\n\n".join(paragraphs))
        self.assertIn("Bayesian hierarchical inference", excerpt)
        self.assertIn("posterior constrains ejecta opacity", excerpt)
        self.assertLessEqual(len(excerpt), summarizer.CONTEXT_LIMIT)

    def test_summary_provenance_exposes_model_fallback(self):
        class FakeSummarizer:
            last_fallback_reason = "Timeout: provider unavailable"

            def summarize(self, text, max_length=300, detailed=False):
                return "A complete sentence generated from deterministic source text. " * 10

            def _active_provider(self):
                return "openai"

            def _active_model(self):
                return "test-model"

        summary, provenance = summarize_with_provenance(
            FakeSummarizer(),
            "Full paper text " * 20,
        )
        self.assertTrue(summary)
        self.assertEqual(provenance["status"], "fallback")
        self.assertEqual(provenance["model"], "test-model")
        self.assertEqual(provenance["input_source"], "full_text")


if __name__ == "__main__":
    unittest.main()
