import unittest

from app.summarizer import PaperSummarizer
from app.summary_workflow import completeness_issues


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


if __name__ == "__main__":
    unittest.main()
