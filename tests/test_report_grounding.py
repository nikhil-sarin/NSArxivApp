import unittest

from app.report_generator import _is_structurally_valid, _quotes_are_grounded


class ReportGroundingTests(unittest.TestCase):
    def test_exact_normalized_quote_is_grounded(self):
        source = "We find that the ejecta opacity is strongly constrained by the observations."
        body = "<h2>Results</h2><q>We find that the ejecta opacity is strongly constrained by the observations.</q>"
        self.assertTrue(_quotes_are_grounded(body, source))
        self.assertTrue(_is_structurally_valid(body, source))

    def test_hallucinated_quote_is_rejected(self):
        source = "The data constrain the ejecta mass."
        body = "<h2>Results</h2><q>The observations prove an entirely different physical mechanism.</q>"
        self.assertFalse(_quotes_are_grounded(body, source))
        self.assertFalse(_is_structurally_valid(body, source))


if __name__ == "__main__":
    unittest.main()
