import tempfile
import json
import unittest
from pathlib import Path

from app import contribution_catalogue


class ContributionCatalogueTests(unittest.TestCase):
    def test_example_catalogue_is_valid_and_aliases_normalize(self):
        catalogue = contribution_catalogue.load(contribution_catalogue.EXAMPLE_PATH)
        redback = contribution_catalogue.get(catalogue, "redback")
        self.assertIn("redback software", redback["aliases"])
        self.assertEqual(redback["canonical_citations"][0]["arxiv_id"], "2308.12806")

    def test_duplicate_and_missing_identifier_errors_include_id(self):
        entry = {
            "id": "duplicate", "name": "Name", "kind": "method", "aliases": [],
            "strong_signals": ["direct method"], "weak_signals": [], "exclusions": [],
            "contact_framing": "Offer help", "canonical_citations": [],
        }
        with self.assertRaisesRegex(contribution_catalogue.CatalogueError, "duplicate"):
            contribution_catalogue.validate({"schema_version": "1.0", "owner": "Owner", "contributions": [entry]})
        valid = {**entry, "unpublished": True}
        with self.assertRaisesRegex(contribution_catalogue.CatalogueError, "duplicate id"):
            contribution_catalogue.validate({
                "schema_version": "1.0", "owner": "Owner", "contributions": [valid, valid],
            })

    def test_fixed_evaluation_manifest_has_required_20_case_mix(self):
        path = Path("tests/fixtures/citation_opportunities/evaluation_cases.json")
        cases = json.loads(path.read_text(encoding="utf-8"))
        counts = {label: sum(case["label"] == label for case in cases) for label in {
            "strong_citation_opportunity", "potentially_useful", "not_relevant", "insufficient_evidence"
        }}
        self.assertEqual(len(cases), 20)
        self.assertEqual(counts["strong_citation_opportunity"], 5)
        self.assertEqual(counts["potentially_useful"], 5)
        self.assertEqual(counts["insufficient_evidence"], 2)


if __name__ == "__main__":
    unittest.main()
