import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import citation_opportunities, citation_opportunity_store, research_db


PACKET = {
    "paper_id": "2609.1", "contribution_id": "redback", "candidate": True,
    "passages": [{"locator": "Methods / page 3", "quote": "We use the method for inference.", "start_offset": 10, "end_offset": 42}],
    "reference_check": {"canonical_citation_found": False, "owner_name_found": False, "contribution_alias_found": False, "matched_identifiers": []},
}
CONTRIBUTION = {
    "id": "redback", "name": "Redback", "kind": "software", "contact_framing": "Offer help",
    "canonical_citations": [{"preferred_text": "Sarin et al. 2024", "url": "https://doi.org/example"}],
    "public_url": "https://example.org/redback",
}


class CitationOpportunityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(research_db, "DB_PATH", Path(self.tempdir.name) / "research.db")
        self.patch.start()

    def tearDown(self):
        self.patch.stop(); self.tempdir.cleanup()

    def test_valid_judgement_is_stable_and_persisted(self):
        response = json.dumps({
            "classification": "strong_citation_opportunity", "confidence": 0.86,
            "rationale": "The software directly supports the described inference.",
            "counterargument": "A bespoke implementation may be reasonable.",
            "evidence_locators": ["Methods / page 3"],
        })
        first = citation_opportunities.judge(
            PACKET, CONTRIBUTION, catalogue_version="1.0", complete=lambda s, u: response,
            provider="ollama", model_name="test",
        )
        second = citation_opportunities.judge(
            PACKET, CONTRIBUTION, catalogue_version="1.0", complete=lambda s, u: response,
            provider="ollama", model_name="test",
        )
        self.assertEqual(first["opportunity_id"], second["opportunity_id"])
        self.assertEqual(len(citation_opportunity_store.list_opportunities()), 1)

    def test_malformed_or_unsupported_model_output_is_insufficient(self):
        malformed = citation_opportunities.judge(
            PACKET, CONTRIBUTION, catalogue_version="1.0", complete=lambda s, u: "not json",
            provider="ollama", model_name="test",
        )
        self.assertEqual(malformed["classification"], "insufficient_evidence")
        unsupported = json.dumps({
            "classification": "strong_citation_opportunity", "confidence": 0.9,
            "rationale": "Direct", "counterargument": "Could be bespoke", "evidence_locators": ["Invented page"],
        })
        result = citation_opportunities.judge(
            PACKET, CONTRIBUTION, catalogue_version="1.1", complete=lambda s, u: unsupported,
            provider="ollama", model_name="test",
        )
        self.assertEqual(result["classification"], "insufficient_evidence")

    def test_export_requires_confirmation_and_is_bounded(self):
        opportunity = {
            "opportunity_id": "cop_1", "paper_id": "2609.1", "analysis_version": "1",
            "catalogue_version": "1.0",
            "classification": "potentially_useful", "confidence": 0.7, "rationale": "Relevant",
            "counterargument": "May not be needed", "evidence": PACKET["passages"], "status": "proposed",
        }
        paper = {"title": "A paper", "authors": ["A. Author"]}
        with self.assertRaises(ValueError):
            citation_opportunities.build_import_bundle(opportunity, paper, CONTRIBUTION)
        bundle = citation_opportunities.build_import_bundle({**opportunity, "status": "confirmed"}, paper, CONTRIBUTION)
        self.assertEqual(bundle["candidates"][0]["external_id"], "cop_1:draft-email")
        context = json.loads(bundle["candidates"][0]["context"])
        self.assertNotIn("full_text", context)
        self.assertEqual(context["paper"]["arxiv_id"], "2609.1")
        self.assertEqual(context["contribution"]["url"], "https://example.org/redback")
        self.assertEqual(context["evidence"][0]["quote"], PACKET["passages"][0]["quote"])


if __name__ == "__main__":
    unittest.main()
