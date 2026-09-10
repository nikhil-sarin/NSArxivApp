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

    def test_actionable_queue_is_not_crowded_out_by_newer_negative_results(self):
        base = {
            "analysis_version": "2", "catalogue_version": "1.0", "confidence": 0.5,
            "status": "proposed", "rationale": "Test", "counterargument": "Test",
            "evidence": [], "reference_check": {}, "model": {},
        }
        positive = {
            **base, "opportunity_id": "positive", "paper_id": "positive-paper",
            "contribution_id": "redback", "classification": "strong_citation_opportunity",
            "confidence": 0.9,
        }
        citation_opportunity_store.save(positive)
        for index in range(105):
            citation_opportunity_store.save({
                **base, "opportunity_id": f"negative-{index}", "paper_id": f"paper-{index}",
                "contribution_id": "redback", "classification": "insufficient_evidence",
            })

        self.assertNotIn("positive", {
            item["opportunity_id"] for item in citation_opportunity_store.list_opportunities()
        })
        actionable = citation_opportunity_store.list_actionable()
        self.assertEqual([item["opportunity_id"] for item in actionable], ["positive"])

    def test_complete_json_fence_is_accepted_but_surrounding_prose_is_not(self):
        payload = json.dumps({
            "classification": "potentially_useful", "confidence": 0.72,
            "rationale": "The method is directly relevant.",
            "counterargument": "The implementation may be independent.",
            "evidence_locators": ["Methods / page 3"],
        })
        accepted = citation_opportunities.judge(
            PACKET, CONTRIBUTION, catalogue_version="1.0",
            complete=lambda system, user: f"```json\n{payload}\n```",
            provider="openai", model_name="test",
        )
        self.assertEqual(accepted["classification"], "potentially_useful")

        rejected = citation_opportunities.judge(
            PACKET, CONTRIBUTION, catalogue_version="1.1",
            complete=lambda system, user: f"Here is the result:\n{payload}",
            provider="openai", model_name="test",
        )
        self.assertEqual(rejected["classification"], "insufficient_evidence")

    def test_judgement_receives_curated_contribution_claims(self):
        captured = {}
        response = json.dumps({
            "classification": "potentially_useful", "confidence": 0.7,
            "rationale": "The method is relevant.", "counterargument": "It may not be needed.",
            "evidence_locators": ["Methods / page 3"],
        })
        contribution = {**CONTRIBUTION, "key_claims": ["A specific methodological limitation."]}

        def complete(system, user):
            captured["payload"] = json.loads(user)
            return response

        citation_opportunities.judge(
            PACKET, contribution, catalogue_version="1.2", complete=complete,
            provider="openai", model_name="test",
        )
        self.assertEqual(
            captured["payload"]["contribution"]["key_claims"],
            ["A specific methodological limitation."],
        )

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
