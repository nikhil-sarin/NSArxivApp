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

    def test_manual_opportunity_requires_grounded_evidence(self):
        result = citation_opportunities.create_manual(
            paper_id="2609.09520",
            contribution=CONTRIBUTION,
            catalogue_version="1.2",
            classification="potentially_useful",
            confidence=0.9,
            rationale="The mass-loss treatment is directly relevant.",
            counterargument="The existing approximation may be sufficient.",
            evidence_quote="We assume a steady wind density profile.",
            evidence_locator="Methods, page 4",
            paper_text="Methods\nWe assume a steady wind density profile.\nResults",
            reference_check={"canonical_citation_found": False},
        )

        self.assertTrue(result["opportunity_id"].startswith("cop_manual_"))
        self.assertEqual(result["model"]["provider"], "manual")
        self.assertEqual(
            citation_opportunity_store.list_actionable()[0]["opportunity_id"],
            result["opportunity_id"],
        )

        with self.assertRaisesRegex(ValueError, "not found"):
            citation_opportunities.create_manual(
                paper_id="2609.09520",
                contribution=CONTRIBUTION,
                catalogue_version="1.2",
                classification="potentially_useful",
                confidence=0.9,
                rationale="Relevant connection.",
                counterargument="May not be needed.",
                evidence_quote="This sentence is fabricated.",
                evidence_locator="Page 4",
                paper_text="The real paper text.",
                reference_check={"canonical_citation_found": False},
            )

    def test_manual_strong_opportunity_rejects_existing_citation(self):
        with self.assertRaisesRegex(ValueError, "canonical citation"):
            citation_opportunities.create_manual(
                paper_id="2609.09520",
                contribution=CONTRIBUTION,
                catalogue_version="1.2",
                classification="strong_citation_opportunity",
                confidence=0.9,
                rationale="Direct software connection.",
                counterargument="The citation may be incidental.",
                evidence_quote="We use Redback for transient inference.",
                evidence_locator="Methods",
                paper_text="We use Redback for transient inference.",
                reference_check={"canonical_citation_found": True},
            )

    def test_manual_opportunity_rejects_overlong_evidence(self):
        quote = "x" * (citation_opportunities.MAX_MANUAL_EVIDENCE_CHARS + 1)
        with self.assertRaisesRegex(ValueError, "at most"):
            citation_opportunities.create_manual(
                paper_id="2609.09520",
                contribution=CONTRIBUTION,
                catalogue_version="1.2",
                classification="potentially_useful",
                confidence=0.9,
                rationale="Relevant connection.",
                counterargument="May not be needed.",
                evidence_quote=quote,
                evidence_locator="Methods",
                paper_text=quote,
                reference_check={"canonical_citation_found": False},
            )

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

    def test_actionable_queue_can_filter_by_versionless_paper_id(self):
        base = {
            "analysis_version": "3", "catalogue_version": "1.0", "confidence": 0.8,
            "status": "proposed", "rationale": "Test", "counterargument": "Test",
            "evidence": [], "reference_check": {}, "model": {},
            "contribution_id": "redback", "classification": "potentially_useful",
        }
        citation_opportunity_store.save({
            **base, "opportunity_id": "wanted", "paper_id": "2609.09520v1",
        })
        citation_opportunity_store.save({
            **base, "opportunity_id": "other", "paper_id": "2609.08356",
        })

        actionable = citation_opportunity_store.list_actionable(paper_id="2609.09520")

        self.assertEqual([item["opportunity_id"] for item in actionable], ["wanted"])

    def test_arxiv_id_lookup_accepts_urls_labels_and_versions(self):
        self.assertEqual(
            citation_opportunities.arxiv_id_from_input(
                "https://arxiv.org/abs/2609.09520v2"
            ),
            "2609.09520",
        )
        self.assertEqual(
            citation_opportunities.arxiv_id_from_input("arXiv:2609.08356"),
            "2609.08356",
        )
        self.assertIsNone(citation_opportunities.arxiv_id_from_input("mass loss history"))

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
            "contribution_id": "redback",
            "catalogue_version": "1.0",
            "classification": "potentially_useful", "confidence": 0.7, "rationale": "Relevant",
            "counterargument": "May not be needed", "evidence": PACKET["passages"], "status": "proposed",
        }
        paper = {
            "title": "A paper", "authors": ["A. Author"],
            "corresponding_author": {"name": "A. Author", "email": "author@example.edu"},
        }
        with self.assertRaises(ValueError):
            citation_opportunities.build_import_bundle(
                [opportunity], paper, {"redback": CONTRIBUTION}
            )
        bundle = citation_opportunities.build_import_bundle(
            [{**opportunity, "status": "confirmed"}],
            paper,
            {"redback": CONTRIBUTION},
        )
        self.assertTrue(
            bundle["candidates"][0]["external_id"].startswith("cop_group_")
        )
        context = json.loads(bundle["candidates"][0]["context"])
        self.assertNotIn("full_text", context)
        self.assertEqual(context["paper"]["arxiv_id"], "2609.1")
        self.assertEqual(context["paper"]["corresponding_author"]["email"], "author@example.edu")
        self.assertEqual(context["contributions"][0]["url"], "https://example.org/redback")
        self.assertEqual(context["evidence"][0]["quote"], PACKET["passages"][0]["quote"])
        self.assertEqual(
            context["citation_request"],
            "I would kindly ask you to consider citing this work.",
        )

    def test_multiple_opportunities_share_one_email_candidate(self):
        second_contribution = {
            **CONTRIBUTION,
            "id": "mass-loss",
            "name": "Mass-loss histories",
            "public_url": "https://example.org/mass-loss",
        }
        opportunities = [
            {
                "opportunity_id": "cop_1",
                "paper_id": "2609.1",
                "contribution_id": "redback",
                "catalogue_version": "1.2",
                "classification": "strong_citation_opportunity",
                "confidence": 0.9,
                "rationale": "Bilby is used for transient inference.",
                "counterargument": "A bespoke workflow may be sufficient.",
                "evidence": PACKET["passages"],
                "status": "confirmed",
            },
            {
                "opportunity_id": "cop_2",
                "paper_id": "2609.1",
                "contribution_id": "mass-loss",
                "catalogue_version": "1.2",
                "classification": "potentially_useful",
                "confidence": 0.7,
                "rationale": "The wind mapping affects the inferred history.",
                "counterargument": "The approximation may be adequate.",
                "evidence": [{"locator": "Discussion", "quote": "A steady wind is assumed."}],
                "status": "confirmed",
            },
        ]

        bundle = citation_opportunities.build_import_bundle(
            opportunities,
            {"title": "A paper", "authors": ["A. Author"]},
            {"redback": CONTRIBUTION, "mass-loss": second_contribution},
        )

        self.assertEqual(len(bundle["candidates"]), 1)
        context = json.loads(bundle["candidates"][0]["context"])
        self.assertEqual(
            [item["id"] for item in context["contributions"]],
            ["redback", "mass-loss"],
        )
        self.assertEqual(len(context["evidence"]), 2)
        self.assertEqual(
            context["citation_request"],
            "I would kindly ask you to consider citing these works.",
        )

    def test_grouping_preserves_paper_and_queue_order(self):
        grouped = citation_opportunities.group_by_paper([
            {"paper_id": "paper-a", "contribution_id": "one", "opportunity_id": "a1"},
            {"paper_id": "paper-b", "contribution_id": "one", "opportunity_id": "b1"},
            {"paper_id": "paper-a", "contribution_id": "two", "opportunity_id": "a2"},
            {"paper_id": "paper-a", "contribution_id": "one", "opportunity_id": "stale"},
        ])

        self.assertEqual(
            [[item["opportunity_id"] for item in group] for group in grouped],
            [["a1", "a2"], ["b1"]],
        )

    @mock.patch("app.citation_opportunities.requests.post")
    def test_export_uses_configured_orchestrator_bearer_token(self, post):
        post.return_value.json.return_value = {"candidate_ids": ["candidate_1"]}
        opportunity = {
            "opportunity_id": "cop_1", "paper_id": "2609.1", "analysis_version": "2",
            "contribution_id": "redback",
            "catalogue_version": "1.2", "classification": "strong_citation_opportunity",
            "confidence": 0.9, "rationale": "Relevant", "counterargument": "May not apply",
            "evidence": PACKET["passages"], "status": "confirmed",
        }
        with mock.patch.dict(
            "os.environ", {"LOCAL_ORCHESTRATOR_API_TOKEN": "machine-secret"}
        ):
            citation_opportunities.export_bundle(
                [opportunity],
                {"title": "A paper", "authors": ["A. Author"]},
                {"redback": CONTRIBUTION},
                url="http://127.0.0.1:8775/v1/import-bundles",
            )

        self.assertEqual(
            post.call_args.kwargs["headers"],
            {"Authorization": "Bearer machine-secret"},
        )


if __name__ == "__main__":
    unittest.main()
