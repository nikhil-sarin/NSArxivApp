import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import citation_discovery, research_db


class CitationDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = mock.patch.object(research_db, "DB_PATH", Path(self.tempdir.name) / "research.db")
        self.db_patch.start()

    def tearDown(self):
        self.db_patch.stop()
        self.tempdir.cleanup()

    @mock.patch("app.citation_discovery.citation_opportunities.export_bundle")
    @mock.patch("app.citation_discovery.citation_opportunities.judge")
    @mock.patch("app.citation_discovery.citation_evidence.build_evidence_packet")
    @mock.patch("app.citation_discovery.contribution_catalogue.load")
    def test_discovery_judges_filtered_candidates_but_never_exports(
        self, load_catalogue, build_packet, judge, export_bundle
    ):
        load_catalogue.return_value = {
            "schema_version": "1.0",
            "owner": "Researcher",
            "owner_name_variants": ["Researcher"],
            "contributions": [{"id": "tool", "name": "Tool", "enabled": True}],
        }
        build_packet.return_value = {
            "paper_id": "2609.1", "contribution_id": "tool", "candidate": True,
            "passages": [{"locator": "Methods", "quote": "Direct evidence"}],
            "reference_check": {"canonical_citation_found": False},
        }
        judge.return_value = {"classification": "potentially_useful"}

        result = citation_discovery.discover_paper(
            {"arxiv_id": "2609.1", "title": "Paper"},
            "A sufficiently complete public paper body.",
        )

        self.assertEqual(result["reviewable"], 1)
        judge.assert_called_once()
        export_bundle.assert_not_called()

    @mock.patch("app.citation_discovery.citation_opportunities.judge")
    @mock.patch("app.citation_discovery.citation_evidence.build_evidence_packet")
    @mock.patch("app.citation_discovery.contribution_catalogue.load")
    def test_deterministic_rejection_avoids_model_call(self, load_catalogue, build_packet, judge):
        load_catalogue.return_value = {
            "schema_version": "1.0", "owner": "Researcher",
            "contributions": [{"id": "tool", "name": "Tool", "enabled": True}],
        }
        build_packet.return_value = {"candidate": False}

        result = citation_discovery.discover_paper(
            {"arxiv_id": "2609.2", "title": "Unrelated"}, "Public paper body"
        )

        self.assertEqual(result["model_judgements"], 0)
        judge.assert_not_called()

    @mock.patch("app.citation_discovery.citation_opportunity_store.delete_unreviewed_analyses")
    @mock.patch("app.citation_discovery.citation_opportunities.judge")
    @mock.patch("app.citation_discovery.citation_evidence.build_evidence_packet")
    @mock.patch("app.citation_discovery.contribution_catalogue.load")
    def test_force_removes_stale_result_when_pair_is_no_longer_a_candidate(
        self, load_catalogue, build_packet, judge, delete_unreviewed
    ):
        load_catalogue.return_value = {
            "schema_version": "1.2", "owner": "Researcher",
            "contributions": [{"id": "tool", "name": "Tool", "enabled": True}],
        }
        build_packet.return_value = {"candidate": False}

        citation_discovery.discover_paper(
            {"arxiv_id": "2609.2", "title": "Unrelated"}, "Public paper body", force=True
        )

        delete_unreviewed.assert_called_once_with("2609.2", "tool")
        judge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
