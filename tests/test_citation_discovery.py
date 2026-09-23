import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import citation_discovery, research_db


class CitationDiscoveryTests(unittest.TestCase):
    def test_personal_citation_discovery_is_opt_in(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(citation_discovery.enabled())
        with mock.patch.dict("os.environ", {"AUTO_CITATION_DISCOVERY": "true"}):
            self.assertTrue(citation_discovery.enabled())

    @mock.patch(
        "app.citation_discovery.citation_opportunity_store.paper_is_dismissed",
        return_value=True,
    )
    @mock.patch("app.citation_discovery.contribution_catalogue.load")
    def test_dismissed_paper_is_not_rediscovered(self, load_catalogue, is_dismissed):
        result = citation_discovery.discover_paper(
            {"arxiv_id": "2609.2", "title": "Relevant but declined"},
            "Public paper body",
            force=True,
        )

        self.assertEqual(result["ineligible_reason"], "dismissed_by_user")
        self.assertEqual(result["model_judgements"], 0)
        is_dismissed.assert_called_once_with("2609.2")
        load_catalogue.assert_not_called()

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = mock.patch.object(research_db, "DB_PATH", Path(self.tempdir.name) / "research.db")
        self.db_patch.start()

    def tearDown(self):
        self.db_patch.stop()
        self.tempdir.cleanup()

    @mock.patch("app.citation_discovery.citation_opportunities.judge")
    @mock.patch("app.citation_discovery.citation_evidence.build_evidence_packet")
    @mock.patch("app.citation_discovery.contribution_catalogue.load")
    def test_discovery_can_recheck_only_one_contribution(
        self, load_catalogue, build_packet, judge
    ):
        load_catalogue.return_value = {
            "schema_version": "1.3",
            "owner": "Researcher",
            "contributions": [
                {"id": "redback", "name": "Redback", "enabled": True},
                {"id": "other", "name": "Other", "enabled": True},
            ],
        }
        build_packet.return_value = {"candidate": False}

        result = citation_discovery.discover_paper(
            {"arxiv_id": "2609.2", "title": "Transient"},
            "Public transient paper body",
            force=True,
            contribution_ids={"redback"},
        )

        self.assertEqual(result["checked"], 1)
        self.assertEqual(build_packet.call_args.args[2]["id"], "redback")
        judge.assert_not_called()

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

        delete_unreviewed.assert_called_once_with("2609.2", ["tool"])
        judge.assert_not_called()

    @mock.patch("app.citation_discovery.citation_opportunities.judge")
    @mock.patch("app.citation_discovery.citation_evidence.build_evidence_packet")
    @mock.patch("app.citation_discovery.contribution_catalogue.load")
    @mock.patch("app.citation_discovery.citation_opportunity_store.delete_active_for_paper", return_value=2)
    def test_published_paper_is_cleaned_but_never_judged(
        self, delete_active, load_catalogue, build_packet, judge
    ):
        load_catalogue.return_value = {
            "schema_version": "1.2", "owner": "Researcher",
            "contributions": [{"id": "tool", "name": "Tool", "enabled": True}],
        }
        result = citation_discovery.discover_paper(
            {"arxiv_id": "2609.2", "comment": "Accepted for publication in ApJ"},
            "Public paper body",
            force=True,
        )
        self.assertEqual(result["ineligible_reason"], "accepted_or_published")
        self.assertEqual(result["removed"], 2)
        delete_active.assert_called_once_with("2609.2")
        build_packet.assert_not_called()
        judge.assert_not_called()

    @mock.patch("app.citation_discovery.paper_store.save_paper")
    @mock.patch("app.citation_discovery.paper_store.get_paper")
    @mock.patch("app.citation_discovery.citation_opportunity_store.delete_active_for_paper")
    @mock.patch("app.citation_discovery.citation_opportunity_store.list_actionable")
    def test_refresh_removes_papers_that_are_now_published(
        self, list_actionable, delete_active, get_paper, save_paper
    ):
        list_actionable.return_value = [
            {"paper_id": "2609.1", "status": "proposed"},
            {"paper_id": "2609.1", "status": "confirmed"},
            {"paper_id": "2609.2", "status": "exported"},
        ]
        delete_active.return_value = 2
        get_paper.return_value = {"arxiv_id": "2609.1", "summary": "Existing summary"}
        client = mock.Mock()
        client.get_paper_metadata.return_value = {
            "arxiv_id": "2609.1", "journal_ref": "ApJ 999, 1"
        }

        result = citation_discovery.refresh_active_eligibility(client)

        self.assertEqual(result, {
            "papers_checked": 1, "opportunities_removed": 2, "failed": 0,
        })
        client.get_result_by_id.assert_called_once_with("2609.1")
        delete_active.assert_called_once_with("2609.1")
        save_paper.assert_called_once()


if __name__ == "__main__":
    unittest.main()
