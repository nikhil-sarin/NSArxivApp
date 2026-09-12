import unittest
from unittest import mock

from app import fetch_job


class FetchTrackingTests(unittest.TestCase):
    @mock.patch("app.fetch_job.citation_discovery.enabled", return_value=False)
    @mock.patch("app.fetch_job.summarize_with_provenance")
    @mock.patch("app.fetch_job.get_paper_text", return_value="Full public paper text")
    @mock.patch("app.fetch_job.paper_store.update_triage")
    @mock.patch("app.fetch_job.paper_store.save_paper")
    @mock.patch("app.fetch_job.paper_store.paper_exists", return_value=False)
    @mock.patch("app.fetch_job.paper_store.load_triage", return_value=[])
    @mock.patch("app.fetch_job.researcher_profile.load", return_value={})
    @mock.patch("app.fetch_job.idea_store.load_ideas", return_value=[])
    @mock.patch(
        "app.fetch_job.contribution_catalogue.load",
        return_value={"contributions": []},
    )
    @mock.patch("app.fetch_job.relevance.score_tracking_papers")
    def test_monitor_only_paper_is_hidden_but_retained_for_citation_checks(
        self,
        score_tracking,
        catalogue_load,
        load_ideas,
        profile_load,
        load_triage,
        paper_exists,
        save_paper,
        update_triage,
        get_text,
        summarize,
        discovery_enabled,
    ):
        paper = {
            "arxiv_id": "2609.10521",
            "title": "Point particles in general relativity",
            "abstract": "An unrelated mathematical-relativity paper.",
            "authors": ["A. Author"],
            "categories": ["gr-qc"],
        }
        score_tracking.return_value = [{
            **paper,
            "relevance_score": 20.0,
            "relevance_reason": "Low semantic relevance",
        }]
        vdb = mock.Mock()

        count = fetch_job._save_new_papers(
            [paper],
            client=mock.Mock(),
            extractor=mock.Mock(),
            summarizer=mock.Mock(),
            vdb=vdb,
        )

        self.assertEqual(count, 1)
        save_paper.assert_called_once()
        update_triage.assert_called_once_with(
            "2609.10521",
            status="irrelevant",
            relevance_score=20.0,
            relevance_reason="Low semantic relevance",
        )
        summarize.assert_not_called()
        vdb.add_paper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
