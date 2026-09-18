import unittest
from contextlib import ExitStack
from datetime import date
from unittest import mock

from app import fetch_job


class FetchTrackingTests(unittest.TestCase):
    @mock.patch("app.fetch_job._load_fetch_watermark", return_value=date(2026, 9, 15))
    def test_pending_dates_recover_every_day_after_watermark(self, load_watermark):
        self.assertEqual(
            fetch_job._pending_announcement_dates(date(2026, 9, 18)),
            [date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18)],
        )

    def test_vector_failure_does_not_abort_daily_batch(self):
        paper = {
            "arxiv_id": "2609.12044",
            "title": "Relevant transient",
            "abstract": "Transient inference abstract.",
            "authors": ["A. Author"],
            "categories": ["astro-ph.HE"],
        }
        provenance = {"status": "complete", "provider": "openai"}
        vdb = mock.Mock()
        vdb.add_paper.side_effect = ValueError("nested metadata")
        with ExitStack() as stack:
            stack.enter_context(mock.patch("app.fetch_job.citation_discovery.enabled", return_value=False))
            stack.enter_context(mock.patch("app.fetch_job.get_paper_text", return_value="Full public paper text"))
            stack.enter_context(mock.patch("app.fetch_job.paper_store.paper_exists", return_value=False))
            save_paper = stack.enter_context(mock.patch("app.fetch_job.paper_store.save_paper"))
            stack.enter_context(mock.patch("app.fetch_job.paper_store.update_triage"))
            stack.enter_context(mock.patch("app.fetch_job.paper_store.load_triage", return_value=[]))
            stack.enter_context(mock.patch("app.fetch_job.researcher_profile.load", return_value={}))
            stack.enter_context(mock.patch("app.fetch_job.idea_store.load_ideas", return_value=[]))
            stack.enter_context(mock.patch(
                "app.fetch_job.contribution_catalogue.load",
                return_value={"contributions": []},
            ))
            stack.enter_context(mock.patch(
                "app.fetch_job.relevance.score_tracking_papers",
                return_value=[{**paper, "relevance_score": 80.0, "relevance_reason": "Relevant"}],
            ))
            stack.enter_context(mock.patch(
                "app.fetch_job.summarize_with_provenance",
                return_value=("Generated summary", provenance),
            ))

            count = fetch_job._save_new_papers(
                [paper],
                client=mock.Mock(),
                extractor=mock.Mock(),
                summarizer=mock.Mock(),
                vdb=vdb,
            )

        self.assertEqual(count, 1)
        save_paper.assert_called_once()

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
