import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from app import citation_backfill
from app.citation_backfill import eligible_papers


class CitationBackfillTests(unittest.TestCase):
    def test_filters_owner_old_and_duplicate_versions(self):
        now = datetime.now(timezone.utc)
        papers = [
            {"arxiv_id": "2609.1v1", "published": now.isoformat(), "authors": ["Other"]},
            {"arxiv_id": "2609.1v2", "published": now.isoformat(), "authors": ["Other"]},
            {"arxiv_id": "2609.2", "published": now.isoformat(), "authors": ["Nikhil Sarin"]},
            {"arxiv_id": "2401.1", "published": (now - timedelta(days=500)).isoformat(), "authors": ["Other"]},
        ]
        selected = eligible_papers(papers, owner="Nikhil Sarin", days=90, limit=10, now=now)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["arxiv_id"], "2609.1v1")

    @mock.patch("app.citation_backfill.citation_discovery.discover_paper")
    @mock.patch("app.citation_backfill.get_paper_text", return_value="Full public paper")
    @mock.patch("app.citation_backfill.paper_store.save_paper")
    @mock.patch("app.citation_backfill.paper_store.get_paper", return_value=None)
    @mock.patch("app.citation_backfill.PDFExtractor")
    @mock.patch("app.citation_backfill.ArxivClient")
    def test_explicit_ids_are_fetched_stored_and_checked(
        self, client_class, extractor_class, get_paper, save_paper, get_text, discover
    ):
        client = client_class.return_value
        result = mock.Mock()
        client.get_result_by_id.return_value = result
        client.get_paper_metadata.return_value = {
            "arxiv_id": "2609.08324", "title": "A paper", "abstract": "Abstract",
            "authors": ["Other"], "published": datetime.now(timezone.utc).isoformat(),
        }
        discover.return_value = {
            "checked": 22, "model_judgements": 1, "reviewable": 1,
        }

        totals = citation_backfill.run_ids(["https://arxiv.org/abs/2609.08324"])

        client.get_result_by_id.assert_called_once_with("2609.08324")
        save_paper.assert_called_once()
        discover.assert_called_once()
        self.assertEqual(totals["papers_processed"], 1)
        self.assertEqual(totals["reviewable"], 1)

    @mock.patch("app.citation_backfill.citation_discovery.discover_paper")
    @mock.patch("app.citation_backfill.get_paper_text", return_value="Full public paper")
    @mock.patch("app.citation_backfill.paper_store.save_paper")
    @mock.patch(
        "app.citation_backfill.paper_store.get_paper",
        return_value={"summary": "Existing generated summary"},
    )
    @mock.patch("app.citation_backfill.PDFExtractor")
    @mock.patch("app.citation_backfill.ArxivClient")
    def test_explicit_ids_refresh_metadata_without_replacing_summary(
        self, client_class, extractor_class, get_paper, save_paper, get_text, discover
    ):
        client = client_class.return_value
        client.get_result_by_id.return_value = mock.Mock()
        client.get_paper_metadata.return_value = {
            "arxiv_id": "2609.08324",
            "abstract": "Abstract",
            "comment": "Accepted for publication in ApJ",
        }
        discover.return_value = {"checked": 0, "model_judgements": 0, "reviewable": 0}

        citation_backfill.run_ids(["2609.08324"])

        save_paper.assert_called_once_with(
            "2609.08324",
            client.get_paper_metadata.return_value,
            "Existing generated summary",
        )


if __name__ == "__main__":
    unittest.main()
