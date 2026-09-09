import unittest
from datetime import datetime, timedelta, timezone

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


if __name__ == "__main__":
    unittest.main()
