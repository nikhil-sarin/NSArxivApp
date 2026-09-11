import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import paper_store, research_db


class ResearchStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.db_path = root / "research.db"
        self.legacy_path = root / "papers.json"
        self.patches = [
            mock.patch.object(research_db, "DB_PATH", self.db_path),
            mock.patch.object(paper_store, "STORE_PATH", self.db_path),
            mock.patch.object(paper_store, "LEGACY_STORE_PATH", self.legacy_path),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tempdir.cleanup()

    def test_legacy_json_is_imported_without_modifying_source(self):
        legacy = {
            "2601.00001": {
                "arxiv_id": "2601.00001",
                "title": "A useful transient paper",
                "summary": "A summary about transient inference.",
            }
        }
        original = json.dumps(legacy, indent=2)
        self.legacy_path.write_text(original, encoding="utf-8")

        self.assertEqual(paper_store.load_all_papers(), list(legacy.values()))
        self.assertEqual(self.legacy_path.read_text(encoding="utf-8"), original)
        self.assertEqual(paper_store.load_triage(status="saved")[0]["arxiv_id"], "2601.00001")

    def test_new_paper_enters_inbox_and_updates_preserve_notes(self):
        paper_store.save_paper(
            "2601.00002",
            {"arxiv_id": "2601.00002", "title": "Kilonova constraints"},
            "Initial summary",
        )
        inbox = paper_store.load_triage()
        self.assertEqual([paper["arxiv_id"] for paper in inbox], ["2601.00002"])

        paper_store.save_notes(
            "2601.00002",
            {"cite_for": "Opacity constraints", "key_result": "Strong bias"},
        )
        paper_store.save_paper(
            "2601.00002",
            {"arxiv_id": "2601.00002", "title": "Kilonova constraints"},
            "Updated summary",
        )
        stored = paper_store.get_paper("2601.00002")
        self.assertEqual(stored["summary"], "Updated summary")
        self.assertEqual(stored["research_notes"]["cite_for"], "Opacity constraints")

    def test_reading_library_excludes_monitoring_only_papers(self):
        paper_store.save_paper(
            "2601.00010",
            {"arxiv_id": "2601.00010", "title": "Relevant transient"},
            "Summary",
        )
        paper_store.save_paper(
            "2601.00011",
            {"arxiv_id": "2601.00011", "title": "Unrelated theory"},
            "Summary",
        )
        paper_store.update_triage("2601.00011", status="irrelevant")

        self.assertEqual(
            [paper["arxiv_id"] for paper in paper_store.load_reading_papers()],
            ["2601.00010"],
        )
        self.assertEqual(len(paper_store.load_all_papers()), 2)

    def test_notes_are_full_text_searchable(self):
        paper_store.save_paper(
            "2601.00003",
            {"arxiv_id": "2601.00003", "title": "Jet modelling"},
            "A generic summary",
        )
        paper_store.save_notes("2601.00003", {"caveats": "Hydrodynamic misspecification bias"})

        matches = paper_store.search_by_text("misspecification")
        self.assertEqual([paper["arxiv_id"] for paper in matches], ["2601.00003"])

    def test_transaction_rolls_back_on_error(self):
        with self.assertRaises(RuntimeError):
            with research_db.transaction() as db:
                db.execute(
                    "INSERT INTO jobs(job_id, kind, status, created_at, updated_at) VALUES(?, ?, ?, ?, ?)",
                    ("job-1", "test", "running", "now", "now"),
                )
                raise RuntimeError("stop")

        db = research_db.connect()
        try:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
