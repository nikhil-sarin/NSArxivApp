import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from app import evaluation, index_jobs, job_queue, maintenance, paper_store, privacy, research_db, summary_jobs, synthesis, trends
from app.corpus_index import index_text_record


class FakeVectorDB:
    def __init__(self):
        self.documents = {}
        self.collection = mock.Mock()

    def upsert_document(self, document_id, text, metadata):
        self.documents[document_id] = {"text": text, "metadata": metadata}

    def delete_documents(self, owner_type, owner_id):
        prefix = f"{owner_type}:{owner_id}:"
        self.documents = {key: value for key, value in self.documents.items() if not key.startswith(prefix)}

    def search_documents(self, query, top_k=10, paper_ids=None):
        return []

    def add_paper(self, paper_id, title, summary, metadata):
        self.documents[paper_id] = {"text": summary, "metadata": metadata}


class ResearchOpsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "research.db"
        self.patches = [
            mock.patch.object(research_db, "DB_PATH", self.db_path),
            mock.patch.object(paper_store, "LEGACY_STORE_PATH", Path(self.tempdir.name) / "missing.json"),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tempdir.cleanup()

    def test_privacy_policy_is_persisted(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            privacy.set_policy("local_only")
            self.assertEqual(privacy.active_policy(), "local_only")
            self.assertEqual(
                privacy.choose_provider("gemini", contains_private_data=False),
                "ollama",
            )

    def test_generic_record_is_attributable_and_searchable(self):
        vector = FakeVectorDB()
        count = index_text_record(
            owner_type="meeting",
            owner_id="weekly-1",
            kind="meeting",
            title="Weekly meeting",
            text="We decided to compare kilonova opacity inference across simulations.",
            locator={"project_id": "p1"},
            vector_db=vector,
        )
        self.assertEqual(count, 1)
        result = research_db.search_documents("kilonova opacity", kinds=["meeting"])[0]
        self.assertEqual(result["locator"]["project_id"], "p1")
        self.assertIn("meeting:weekly-1:1", vector.documents)

    def test_evaluation_persists_metrics_and_routes_locally(self):
        result = evaluation.run_suite(
            suite="test",
            provider="ollama",
            model="test-model",
            complete=lambda question: "Contribution, evidence, and limitation are addressed.",
        )
        self.assertEqual(result["cases"], 3)
        runs = evaluation.list_runs()
        self.assertEqual(runs[0]["model"], "test-model")
        self.assertEqual(evaluation.recommend_route(runs, contains_private_data=True)["provider"], "ollama")

    def test_bibtex_and_emerging_theme(self):
        now = datetime.now(timezone.utc)
        papers = [
            {
                "arxiv_id": "2609.00001",
                "title": "Kilonova Opacity Inference",
                "authors": ["Nikhil Sarin", "A. Person"],
                "published": (now - timedelta(days=10)).isoformat(),
                "abstract": "Neural surrogate opacity inference for kilonova light curves.",
            },
            {
                "arxiv_id": "2609.00002",
                "title": "Fast Opacity Constraints",
                "authors": ["B. Person"],
                "published": (now - timedelta(days=20)).isoformat(),
                "abstract": "Neural surrogate opacity constraints for kilonova models.",
            },
            {
                "arxiv_id": "2609.00003",
                "title": "Opacity Effects in Kilonovae",
                "authors": ["D. Person"],
                "published": (now - timedelta(days=30)).isoformat(),
                "abstract": "Opacity changes the inferred kilonova ejecta composition.",
            },
            {
                "arxiv_id": "2601.00001",
                "title": "Older Transient Models",
                "authors": ["C. Person"],
                "published": (now - timedelta(days=150)).isoformat(),
                "abstract": "Analytic transient models and ejecta dynamics.",
            },
        ]
        exported = synthesis.bibtex(papers[:1])
        self.assertIn("@misc{Sarin", exported)
        self.assertIn("eprint = {2609.00001}", exported)
        theme_names = {row["theme"] for row in trends.emerging_themes(papers, now=now)}
        self.assertIn("opacity", theme_names)

    def test_background_index_job_records_completion(self):
        paper_store.save_paper(
            "2609.1",
            {"arxiv_id": "2609.1", "title": "Test", "abstract": "Useful abstract"},
            "Useful summary",
        )
        with mock.patch("app.index_jobs.corpus_index.index_paper", return_value=2):
            job_id = index_jobs.enqueue_library_index(
                arxiv_client=object(), pdf_extractor=object(), vector_db=FakeVectorDB()
            )
            deadline = time.time() + 3
            while time.time() < deadline:
                job = next(item for item in index_jobs.list_jobs() if item["job_id"] == job_id)
                if job["status"] not in {"queued", "running"}:
                    break
                time.sleep(0.02)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"]["chunks_written"], 2)

    def test_interrupted_jobs_are_requeued(self):
        with research_db.transaction() as db:
            db.execute(
                "INSERT INTO jobs(job_id, kind, status, created_at, updated_at) "
                "VALUES('interrupted', 'test', 'running', 'now', 'now')"
            )
        self.assertEqual(job_queue.recover_interrupted(), 1)
        db = research_db.connect()
        try:
            status = db.execute("SELECT status FROM jobs WHERE job_id='interrupted'").fetchone()[0]
        finally:
            db.close()
        self.assertEqual(status, "queued")

    def test_summary_job_persists_provenance_and_embedding(self):
        paper_store.save_paper(
            "2609.21",
            {"arxiv_id": "2609.21", "title": "Transient summary", "abstract": "Abstract"},
            "Old summary",
        )
        with research_db.transaction() as db:
            db.execute(
                "INSERT INTO jobs(job_id, kind, status, created_at, updated_at) "
                "VALUES('summary-test', 'summary_regeneration', 'running', 'now', 'now')"
            )

        class FakeSummarizer:
            last_fallback_reason = ""

            def summarize(self, text, max_length=300, detailed=False):
                return ("A complete technical summary of methods, results, assumptions, and limitations. " * 10).strip()

            def _active_provider(self):
                return "openai"

            def _active_model(self):
                return "test-model"

        vector_db = FakeVectorDB()
        with mock.patch("app.summary_jobs.get_paper_text", return_value="Full paper text " * 20):
            result = summary_jobs._run(
                {"paper_ids": ["2609.21"], "detailed": True},
                {
                    "job_id": "summary-test",
                    "arxiv_client": object(),
                    "pdf_extractor": object(),
                    "vector_db": vector_db,
                    "summarizer": FakeSummarizer(),
                },
            )
        stored = paper_store.get_paper("2609.21")
        self.assertEqual(result["papers_completed"], 1)
        self.assertEqual(stored["summary_provenance"]["status"], "complete")
        self.assertEqual(stored["summary_provenance"]["model"], "test-model")
        self.assertIn("2609.21", vector_db.documents)

    def test_semantic_clusters_are_filtered_by_profile_relevance(self):
        now = datetime.now(timezone.utc)
        papers = [
            {
                "arxiv_id": "2609.11",
                "title": "Kilonova ejecta inference",
                "abstract": "Transient ejecta and opacity modelling",
                "published": (now - timedelta(days=3)).isoformat(),
            },
            {
                "arxiv_id": "2609.12",
                "title": "Kilonova opacity constraints",
                "abstract": "Transient ejecta inference",
                "published": (now - timedelta(days=5)).isoformat(),
            },
            {
                "arxiv_id": "2609.13",
                "title": "Black hole thermodynamics",
                "abstract": "Entropy and geometry",
                "published": (now - timedelta(days=4)).isoformat(),
            },
        ]

        def encode(text):
            lower = text.lower()
            return [1.0, 0.0] if "kilonova" in lower or "transient" in lower else [0.0, 1.0]

        rows = trends.semantic_theme_clusters(
            papers,
            interest_text="kilonova transient inference",
            encode=encode,
            now=now,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["paper_count"], 2)

    def test_database_backup_and_restore(self):
        paper_store.save_paper(
            "2609.20",
            {"arxiv_id": "2609.20", "title": "Original"},
            "Original summary",
        )
        backup_path = Path(self.tempdir.name) / "snapshot.db"
        maintenance.backup_database(backup_path)
        paper_store.update_paper("2609.20", {"title": "Changed"})
        maintenance.restore_database(backup_path)
        self.assertEqual(paper_store.get_paper("2609.20")["title"], "Original")


if __name__ == "__main__":
    unittest.main()
