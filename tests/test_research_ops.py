import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from app import evaluation, index_jobs, paper_store, privacy, research_db, synthesis, trends
from app.corpus_index import index_text_record


class FakeVectorDB:
    def __init__(self):
        self.documents = {}

    def upsert_document(self, document_id, text, metadata):
        self.documents[document_id] = {"text": text, "metadata": metadata}

    def delete_documents(self, owner_type, owner_id):
        prefix = f"{owner_type}:{owner_id}:"
        self.documents = {key: value for key, value in self.documents.items() if not key.startswith(prefix)}

    def search_documents(self, query, top_k=10, paper_ids=None):
        return []


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


if __name__ == "__main__":
    unittest.main()
