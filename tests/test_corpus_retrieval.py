import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import research_db, retrieval
from app.corpus_index import split_text


class FakeVectorDB:
    def search_documents(self, query, top_k=10, paper_ids=None):
        return [
            {
                "document_id": "paper_content:2601.1:2",
                "text": "The inferred ejecta opacity is strongly constrained.",
                "metadata": {
                    "paper_id": "2601.1",
                    "title": "Kilonova inference",
                    "kind": "paper_chunk",
                    "locator_json": '{"source":"pdf","page":4,"chunk":2,"arxiv_id":"2601.1"}',
                },
                "distance": 0.1,
            }
        ]


class CorpusTests(unittest.TestCase):
    def test_page_locator_is_preserved(self):
        text = "Methods\n\n" + "Opacity constraints from the light curve. " * 120
        chunks = split_text(text, source="pdf", page=4)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["locator"]["page"] == 4 for chunk in chunks))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "research.db"
        self.patch = mock.patch.object(research_db, "DB_PATH", self.db_path)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tempdir.cleanup()

    def test_hybrid_result_has_page_citation(self):
        research_db.upsert_document(
            "paper_content:2601.1:2",
            kind="paper_chunk",
            owner_type="paper_content",
            owner_id="2601.1",
            paper_id="2601.1",
            title="Kilonova inference",
            text="The inferred ejecta opacity is strongly constrained.",
            locator={"source": "pdf", "page": 4, "chunk": 2, "arxiv_id": "2601.1"},
        )
        results = retrieval.hybrid_search(
            "ejecta opacity",
            vector_db=FakeVectorDB(),
            paper_ids=["2601.1"],
            limit=3,
        )
        self.assertEqual(results[0]["document_id"], "paper_content:2601.1:2")
        self.assertEqual(retrieval.citation_label(results[0]), "arXiv:2601.1 p.4 chunk 2")


if __name__ == "__main__":
    unittest.main()
