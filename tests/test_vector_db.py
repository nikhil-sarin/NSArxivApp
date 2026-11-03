import json
import unittest
from unittest import mock

from app.vector_db import PaperVectorDB


class VectorMetadataTests(unittest.TestCase):
    def test_nested_paper_metadata_is_normalized_before_upsert(self):
        vector_db = PaperVectorDB.__new__(PaperVectorDB)
        vector_db.embedder = mock.Mock()
        vector_db.embedder.encode.return_value.tolist.return_value = [0.1, 0.2]
        vector_db.collection = mock.Mock()

        vector_db.add_paper(
            "2609.12044",
            "Test paper",
            "Summary",
            {
                "authors": ["A. Author", "B. Author"],
                "summary_provenance": {
                    "status": "complete",
                    "provider": "openai",
                },
                "missing": None,
            },
        )

        metadata = vector_db.collection.upsert.call_args.kwargs["metadatas"][0]
        self.assertEqual(json.loads(metadata["authors"]), ["A. Author", "B. Author"])
        self.assertEqual(json.loads(metadata["summary_provenance"])["status"], "complete")
        self.assertNotIn("missing", metadata)


if __name__ == "__main__":
    unittest.main()
