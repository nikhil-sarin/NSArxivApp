import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import action_contract, idea_store, project_store, research_db


class ProjectActionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.db_path = root / "research.db"
        self.ideas_path = root / "ideas.json"
        self.patches = [
            mock.patch.object(research_db, "DB_PATH", self.db_path),
            mock.patch.object(idea_store, "STORE_PATH", self.ideas_path),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tempdir.cleanup()

    def test_legacy_idea_becomes_project(self):
        self.ideas_path.write_text(
            json.dumps({
                "paper": {
                    "abc": {
                        "id": "abc",
                        "title": "Opacity project",
                        "description": "Measure kilonova opacity",
                        "status": "active",
                    }
                },
                "grant": {},
            }),
            encoding="utf-8",
        )
        projects = project_store.load_projects()
        self.assertEqual(projects[0]["project_id"], "idea-paper-abc")
        self.assertEqual(projects[0]["title"], "Opacity project")

    def test_action_requires_approval_before_execution(self):
        action_id = action_contract.propose(
            "local-whisperer",
            "calendar.create",
            "Schedule collaboration meeting",
            {"attendees": ["person@example.com"]},
        )
        with self.assertRaises(ValueError):
            action_contract.update_status(action_id, "executed")
        action_contract.update_status(action_id, "approved")
        action_contract.update_status(action_id, "executed")
        self.assertEqual(action_contract.list_actions()[0]["status"], "executed")

    def test_external_handoff_imports_only_valid_proposals(self):
        ids = action_contract.import_proposals(
            "TheLocalWhisperer",
            [{"action_type": "task.create", "title": "Read paper", "payload": {"paper_id": "2609.1"}}],
        )
        self.assertEqual(len(ids), 1)
        self.assertEqual(action_contract.list_actions()[0]["status"], "proposed")
        with self.assertRaises(ValueError):
            action_contract.import_proposals(
                "TheLocalWhisperer",
                [{"action_type": "shell.execute", "title": "Unsafe", "payload": {}}],
            )


if __name__ == "__main__":
    unittest.main()
