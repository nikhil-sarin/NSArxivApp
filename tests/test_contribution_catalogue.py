import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import contribution_catalogue


class ContributionCatalogueTests(unittest.TestCase):
    def test_synthetic_catalogue_is_valid_and_aliases_normalize(self):
        catalogue = contribution_catalogue.validate({
            "schema_version": "1.0",
            "owner": "Example Researcher",
            "contributions": [{
                "id": "example-method",
                "name": "Example Method",
                "kind": "method",
                "aliases": ["Example Method", "Example Toolkit"],
                "canonical_citations": [{"doi": "10.0000/example"}],
                "strong_signals": ["example method analysis"],
                "weak_signals": [],
                "contact_framing": "Describe the relevant method.",
            }],
        })
        contribution = contribution_catalogue.get(catalogue, "example-method")
        self.assertEqual(contribution["aliases"], ["example method", "example toolkit"])

    def test_public_install_does_not_load_maintainer_example_implicitly(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            contribution_catalogue,
            "DEFAULT_PATH",
            Path(directory) / "contributions.json",
        ), mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(contribution_catalogue.load_if_configured())
            self.assertEqual(contribution_catalogue.enabled_contributions(), [])
            with self.assertRaisesRegex(
                contribution_catalogue.CatalogueError,
                "could not read contribution catalogue",
            ):
                contribution_catalogue.load()

    def test_explicit_missing_catalogue_is_reported_as_configuration_error(self):
        missing = "/definitely/missing/contributions.json"
        with mock.patch.dict(
            "os.environ", {"NSARXIV_CONTRIBUTIONS_PATH": missing}, clear=True
        ):
            with self.assertRaisesRegex(
                contribution_catalogue.CatalogueError, "file not found"
            ):
                contribution_catalogue.load_if_configured()

    def test_duplicate_and_missing_identifier_errors_include_id(self):
        entry = {
            "id": "duplicate", "name": "Name", "kind": "method", "aliases": [],
            "strong_signals": ["direct method"], "weak_signals": [], "exclusions": [],
            "contact_framing": "Offer help", "canonical_citations": [],
        }
        with self.assertRaisesRegex(contribution_catalogue.CatalogueError, "duplicate"):
            contribution_catalogue.validate({"schema_version": "1.0", "owner": "Owner", "contributions": [entry]})
        valid = {**entry, "unpublished": True}
        with self.assertRaisesRegex(contribution_catalogue.CatalogueError, "duplicate id"):
            contribution_catalogue.validate({
                "schema_version": "1.0", "owner": "Owner", "contributions": [valid, valid],
            })

if __name__ == "__main__":
    unittest.main()
