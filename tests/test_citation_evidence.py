import unittest

from app.citation_evidence import build_evidence_packet


CONTRIBUTION = {
    "id": "redback", "aliases": ["redback"],
    "canonical_citations": [{
        "title": "REDBACK a Bayesian inference software package for electromagnetic transients",
        "doi": "10.1093/mnras/stae1238", "arxiv_id": "2308.12806", "bibcode": "2024MNRAS.531.1203S",
    }],
    "strong_signals": ["bilby transient light curve inference"],
    "weak_signals": ["transient parameter inference", "kilonova light curve modelling"],
    "exclusions": ["bilby gravitational wave inference"],
}


class CitationEvidenceTests(unittest.TestCase):
    def test_exact_evidence_and_stable_locator(self):
        text = (
            "Methods\n\nWe use Bilby for transient light curve inference on the optical observations.\n\n"
            "Results\n\nThe fit constrains the ejecta mass.\n\nReferences\n\nOther et al. 2025"
        )
        packet = build_evidence_packet("2609.1", text, CONTRIBUTION, owner_name_variants=["N. Sarin"])
        self.assertTrue(packet["candidate"])
        self.assertIn(packet["passages"][0]["quote"], text)
        self.assertIn("Methods", packet["passages"][0]["locator"])
        self.assertFalse(packet["reference_check"]["canonical_citation_found"])

    def test_doi_arxiv_bibcode_and_title_detection(self):
        identifiers = [
            "doi:10.1093/mnras/stae1238", "arXiv:2308.12806", "2024MNRAS.531.1203S",
            "REDBACK: a Bayesian inference software package for electromagnetic transients",
        ]
        for identifier in identifiers:
            packet = build_evidence_packet(
                "2609.1", f"Methods\n\nBilby transient light curve inference.\n\nReferences\n\n{identifier}",
                CONTRIBUTION, owner_name_variants=["Nikhil Sarin"],
            )
            self.assertTrue(packet["reference_check"]["canonical_citation_found"], identifier)
            self.assertFalse(packet["candidate"], identifier)

    def test_owner_name_alone_is_not_canonical_and_exclusion_suppresses(self):
        text = "Methods\n\nN. Sarin uses Bilby for gravitational wave inference.\n\nReferences\n\nN. Sarin 2024"
        packet = build_evidence_packet("2609.1", text, CONTRIBUTION, owner_name_variants=["N. Sarin"])
        self.assertTrue(packet["reference_check"]["owner_name_found"])
        self.assertFalse(packet["reference_check"]["canonical_citation_found"])
        self.assertFalse(packet["candidate"])
        self.assertTrue(packet["matched_exclusions"])

    def test_identifier_suppresses_when_reference_heading_is_not_extracted(self):
        text = (
            "Methods\n\nWe use Bilby for transient light curve inference.\n\n"
            "Sarin et al. 2024, doi:10.1093/mnras/stae1238"
        )
        packet = build_evidence_packet("2609.1", text, CONTRIBUTION, owner_name_variants=[])
        self.assertTrue(packet["reference_check"]["canonical_citation_found"])
        self.assertEqual(packet["reference_check"]["matched_identifiers"][0]["location"], "document")
        self.assertFalse(packet["candidate"])


if __name__ == "__main__":
    unittest.main()
