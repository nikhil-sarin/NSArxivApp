import unittest
from datetime import datetime, timedelta, timezone

from app import contribution_catalogue
from app.citation_evidence import (
    build_evidence_packet,
    citation_opportunity_ineligibility_reason,
    contribution_predates_paper,
    paper_is_already_published,
)


CONTRIBUTION = {
    "id": "redback", "aliases": ["redback"],
    "canonical_citations": [{
        "title": "REDBACK a Bayesian inference software package for electromagnetic transients",
        "doi": "10.1093/mnras/stae1238", "arxiv_id": "2308.12806", "bibcode": "2024MNRAS.531.1203S",
    }],
    "strong_signals": ["bilby transient light curve inference"],
    "weak_signals": ["transient parameter inference", "kilonova light curve modelling"],
    "exclusions": ["bilby gravitational wave inference"],
    "discovery_rules": [{
        "name": "Bilby EM inference", "strength": "strong",
        "all": [["bilby"], ["bayesian inference", "fitting"], ["radio", "transient"]],
    }],
}


class CitationEvidenceTests(unittest.TestCase):
    def test_temporal_and_publication_eligibility(self):
        contribution = {"canonical_citations": [{"arxiv_id": "2605.19571"}]}
        self.assertTrue(contribution_predates_paper("2609.09520v1", contribution))
        self.assertFalse(contribution_predates_paper("2509.09520", contribution))
        self.assertFalse(contribution_predates_paper("2605.19571", contribution))
        self.assertTrue(paper_is_already_published({"comment": "Accepted for publication in MNRAS"}))
        self.assertTrue(paper_is_already_published({"comment": "A&A, accepted"}))
        self.assertTrue(paper_is_already_published({"comment": "ApJ accepted"}))
        self.assertTrue(paper_is_already_published({"journal_ref": "ApJ 999, 1"}))
        self.assertTrue(paper_is_already_published({"doi": "10.1234/example"}))
        self.assertFalse(paper_is_already_published({"comment": "Submitted to ApJ"}))
        self.assertFalse(paper_is_already_published({"comment": "Not yet accepted"}))

    def test_citation_opportunity_expires_after_configured_window(self):
        now = datetime(2026, 9, 23, tzinfo=timezone.utc)
        self.assertIsNone(citation_opportunity_ineligibility_reason(
            {"published": (now - timedelta(days=30)).isoformat()},
            now=now,
            max_age_days=30,
        ))
        self.assertEqual(citation_opportunity_ineligibility_reason(
            {"published": (now - timedelta(days=31)).isoformat()},
            now=now,
            max_age_days=30,
        ), "opportunity_window_expired")
        self.assertEqual(citation_opportunity_ineligibility_reason(
            {
                "published": (now - timedelta(days=2)).isoformat(),
                "comment": "MNRAS accepted",
            },
            now=now,
            max_age_days=30,
        ), "accepted_or_published")

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

    def test_grouped_concepts_match_language_not_covered_by_fixed_phrase(self):
        text = (
            "Methods\n\nWe fitted the radio spectrum at each epoch.\n\n"
            "We performed Bayesian inference using bilby with the dynesty sampler.\n\n"
            "The transient model describes the data.\n\nReferences\n\nOther et al. 2025"
        )
        packet = build_evidence_packet("2609.2", text, CONTRIBUTION, owner_name_variants=[])
        self.assertTrue(packet["candidate"])
        self.assertEqual(packet["matched_strong_rules"], ["Bilby EM inference"])
        self.assertIn("Bayesian inference using bilby", packet["passages"][0]["quote"])

    def test_grouped_concepts_still_respect_canonical_citation_and_exclusions(self):
        cited = (
            "We performed Bayesian inference using bilby to fit the radio transient. "
            "See doi:10.1093/mnras/stae1238."
        )
        self.assertFalse(build_evidence_packet(
            "2609.3", cited, CONTRIBUTION, owner_name_variants=[]
        )["candidate"])

        gravitational_wave = "We performed bilby fitting for gravitational wave inference of a transient."
        self.assertFalse(build_evidence_packet(
            "2609.4", gravitational_wave, CONTRIBUTION, owner_name_variants=[]
        )["candidate"])

    def test_catalogue_rules_cover_bilby_mixing_and_csm_failure_modes(self):
        catalogue = contribution_catalogue.load()
        by_id = {item["id"]: item for item in catalogue["contributions"]}
        cases = [
            (
                "redback",
                "We used Bilby with dynesty for Bayesian inference of the radio transient spectrum.",
            ),
            (
                "nickel-mixing-supernovae",
                "Our hydrodynamical supernova models assume extensive mixing of 56 Ni through the ejecta.",
            ),
            (
                "generalised-csm-framework",
                "The CSM interaction light curve requires a broken density profile and a reconstructed mass loss history.",
            ),
            (
                "generalised-csm-framework",
                "We convert the terminal mass loss history to lookback time using a fixed wind velocity.",
            ),
            (
                "at2020blt-low-efficiency-grb",
                "This gamma-ray quiet relativistic explosion had no prompt gamma-ray counterpart and shows how events evade traditional gamma-ray triggers.",
            ),
        ]
        for contribution_id, text in cases:
            with self.subTest(contribution_id=contribution_id, text=text):
                packet = build_evidence_packet(
                    "2609.test", text, by_id[contribution_id], owner_name_variants=[]
                )
                self.assertTrue(packet["candidate"])
                self.assertTrue(packet["matched_strong_rules"])

        redback = by_id["redback"]
        redback_cases = [
            "We used nested sampling for parameter estimation of a multi-band kilonova light curve.",
            "We fitted the bolometric supernova light curve with a bespoke magnetar forward model.",
            "We used MOSFiT for Bayesian inference of the tidal disruption transient light curve.",
        ]
        for text in redback_cases:
            with self.subTest(redback_text=text):
                packet = build_evidence_packet(
                    "2609.test", text, redback, owner_name_variants=[]
                )
                self.assertTrue(packet["candidate"])
                self.assertTrue(packet["matched_strong_rules"])

        excluded = build_evidence_packet(
            "2609.test",
            "We use a periodogram variability analysis for pulsar timing inference.",
            redback,
            owner_name_variants=[],
        )
        self.assertFalse(excluded["candidate"])
        self.assertIn("pulsar timing inference", excluded["matched_exclusions"])

        kilonova = by_id["magnetar-driven-kilonovae"]
        supernova_only = (
            "A magnetar powered supernova receives energy injection into its ejecta and light curve."
        )
        self.assertFalse(build_evidence_packet(
            "2609.test", supernova_only, kilonova, owner_name_variants=[]
        )["candidate"])


if __name__ == "__main__":
    unittest.main()
