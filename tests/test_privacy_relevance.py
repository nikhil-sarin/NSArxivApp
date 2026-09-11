import os
import unittest
from unittest import mock

from app import privacy, relevance


class PrivacyTests(unittest.TestCase):
    def test_private_context_routes_to_local_provider_by_default(self):
        with mock.patch.dict(os.environ, {"DATA_ROUTING_POLICY": "paper_cloud"}, clear=False):
            provider = privacy.choose_provider(
                "gemini",
                preferred_cloud_provider="gemini",
                contains_private_data=True,
            )
        self.assertEqual(provider, "ollama")

    def test_public_paper_context_can_use_cloud(self):
        with mock.patch.dict(os.environ, {"DATA_ROUTING_POLICY": "paper_cloud"}, clear=False):
            provider = privacy.choose_provider(
                "ollama",
                preferred_cloud_provider="gemini",
                contains_private_data=False,
            )
        self.assertEqual(provider, "gemini")


class RelevanceTests(unittest.TestCase):
    def test_profile_match_ranks_above_unrelated_paper(self):
        papers = [
            {"arxiv_id": "1", "title": "Kilonova opacity constraints", "summary": "neutron star merger ejecta"},
            {"arxiv_id": "2", "title": "Protein folding", "summary": "amino acid structure"},
        ]
        scored = relevance.score_papers(
            papers,
            interest_text="kilonova neutron star merger opacity Bayesian inference",
        )
        self.assertEqual(scored[0]["arxiv_id"], "1")
        self.assertGreater(scored[0]["relevance_score"], scored[1]["relevance_score"])

    def test_negative_feedback_reduces_score(self):
        paper = {"arxiv_id": "1", "title": "Jet simulations", "summary": "relativistic jet modelling"}
        baseline = relevance.score_papers([paper], interest_text="jet modelling")[0]["relevance_score"]
        penalized = relevance.score_papers(
            [paper],
            interest_text="jet modelling",
            negative_papers=[paper],
        )[0]["relevance_score"]
        self.assertLess(penalized, baseline)

    def test_tracking_uses_separate_semantic_preference_documents(self):
        def encode(text):
            if "transient astrophysics" in text or "CSM framework" in text:
                return [1.0, 0.0]
            if "supernova interacting" in text:
                return [1.0, 0.0]
            return [0.0, 1.0]

        scored = relevance.score_tracking_papers(
            [
                {"arxiv_id": "relevant", "title": "supernova interacting with circumstellar material"},
                {"arxiv_id": "unrelated", "title": "point particles in mathematical general relativity"},
            ],
            profile={"research_areas": "transient astrophysics"},
            ideas=[],
            contributions=[{"id": "csm", "name": "CSM framework"}],
            encode=encode,
        )

        self.assertEqual(scored[0]["arxiv_id"], "relevant")
        self.assertGreater(scored[0]["relevance_score"], scored[1]["relevance_score"])

    def test_tracking_exclusions_override_semantic_similarity(self):
        scored = relevance.score_tracking_papers(
            [{"arxiv_id": "solar", "title": "Solar corona transient"}],
            profile={
                "research_areas": "transient astrophysics",
                "tracking_exclusions": "solar corona",
            },
            ideas=[],
            contributions=[],
            encode=lambda text: [1.0, 0.0],
        )

        self.assertEqual(scored[0]["relevance_score"], 0.0)
        self.assertIn("Excluded", scored[0]["relevance_reason"])


if __name__ == "__main__":
    unittest.main()
