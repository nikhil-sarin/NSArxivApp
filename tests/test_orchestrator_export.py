from __future__ import annotations

from unittest import mock

import pytest

from app import orchestrator_export


def paper() -> dict:
    return {
        "arxiv_id": "2610.12345",
        "title": "A useful transient model",
        "authors": ["Alice Example", "Bob Example", "Carol Example", "Dan Example"],
        "categories": ["astro-ph.HE"],
        "abstract": "Public abstract.",
        "abs_url": "https://arxiv.org/abs/2610.12345",
        "tracking_relevance_score": 87.5,
        "tracking_relevance_reason": "Directly relevant to transient inference.",
        "summary_provenance": {
            "provider": "openai",
            "model": "local-model",
            "api_key": "must-not-export",
        },
    }


def test_relevant_paper_export_is_separately_opt_in(monkeypatch):
    monkeypatch.setenv("LOCAL_ORCHESTRATOR_URL", "http://127.0.0.1:8775/v1/import-bundles")
    monkeypatch.delenv("LOCAL_ORCHESTRATOR_EXPORT_RELEVANT_PAPERS", raising=False)
    assert orchestrator_export.enabled() is False
    with pytest.raises(ValueError, match="not enabled"):
        orchestrator_export.export_relevant_paper(paper(), "Summary")


def test_bundle_is_artifact_only_bounded_and_contains_no_full_text():
    source = paper()
    source["full_text"] = "private or very large extracted text"
    bundle = orchestrator_export.build_paper_bundle(source, "Structured summary")

    assert bundle["source"]["source_system"] == "nsarxivapp.daily-relevance"
    assert bundle["source"]["external_id"] == "2610.12345"
    assert bundle["source"]["trust"] == "public"
    assert bundle["source"]["metadata"]["relevance_score"] == 87.5
    assert bundle["source"]["metadata"]["summary"] == "Structured summary"
    assert bundle["source"]["metadata"]["summary_provenance"] == {
        "provider": "openai",
        "model": "local-model",
    }
    assert bundle["candidates"] == []
    assert bundle["update_existing_source"] is True
    assert "full_text" not in str(bundle)
    assert "must-not-export" not in str(bundle)


@mock.patch("app.orchestrator_export.requests.post")
def test_export_uses_optional_token_and_requires_artifact_response(post, monkeypatch):
    monkeypatch.setenv("LOCAL_ORCHESTRATOR_EXPORT_RELEVANT_PAPERS", "true")
    monkeypatch.setenv("LOCAL_ORCHESTRATOR_API_TOKEN", "private-token")
    response = mock.Mock()
    response.json.return_value = {"artifact_id": "artifact_123", "candidate_ids": []}
    post.return_value = response

    result = orchestrator_export.export_relevant_paper(
        paper(),
        "Summary",
        url="http://127.0.0.1:8775/v1/import-bundles",
    )

    assert result["artifact_id"] == "artifact_123"
    kwargs = post.call_args.kwargs
    assert kwargs["headers"] == {"Authorization": "Bearer private-token"}
    assert kwargs["timeout"] == 15
    assert kwargs["json"]["candidates"] == []
    assert "private-token" not in str(kwargs["json"])


@mock.patch("app.orchestrator_export.requests.post")
def test_export_rejects_non_artifact_response(post, monkeypatch):
    monkeypatch.setenv("LOCAL_ORCHESTRATOR_EXPORT_RELEVANT_PAPERS", "yes")
    response = mock.Mock()
    response.json.return_value = {"status": "ok"}
    post.return_value = response

    with pytest.raises(RuntimeError, match="invalid response"):
        orchestrator_export.export_relevant_paper(
            paper(), "Summary", url="http://127.0.0.1:8775/v1/import-bundles"
        )
