"""Metrics for labelled citation-opportunity launch evaluations."""

from __future__ import annotations


def evaluate(cases: list[dict], predictions: list[dict]) -> dict:
    """Compute precision-first launch metrics from stable case IDs."""
    expected = {case["id"]: case for case in cases}
    actual = {item["id"]: item for item in predictions}
    if set(expected) != set(actual):
        raise ValueError("prediction IDs must exactly match evaluation case IDs")
    tp = fp = fn = negatives = negative_fp = cited = cited_correct = quotes = valid_quotes = unsupported = failures = 0
    latencies = []
    surfaced = 0
    for case_id, case in expected.items():
        prediction = actual[case_id]
        target_strong = case["label"] == "strong_citation_opportunity"
        predicted_strong = prediction.get("classification") == "strong_citation_opportunity"
        tp += target_strong and predicted_strong
        fp += not target_strong and predicted_strong
        fn += target_strong and not predicted_strong
        if case["label"] == "not_relevant":
            negatives += 1
            negative_fp += prediction.get("classification") in {"strong_citation_opportunity", "potentially_useful"}
        if case.get("canonical_citation_present"):
            cited += 1
            cited_correct += bool(prediction.get("reference_check_correct"))
        evidence = prediction.get("evidence", [])
        quotes += len(evidence)
        valid_quotes += sum(item.get("quote_valid") is True for item in evidence)
        unsupported += int(prediction.get("unsupported_claims", 0))
        failures += bool(prediction.get("failed"))
        latencies.append(float(prediction.get("latency_seconds", 0)))
        surfaced += prediction.get("classification") in {"strong_citation_opportunity", "potentially_useful"}
    return {
        "strong_precision": tp / max(tp + fp, 1), "strong_recall": tp / max(tp + fn, 1),
        "not_relevant_false_positive_rate": negative_fp / max(negatives, 1),
        "reference_check_accuracy": cited_correct / max(cited, 1),
        "evidence_quote_validity": valid_quotes / max(quotes, 1),
        "unsupported_claim_rate": unsupported / max(len(cases), 1),
        "average_papers_surfaced": surfaced / max(len(cases), 1),
        "model_failure_rate": failures / max(len(cases), 1),
        "mean_latency_seconds": sum(latencies) / max(len(latencies), 1),
        "launch_ready": quotes > 0 and valid_quotes == quotes and unsupported == 0 and tp / max(tp + fp, 1) >= 0.8,
    }
