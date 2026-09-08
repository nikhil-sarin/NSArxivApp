"""Explicit provider/data routing rules for research workflows."""

from __future__ import annotations

import os


LOCAL_PROVIDERS = {"ollama"}
CLOUD_PROVIDERS = {"gemini", "anthropic", "openai"}
POLICIES = {"local_only", "paper_cloud", "allow_cloud"}


class PrivacyViolation(RuntimeError):
    """Raised when a workflow attempts to send restricted data to a provider."""


def active_policy() -> str:
    policy = os.getenv("DATA_ROUTING_POLICY", "paper_cloud").strip().lower()
    return policy if policy in POLICIES else "paper_cloud"


def cloud_allowed(*, contains_private_data: bool) -> bool:
    policy = active_policy()
    if policy == "allow_cloud":
        return True
    if policy == "paper_cloud":
        return not contains_private_data
    return False


def choose_provider(
    configured_provider: str,
    *,
    preferred_cloud_provider: str | None = None,
    contains_private_data: bool = False,
) -> str:
    """Choose a provider without crossing the configured privacy boundary."""
    configured = configured_provider.strip().lower()
    preferred = (preferred_cloud_provider or "").strip().lower()
    if preferred in CLOUD_PROVIDERS and cloud_allowed(contains_private_data=contains_private_data):
        return preferred
    if configured in CLOUD_PROVIDERS and not cloud_allowed(contains_private_data=contains_private_data):
        local_fallback = os.getenv("PRIVATE_LLM_PROVIDER", "ollama").strip().lower()
        if local_fallback not in LOCAL_PROVIDERS:
            raise PrivacyViolation(
                f"{active_policy()} blocks cloud processing and PRIVATE_LLM_PROVIDER is not local"
            )
        return local_fallback
    return configured or "ollama"


def routing_label(provider: str, *, contains_private_data: bool) -> str:
    location = "local" if provider in LOCAL_PROVIDERS else "cloud"
    sensitivity = "private context" if contains_private_data else "paper content"
    return f"{provider} ({location}; {sensitivity}; policy={active_policy()})"
