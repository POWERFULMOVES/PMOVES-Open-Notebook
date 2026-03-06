"""Configuration for PMOVES TensorZero provider mode."""

import os
from enum import Enum


class ProviderMode(str, Enum):
    """How Open Notebook sources its LLM/embedding providers."""

    TENSORZERO = "tensorzero"
    NATIVE = "native"
    HYBRID = "hybrid"


# Credential record constants
TENSORZERO_CREDENTIAL_NAME = "TensorZero Gateway (PMOVES)"
TENSORZERO_PROVIDER = "openai_compatible"
REGISTRY_CREDENTIAL_SUFFIX = "(PMOVES Registry)"


def get_provider_mode() -> ProviderMode:
    """Read NOTEBOOK_PROVIDER_MODE from env (default: tensorzero)."""
    raw = os.environ.get("NOTEBOOK_PROVIDER_MODE", "tensorzero").strip().lower()
    try:
        return ProviderMode(raw)
    except ValueError:
        return ProviderMode.TENSORZERO


def get_tensorzero_base_url() -> str:
    """TensorZero gateway base URL (OpenAI-compatible /v1 prefix)."""
    return os.environ.get(
        "TENSORZERO_BASE_URL", "http://tensorzero-gateway:3030/v1"
    )


def get_tensorzero_api_key() -> str:
    """API key for TensorZero gateway (internal traffic, usually a sentinel)."""
    return os.environ.get("TENSORZERO_API_KEY", "pmoves-internal")


def get_model_registry_url() -> str:
    """PMOVES Model Registry base URL for hybrid mode."""
    return os.environ.get("MODEL_REGISTRY_URL", "http://model-registry:8110")
