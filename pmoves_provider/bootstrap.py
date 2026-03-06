"""Bootstrap TensorZero as the default LLM provider for Open Notebook.

On startup (when mode == tensorzero or hybrid):
  1. Creates/updates a single openai_compatible Credential pointing at TZ
  2. Discovers models via GET /v1/models
  3. Registers discovered models in the Model table

Hybrid mode additionally queries the PMOVES Model Registry to seed
local providers (Ollama, vLLM) directly, bypassing the gateway for
lower latency on local models.

Idempotent — safe to run on every restart.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import httpx
from loguru import logger
from pydantic import SecretStr

from pmoves_provider.config import (
    REGISTRY_CREDENTIAL_SUFFIX,
    TENSORZERO_CREDENTIAL_NAME,
    TENSORZERO_PROVIDER,
    ProviderMode,
    get_model_registry_url,
    get_provider_mode,
    get_tensorzero_api_key,
    get_tensorzero_base_url,
)

# ---------------------------------------------------------------------------
# Registry → Open Notebook type/provider mapping
# ---------------------------------------------------------------------------

REGISTRY_TO_NOTEBOOK_TYPE: Dict[str, str] = {
    "chat": "language",
    "embedding": "embedding",
    "vl": "language",
    "tts": "text_to_speech",
    "audio": "speech_to_text",
}

REGISTRY_TO_NOTEBOOK_PROVIDER: Dict[str, str] = {
    "ollama": "ollama",
    "vllm": "openai_compatible",
    "openai_compatible": "openai_compatible",
    "custom": "openai_compatible",
}


def classify_model_type(model_id: str) -> str:
    """Heuristic: infer Open Notebook model type from a model ID string."""
    lower = model_id.lower()
    if any(tok in lower for tok in ("embed", "e5-", "bge-", "minilm")):
        return "embedding"
    if any(tok in lower for tok in ("whisper", "stt", "speech-to-text")):
        return "speech_to_text"
    if any(tok in lower for tok in ("tts", "text-to-speech")):
        return "text_to_speech"
    return "language"


def map_registry_model_type(registry_type: str) -> Optional[str]:
    """Map a registry model_type to Open Notebook's type string.

    Returns None for unmappable types (reranker, image, etc.).
    """
    return REGISTRY_TO_NOTEBOOK_TYPE.get(registry_type)


def map_registry_provider(registry_provider_type: str) -> str:
    """Map a registry provider_type to Open Notebook's provider string."""
    return REGISTRY_TO_NOTEBOOK_PROVIDER.get(
        registry_provider_type, "openai_compatible"
    )


async def bootstrap_tensorzero() -> None:
    """Entry point — called from api/main.py lifespan."""
    mode = get_provider_mode()
    if mode is ProviderMode.NATIVE:
        logger.debug("Provider mode is 'native'; skipping TensorZero bootstrap")
        return
    if mode is ProviderMode.HYBRID:
        await bootstrap_hybrid()
        return

    base_url = get_tensorzero_base_url()
    api_key = get_tensorzero_api_key()

    logger.info(f"TensorZero bootstrap: mode={mode.value}, url={base_url}")

    credential = await _ensure_credential(base_url, api_key)
    if credential is None:
        logger.warning("TensorZero bootstrap: could not create/update credential")
        return

    discovered = await _discover_tensorzero_models(base_url, api_key)
    if discovered:
        await _register_models(credential, discovered)

    logger.success(
        f"TensorZero bootstrap complete: credential={credential.id}, "
        f"discovered={len(discovered)} models"
    )


async def _ensure_credential(
    base_url: str, api_key: str
) -> Optional["Credential"]:  # noqa: F821
    """Create or update the managed TensorZero credential."""
    from open_notebook.domain.credential import Credential

    existing = await Credential.get_by_provider(TENSORZERO_PROVIDER)

    # Find the PMOVES-managed credential by name
    tz_cred = None
    for cred in existing:
        if cred.name == TENSORZERO_CREDENTIAL_NAME:
            tz_cred = cred
            break

    if tz_cred is not None:
        # Update in place
        changed = False
        if tz_cred.base_url != base_url:
            tz_cred.base_url = base_url
            changed = True
        # Always refresh the api_key (it's encrypted at rest)
        tz_cred.api_key = SecretStr(api_key)
        changed = True  # api_key comparison not meaningful with SecretStr

        if changed:
            await tz_cred.save()
            logger.info("TensorZero credential updated")
        return tz_cred

    # Create new credential
    tz_cred = Credential(
        name=TENSORZERO_CREDENTIAL_NAME,
        provider=TENSORZERO_PROVIDER,
        modalities=["language", "embedding"],
        api_key=SecretStr(api_key),
        base_url=base_url,
    )
    await tz_cred.save()
    logger.info(f"TensorZero credential created: {tz_cred.id}")
    return tz_cred


async def _discover_tensorzero_models(
    base_url: str, api_key: str
) -> List[str]:
    """GET /v1/models from TensorZero (OpenAI-compatible endpoint)."""
    url = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {api_key}"}
            )
            resp.raise_for_status()
            data = resp.json()
            model_ids = [m["id"] for m in data.get("data", [])]
            logger.info(f"TensorZero model discovery: {len(model_ids)} models found")
            return model_ids
    except httpx.ConnectError:
        logger.warning(
            f"TensorZero not reachable at {url} — model discovery skipped "
            "(will retry on next restart)"
        )
        return []
    except Exception as e:
        logger.warning(f"TensorZero model discovery failed: {e}")
        return []


async def _register_models(
    credential: "Credential", discovered: List[str]  # noqa: F821
) -> None:
    """Register discovered models, skipping duplicates."""
    from open_notebook.ai.models import Model

    # Fetch existing models for this credential
    existing_models = await Model.get_by_credential(credential.id)
    existing_names = {m.name for m in existing_models}

    created = 0
    for model_id in discovered:
        if model_id in existing_names:
            continue

        model_type = classify_model_type(model_id)
        model = Model(
            name=model_id,
            provider=TENSORZERO_PROVIDER,
            type=model_type,
            credential=credential.id,
        )
        await model.save()
        created += 1

    if created:
        logger.info(f"TensorZero: registered {created} new models")


# ---------------------------------------------------------------------------
# Hybrid mode — TensorZero + Model Registry
# ---------------------------------------------------------------------------


def _registry_credential_name(provider_name: str) -> str:
    """Generate a managed credential name for a registry provider.

    Example: "Ollama (PMOVES Registry)"
    """
    return f"{provider_name.title()} {REGISTRY_CREDENTIAL_SUFFIX}"


async def _query_model_registry() -> (
    Dict[Tuple[str, str], List[dict]]
):
    """Fetch models from PMOVES Model Registry, grouped by (provider_type, api_base).

    Returns a dict keyed by (provider_type, api_base) with lists of model records.
    Each model record has at least: model_id, model_type, provider_type, api_base.
    """
    url = f"{get_model_registry_url().rstrip('/')}/api/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            models = resp.json()
    except httpx.ConnectError:
        logger.warning(
            f"Model Registry not reachable at {url} — "
            "registry seeding skipped (will retry on next restart)"
        )
        return {}
    except Exception as e:
        logger.warning(f"Model Registry query failed: {e}")
        return {}

    # Group by (provider_type, api_base) so we create one Credential per endpoint
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for model in models:
        provider_type = model.get("provider_type", "custom")
        api_base = model.get("api_base", "")
        grouped[(provider_type, api_base)].append(model)

    logger.info(
        f"Model Registry: {len(models)} models across "
        f"{len(grouped)} provider endpoints"
    )
    return dict(grouped)


async def _ensure_registry_credential(
    provider_name: str,
    provider_type: str,
    api_base: str,
) -> Optional["Credential"]:  # noqa: F821
    """Create or update a managed credential for a registry provider.

    Uses the same idempotent pattern as _ensure_credential.
    Local providers (Ollama, vLLM) get a sentinel API key.
    """
    from open_notebook.domain.credential import Credential

    cred_name = _registry_credential_name(provider_name)
    on_provider = map_registry_provider(provider_type)

    existing = await Credential.get_by_provider(on_provider)

    # Find by managed name
    reg_cred = None
    for cred in existing:
        if cred.name == cred_name:
            reg_cred = cred
            break

    # Determine API key — local providers don't need real auth
    needs_real_key = provider_type in ("openai_compatible",)
    if needs_real_key:
        env_key = f"{provider_name.upper()}_API_KEY"
        api_key = os.environ.get(env_key, "pmoves-registry")
    else:
        api_key = "pmoves-registry"

    modalities = ["language", "embedding"]

    if reg_cred is not None:
        changed = False
        if reg_cred.base_url != api_base:
            reg_cred.base_url = api_base
            changed = True
        reg_cred.api_key = SecretStr(api_key)
        changed = True

        if changed:
            await reg_cred.save()
            logger.info(f"Registry credential updated: {cred_name}")
        return reg_cred

    reg_cred = Credential(
        name=cred_name,
        provider=on_provider,
        modalities=modalities,
        api_key=SecretStr(api_key),
        base_url=api_base,
    )
    await reg_cred.save()
    logger.info(f"Registry credential created: {cred_name} ({reg_cred.id})")
    return reg_cred


async def _register_registry_models(
    credential: "Credential",  # noqa: F821
    models: List[dict],
) -> None:
    """Register models from the registry, skipping duplicates and unmappable types."""
    from open_notebook.ai.models import Model

    existing_models = await Model.get_by_credential(credential.id)
    existing_names = {m.name for m in existing_models}

    created = 0
    skipped = 0
    for model in models:
        model_id = model.get("model_id", model.get("name", ""))
        if not model_id or model_id in existing_names:
            continue

        registry_type = model.get("model_type", "chat")
        notebook_type = map_registry_model_type(registry_type)
        if notebook_type is None:
            skipped += 1
            continue

        on_provider = map_registry_provider(
            model.get("provider_type", "custom")
        )
        m = Model(
            name=model_id,
            provider=on_provider,
            type=notebook_type,
            credential=credential.id,
        )
        await m.save()
        created += 1

    if created:
        logger.info(f"Registry: registered {created} new models")
    if skipped:
        logger.debug(f"Registry: skipped {skipped} unmappable model types")


async def bootstrap_hybrid() -> None:
    """Hybrid mode: seed TensorZero (Phase A) + Model Registry (Phase B).

    Phases are independent — failure in one does not block the other.
    """
    logger.info("Hybrid bootstrap: starting Phase A (TensorZero)")

    # Phase A: TensorZero gateway
    try:
        base_url = get_tensorzero_base_url()
        api_key = get_tensorzero_api_key()

        tz_cred = await _ensure_credential(base_url, api_key)
        if tz_cred:
            discovered = await _discover_tensorzero_models(base_url, api_key)
            if discovered:
                await _register_models(tz_cred, discovered)
            logger.info(
                f"Phase A complete: credential={tz_cred.id}, "
                f"models={len(discovered)}"
            )
        else:
            logger.warning("Phase A: could not create TensorZero credential")
    except Exception as e:
        logger.error(f"Phase A (TensorZero) failed: {e}")

    # Phase B: PMOVES Model Registry
    logger.info("Hybrid bootstrap: starting Phase B (Model Registry)")
    try:
        grouped = await _query_model_registry()
        for (provider_type, api_base), models in grouped.items():
            provider_name = provider_type
            cred = await _ensure_registry_credential(
                provider_name, provider_type, api_base
            )
            if cred:
                await _register_registry_models(cred, models)
        logger.info(
            f"Phase B complete: {len(grouped)} registry provider(s) seeded"
        )
    except Exception as e:
        logger.error(f"Phase B (Model Registry) failed: {e}")

    logger.success("Hybrid bootstrap complete")
