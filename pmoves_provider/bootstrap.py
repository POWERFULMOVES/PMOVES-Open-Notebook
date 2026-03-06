"""Bootstrap TensorZero as the default LLM provider for Open Notebook.

On startup (when mode == tensorzero):
  1. Creates/updates a single openai_compatible Credential pointing at TZ
  2. Discovers models via GET /v1/models
  3. Registers discovered models in the Model table

Idempotent — safe to run on every restart.
"""

from typing import List, Optional

import httpx
from loguru import logger
from pydantic import SecretStr

from pmoves_provider.config import (
    TENSORZERO_CREDENTIAL_NAME,
    TENSORZERO_PROVIDER,
    ProviderMode,
    get_provider_mode,
    get_tensorzero_api_key,
    get_tensorzero_base_url,
)


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


async def bootstrap_tensorzero() -> None:
    """Entry point — called from api/main.py lifespan."""
    mode = get_provider_mode()
    if mode is not ProviderMode.TENSORZERO:
        logger.debug("Provider mode is 'native'; skipping TensorZero bootstrap")
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
