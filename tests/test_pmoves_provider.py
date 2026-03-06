"""Tests for pmoves_provider — TensorZero provider mode."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pmoves_provider.config import (
    TENSORZERO_CREDENTIAL_NAME,
    TENSORZERO_PROVIDER,
    ProviderMode,
    get_provider_mode,
    get_tensorzero_api_key,
    get_tensorzero_base_url,
)
from pmoves_provider.bootstrap import classify_model_type


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_default_mode_is_tensorzero(self, monkeypatch):
        monkeypatch.delenv("NOTEBOOK_PROVIDER_MODE", raising=False)
        assert get_provider_mode() is ProviderMode.TENSORZERO

    def test_native_mode(self, monkeypatch):
        monkeypatch.setenv("NOTEBOOK_PROVIDER_MODE", "native")
        assert get_provider_mode() is ProviderMode.NATIVE

    def test_invalid_mode_falls_back_to_tensorzero(self, monkeypatch):
        monkeypatch.setenv("NOTEBOOK_PROVIDER_MODE", "bogus")
        assert get_provider_mode() is ProviderMode.TENSORZERO

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("NOTEBOOK_PROVIDER_MODE", "NATIVE")
        assert get_provider_mode() is ProviderMode.NATIVE

    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("TENSORZERO_BASE_URL", raising=False)
        assert get_tensorzero_base_url() == "http://tensorzero-gateway:3030/v1"

    def test_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("TENSORZERO_BASE_URL", "http://custom:9999/v1")
        assert get_tensorzero_base_url() == "http://custom:9999/v1"

    def test_default_api_key(self, monkeypatch):
        monkeypatch.delenv("TENSORZERO_API_KEY", raising=False)
        assert get_tensorzero_api_key() == "pmoves-internal"


# ---------------------------------------------------------------------------
# Model type classification
# ---------------------------------------------------------------------------


class TestModelTypeClassification:
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("gpt-4o", "language"),
            ("claude-sonnet-4-5", "language"),
            ("text-embedding-3-small", "embedding"),
            ("gemma_embed_local", "embedding"),
            ("all-MiniLM-L6-v2", "embedding"),
            ("whisper-1", "speech_to_text"),
            ("tts-1-hd", "text_to_speech"),
            ("bge-m3", "embedding"),
            ("e5-large-v2", "embedding"),
        ],
    )
    def test_classification(self, model_id, expected):
        assert classify_model_type(model_id) == expected


# ---------------------------------------------------------------------------
# Bootstrap — native mode (no-op)
# ---------------------------------------------------------------------------


class TestBootstrapNativeMode:
    @pytest.mark.asyncio
    async def test_native_mode_is_noop(self, monkeypatch):
        monkeypatch.setenv("NOTEBOOK_PROVIDER_MODE", "native")

        with patch(
            "pmoves_provider.bootstrap._ensure_credential"
        ) as mock_cred:
            from pmoves_provider.bootstrap import bootstrap_tensorzero

            await bootstrap_tensorzero()
            mock_cred.assert_not_called()


# ---------------------------------------------------------------------------
# Credential ensure — idempotent create/update
# ---------------------------------------------------------------------------


class TestEnsureCredential:
    @pytest.mark.asyncio
    async def test_creates_new_credential(self):
        """When no matching credential exists, creates one."""
        mock_credential_cls = MagicMock()
        mock_credential_cls.get_by_provider = AsyncMock(return_value=[])

        mock_instance = MagicMock()
        mock_instance.id = "credential:tz1"
        mock_instance.name = TENSORZERO_CREDENTIAL_NAME
        mock_instance.save = AsyncMock()
        mock_credential_cls.return_value = mock_instance

        with patch(
            "pmoves_provider.bootstrap.Credential", mock_credential_cls
        ):
            from pmoves_provider.bootstrap import _ensure_credential

            result = await _ensure_credential(
                "http://tz:3030/v1", "test-key"
            )
            assert result is mock_instance
            mock_instance.save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_updates_existing_credential(self):
        """When matching credential exists, updates it."""
        existing = MagicMock()
        existing.name = TENSORZERO_CREDENTIAL_NAME
        existing.base_url = "http://old:3030/v1"
        existing.id = "credential:tz1"
        existing.save = AsyncMock()

        mock_credential_cls = MagicMock()
        mock_credential_cls.get_by_provider = AsyncMock(return_value=[existing])

        with patch(
            "pmoves_provider.bootstrap.Credential", mock_credential_cls
        ):
            from pmoves_provider.bootstrap import _ensure_credential

            result = await _ensure_credential(
                "http://new:3030/v1", "new-key"
            )
            assert result is existing
            assert existing.base_url == "http://new:3030/v1"
            existing.save.assert_awaited_once()


# ---------------------------------------------------------------------------
# Model discovery — httpx mocking
# ---------------------------------------------------------------------------


class TestDiscoverModels:
    @pytest.mark.asyncio
    async def test_successful_discovery(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4o"},
                {"id": "claude-sonnet-4-5"},
                {"id": "text-embedding-3-small"},
            ]
        }

        with patch("pmoves_provider.bootstrap.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(get=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            from pmoves_provider.bootstrap import _discover_tensorzero_models

            result = await _discover_tensorzero_models(
                "http://tz:3030/v1", "key"
            )
            assert result == ["gpt-4o", "claude-sonnet-4-5", "text-embedding-3-small"]

    @pytest.mark.asyncio
    async def test_connection_error_returns_empty(self):
        import httpx

        with patch("pmoves_provider.bootstrap.httpx.AsyncClient") as mock_client:
            mock_cm = AsyncMock()
            mock_cm.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            from pmoves_provider.bootstrap import _discover_tensorzero_models

            result = await _discover_tensorzero_models(
                "http://unreachable:3030/v1", "key"
            )
            assert result == []
