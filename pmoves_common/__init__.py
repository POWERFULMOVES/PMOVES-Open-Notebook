"""
PMOVES.AI Common Types Module

Shared type definitions used across all PMOVES services.
This module should be imported as a dependency to ensure type consistency
across the PMOVES.AI ecosystem.

Usage:
    from pmoves_common import ServiceTier, HealthStatus

    tier = ServiceTier.API
    if tier == ServiceTier.AGENT:
        print("Agent tier service")
"""

from enum import Enum


class ServiceTier(str, Enum):
    """
    PMOVES service tiers (6-tier architecture).

    The 6-tier model provides clear separation of concerns and security boundaries:
    - DATA: Infrastructure services (Postgres, Qdrant, Neo4j, MinIO, NATS)
    - API: Data access APIs (PostgREST, Presign, Hi-RAG, GPU Orchestrator)
    - LLM: LLM Gateway (TensorZero) - ONLY tier with external API keys
    - WORKER: Background workers (Extract, LangExtract, PDF-ingest, Notebook-sync)
    - MEDIA: Media processing (YouTube, Whisper, YOLO analyzers)
    - AGENT: Agent orchestration (Agent Zero, Archon, SupaSerch, DeepResearch)

    Security Note: LLM tier is the SINGLE POINT for external API keys.
    All other services call TensorZero internally and never touch provider keys.
    """
    DATA = "data"
    API = "api"
    LLM = "llm"
    WORKER = "worker"
    MEDIA = "media"
    AGENT = "agent"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string value is a valid tier."""
        return value in (t.value for t in cls)

    def __str__(self) -> str:
        return self.value


class HealthStatus(str, Enum):
    """Health status constants for service health checks."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


__all__ = ["ServiceTier", "HealthStatus"]
