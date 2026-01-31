"""
PMOVES.AI Service Announcer Template

NATS service discovery announcer for all PMOVES services.
Publishes service announcements to the services.announce.v1 subject.

This module provides:
- ServiceAnnouncer: Main class for announcing service availability
- ServiceAnnouncement: Data class for announcement messages
- BackgroundAnnouncer: Periodic re-announcement for long-running services
- announce_service(): Convenience function for one-time announcements

Usage:
    from pmoves_announcer import ServiceAnnouncer, announce_service

    # Create announcement
    announcer = ServiceAnnouncer(
        slug="my-service",
        name="My Service",
        url="http://my-service:8080",
        port=8080,
        tier="api"
    )

    # Announce on startup
    await announcer.announce()

    # Or use the convenience function
    await announce_service(
        slug="my-service",
        name="My Service",
        url="http://my-service:8080",
        port=8080,
        tier="api"
    )

NATS Subject: services.announce.v1
Message Format: JSON with slug, name, url, health_check, tier, port, timestamp, metadata
"""

import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Import ServiceTier from shared types if available, otherwise define locally
try:
    from pmoves_common import ServiceTier
except ImportError:
    from enum import Enum

    class ServiceTier(str, Enum):
        """PMOVES service tiers (6-tier architecture)."""
        DATA = "data"
        API = "api"
        LLM = "llm"
        MEDIA = "media"
        AGENT = "agent"
        WORKER = "worker"


@dataclass
class ServiceAnnouncement:
    """
    Service announcement message format for NATS.

    Services publish announcements on the `services.announce.v1` subject
    to notify other services of their availability and configuration.
    """

    slug: str
    name: str
    url: str
    health_check: str
    tier: ServiceTier
    port: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    # NATS subject for announcements
    SUBJECT: str = "services.announce.v1"

    def to_json(self) -> str:
        """Convert to JSON for NATS publishing."""
        data = {
            "slug": self.slug,
            "name": self.name,
            "url": self.url,
            "health_check": self.health_check,
            "tier": self.tier.value if isinstance(self.tier, ServiceTier) else self.tier,
            "port": self.port,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str | dict) -> "ServiceAnnouncement":
        """Parse from JSON message."""
        if isinstance(data, str):
            data = json.loads(data)
        return cls(
            slug=data["slug"],
            name=data["name"],
            url=data["url"],
            health_check=data["health_check"],
            tier=ServiceTier(data["tier"]),
            port=data["port"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


class ServiceAnnouncer:
    """
    NATS service announcer for PMOVES services.

    Handles announcing service availability to the PMOVES service mesh.
    """

    def __init__(
        self,
        slug: str,
        name: str,
        url: str,
        port: int,
        tier: ServiceTier | str,
        health_check: str = None,
        nats_url: str = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize the service announcer.

        Args:
            slug: Unique service identifier (e.g., "hirag-v2")
            name: Human-readable service name
            url: Full service URL
            port: Service port number
            tier: Service tier (api, agent, worker, etc.)
            health_check: Health check URL (defaults to url + /healthz)
            nats_url: NATS server URL (defaults to NATS_URL env var)
            metadata: Additional service metadata
        """
        self.slug = slug
        self.name = name
        self.url = url
        self.port = port

        if isinstance(tier, str):
            tier = ServiceTier(tier.lower())
        self.tier = tier

        self.health_check = health_check or f"{url.rstrip('/')}/healthz"
        self.nats_url = nats_url or os.getenv("NATS_URL", "nats://nats:4222")
        self.metadata = metadata or {}

    def create_announcement(self) -> ServiceAnnouncement:
        """Create a service announcement object."""
        return ServiceAnnouncement(
            slug=self.slug,
            name=self.name,
            url=self.url,
            health_check=self.health_check,
            tier=self.tier,
            port=self.port,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=self.metadata,
        )

    async def announce(self) -> bool:
        """
        Publish service announcement to NATS.

        Returns:
            True if announcement published successfully
        """
        try:
            from nats.aio.client import Client as NATS

            announcement = self.create_announcement()

            nc = await NATS.connect(self.nats_url, connect_timeout=5)
            await nc.publish(
                ServiceAnnouncement.SUBJECT,
                announcement.to_json().encode(),
            )
            await nc.flush()
            await nc.close()

            return True
        except Exception as e:
            print(f"Failed to announce service: {e}")
            return False

    async def announce_with_retry(
        self, max_retries: int = 3, delay: float = 1.0
    ) -> bool:
        """
        Announce service with retry logic.

        Args:
            max_retries: Maximum number of retry attempts
            delay: Delay between retries in seconds

        Returns:
            True if announcement published successfully
        """
        for attempt in range(max_retries):
            if await self.announce():
                return True
            if attempt < max_retries - 1:
                await asyncio.sleep(delay * (2**attempt))  # Exponential backoff
        return False


async def announce_service(
    slug: str,
    name: str,
    url: str,
    port: int,
    tier: ServiceTier | str,
    health_check: str = None,
    nats_url: str = None,
    metadata: Dict[str, Any] = None,
) -> bool:
    """
    Convenience function to announce a service.

    Args:
        slug: Unique service identifier
        name: Human-readable service name
        url: Full service URL
        port: Service port
        tier: Service tier
        health_check: Health check URL
        nats_url: NATS server URL
        metadata: Additional metadata

    Returns:
        True if announcement successful

    Example:
        await announce_service(
            slug="hirag-v2",
            name="Hi-RAG Gateway v2",
            url="http://hi-rag-gateway-v2:8086",
            port=8086,
            tier="api",
            metadata={"gpu_port": 8087}
        )
    """
    announcer = ServiceAnnouncer(
        slug=slug,
        name=name,
        url=url,
        port=port,
        tier=tier,
        health_check=health_check,
        nats_url=nats_url,
        metadata=metadata,
    )
    return await announcer.announce()


class BackgroundAnnouncer:
    """
    Background service announcer that announces periodically.

    Useful for services that want to periodically re-announce themselves.
    """

    def __init__(
        self,
        announcer: ServiceAnnouncer,
        interval: float = 60.0,
    ):
        """
        Initialize background announcer.

        Args:
            announcer: Service announcer to use
            interval: Announcement interval in seconds
        """
        self.announcer = announcer
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def _announce_loop(self):
        """Internal announcement loop."""
        while self._running:
            await self.announcer.announce()
            await asyncio.sleep(self.interval)

    async def start(self):
        """Start background announcements."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._announce_loop())
            # Initial announcement
            await self.announcer.announce()

    async def stop(self):
        """Stop background announcements."""
        if self._running:
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass


# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Example usage of service announcer."""

        # Example 1: Simple announcement
        await announce_service(
            slug="example-service",
            name="Example Service",
            url="http://localhost:8080",
            port=8080,
            tier="api",
        )
        print("Service announced!")

        # Example 2: With metadata
        await announce_service(
            slug="hirag-v2",
            name="Hi-RAG Gateway v2",
            url="http://hi-rag-gateway-v2:8086",
            port=8086,
            tier="api",
            metadata={
                "gpu_port": 8087,
                "features": ["vector", "graph", "fulltext"],
                "rerank_enabled": True,
            },
        )

        # Example 3: Background announcer
        announcer = ServiceAnnouncer(
            slug="bg-service",
            name="Background Service",
            url="http://localhost:8081",
            port=8081,
            tier="worker",
        )
        bg = BackgroundAnnouncer(announcer, interval=30)
        await bg.start()
        print("Background announcer started (30s interval)")

        # Keep running...
        await asyncio.sleep(10)
        await bg.stop()

    asyncio.run(main())
