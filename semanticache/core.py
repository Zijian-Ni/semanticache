"""Core SemantiCache class providing semantic caching for LLM responses."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from semanticache.backends import BaseBackend
from semanticache.embedders import BaseEmbedder
from semanticache.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CacheResult:
    """Result returned from a cache lookup or generation.

    Attributes:
        response: The LLM response text.
        hit: Whether the result came from cache.
        similarity_score: Cosine similarity score (0.0 for misses).
        latency_ms: Time taken to retrieve or generate the response.
        cached_at: Timestamp when the entry was originally cached, or None for misses.
    """

    response: str
    hit: bool
    similarity_score: float
    latency_ms: float
    cached_at: datetime | None = field(default=None)


class SemantiCache:
    """Semantic caching layer for LLM API calls.

    Caches LLM responses and retrieves them when a semantically similar
    prompt is encountered, reducing latency and cost.

    Args:
        backend: Storage backend for cached entries. Defaults to InMemoryBackend.
        embedder: Embedding provider for computing prompt vectors. Defaults to
            SentenceTransformerEmbedder.
        similarity_threshold: Minimum cosine similarity (0-1) to consider a
            cache hit. Higher values require closer matches.
        ttl: Time-to-live in seconds for cached entries.
        metrics_enabled: Whether to collect usage metrics.
    """

    def __init__(
        self,
        backend: BaseBackend | None = None,
        embedder: BaseEmbedder | None = None,
        similarity_threshold: float = 0.92,
        ttl: int = 86400,
        metrics_enabled: bool = True,
    ) -> None:
        if backend is None:
            from semanticache.backends.memory import InMemoryBackend

            backend = InMemoryBackend()
        if embedder is None:
            from semanticache.embedders.sentence_transformers import (
                SentenceTransformerEmbedder,
            )

            embedder = SentenceTransformerEmbedder()

        self._backend = backend
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold
        self._ttl = ttl
        self._metrics = MetricsTracker() if metrics_enabled else None
        self._lock = asyncio.Lock()

        logger.info(
            "SemantiCache initialised (threshold=%.2f, ttl=%ds, metrics=%s)",
            similarity_threshold,
            ttl,
            metrics_enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def cache(
        self,
        prompt: str,
        generator: Callable[..., Any],
        namespace: str = "default",
    ) -> CacheResult:
        """Check cache first; on miss call *generator* and store the result.

        Args:
            prompt: The user prompt to look up.
            generator: An async or sync callable that produces the response
                string when the cache misses.  It receives no arguments.
            namespace: Logical partition for cached entries.

        Returns:
            A CacheResult indicating whether the response was served from
            cache or freshly generated.
        """
        cached = await self.get(prompt, namespace=namespace)
        if cached is not None:
            return cached

        start = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(generator):
                response_text: str = await generator()
            else:
                loop = asyncio.get_running_loop()
                response_text = await loop.run_in_executor(None, generator)
        except Exception:
            logger.exception("Generator failed for prompt in namespace %r", namespace)
            raise

        latency_ms = (time.perf_counter() - start) * 1000.0

        await self.put(prompt, response_text, namespace=namespace)

        if self._metrics is not None:
            self._metrics.record_miss()

        return CacheResult(
            response=response_text,
            hit=False,
            similarity_score=0.0,
            latency_ms=latency_ms,
            cached_at=None,
        )

    async def get(
        self,
        prompt: str,
        namespace: str = "default",
    ) -> CacheResult | None:
        """Look up a semantically similar prompt in the cache.

        Args:
            prompt: The prompt to search for.
            namespace: Logical partition to search within.

        Returns:
            A CacheResult on hit, or None on miss.
        """
        start = time.perf_counter()
        try:
            embedding = await self._embedder.embed(prompt)
        except Exception:
            logger.exception("Embedding failed for prompt in namespace %r", namespace)
            raise

        async with self._lock:
            result = await self._backend.search(
                embedding=embedding,
                namespace=namespace,
                threshold=self._similarity_threshold,
                ttl=self._ttl,
            )

        latency_ms = (time.perf_counter() - start) * 1000.0

        if result is None:
            logger.debug("Cache miss in namespace %r", namespace)
            return None

        response_text, similarity, cached_at, _metadata = result

        if self._metrics is not None:
            self._metrics.record_hit(similarity_score=similarity)

        logger.debug(
            "Cache hit in namespace %r (similarity=%.4f)", namespace, similarity
        )

        return CacheResult(
            response=response_text,
            hit=True,
            similarity_score=similarity,
            latency_ms=latency_ms,
            cached_at=cached_at,
        )

    async def put(
        self,
        prompt: str,
        response: str,
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a prompt/response pair in the cache.

        Args:
            prompt: The original prompt.
            response: The generated response.
            namespace: Logical partition for the entry.
            metadata: Arbitrary metadata to attach to the cache entry.
        """
        try:
            embedding = await self._embedder.embed(prompt)
        except Exception:
            logger.exception("Embedding failed during put for namespace %r", namespace)
            raise

        async with self._lock:
            await self._backend.store(
                embedding=embedding,
                response=response,
                namespace=namespace,
                metadata=metadata or {},
                ttl=self._ttl,
            )

        logger.debug("Stored entry in namespace %r", namespace)

    async def clear(self, namespace: str | None = None) -> int:
        """Clear cached entries.

        Args:
            namespace: If provided, only clear entries in this namespace.
                If None, clear all namespaces.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            count = await self._backend.clear(namespace=namespace)

        logger.info("Cleared %d entries (namespace=%r)", count, namespace)
        return count

    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of collected metrics.

        Returns:
            Dictionary of metric values, or an empty dict if metrics are
            disabled.
        """
        if self._metrics is None:
            return {}
        return self._metrics.to_dict()
