"""In-memory cache backend using numpy for similarity computation."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from semanticache.backends import BaseBackend

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _CacheEntry:
    """Internal representation of a single cached item."""

    key: str
    embedding: np.ndarray
    response: str
    metadata: dict[str, Any]
    cached_at: datetime
    ttl: int


class InMemoryBackend(BaseBackend):
    """Thread-safe in-memory cache backend.

    Stores embeddings and responses in a dictionary keyed by namespace.
    Cosine similarity is computed with numpy.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, _CacheEntry]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _is_expired(self, entry: _CacheEntry) -> bool:
        """Check whether an entry has exceeded its TTL."""
        age = (datetime.now(timezone.utc) - entry.cached_at).total_seconds()
        return age > entry.ttl

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    async def store(
        self,
        embedding: np.ndarray,
        response: str,
        namespace: str,
        metadata: dict[str, Any],
        ttl: int,
    ) -> str:
        """Store an embedding/response pair in memory.

        Args:
            embedding: The prompt embedding vector.
            response: The LLM response text.
            namespace: Logical partition key.
            metadata: Arbitrary metadata dictionary.
            ttl: Time-to-live in seconds.

        Returns:
            A unique key identifying the stored entry.
        """
        key = uuid.uuid4().hex
        entry = _CacheEntry(
            key=key,
            embedding=embedding,
            response=response,
            metadata=metadata,
            cached_at=datetime.now(timezone.utc),
            ttl=ttl,
        )
        async with self._lock:
            self._data.setdefault(namespace, {})[key] = entry
        logger.debug("Stored key %s in namespace %r", key, namespace)
        return key

    async def search(
        self,
        embedding: np.ndarray,
        namespace: str,
        threshold: float,
        ttl: int,
    ) -> tuple[str, float, datetime, dict[str, Any]] | None:
        """Find the best matching cached entry in *namespace*.

        Iterates over all entries, computes cosine similarity, skips expired
        entries, and returns the best match above *threshold*.

        Args:
            embedding: The query embedding vector.
            namespace: Logical partition to search within.
            threshold: Minimum cosine similarity to accept.
            ttl: Maximum age in seconds (used only for expiration check).

        Returns:
            (response, similarity, cached_at, metadata) or None.
        """
        async with self._lock:
            ns_data = self._data.get(namespace)
            if not ns_data:
                return None

            best_score: float = -1.0
            best_entry: _CacheEntry | None = None
            expired_keys: list[str] = []

            for key, entry in ns_data.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
                    continue
                score = self._cosine_similarity(embedding, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            # Evict expired entries lazily
            for key in expired_keys:
                del ns_data[key]

        if best_entry is None or best_score < threshold:
            return None

        return (
            best_entry.response,
            best_score,
            best_entry.cached_at,
            best_entry.metadata,
        )

    async def delete(self, key: str, namespace: str) -> bool:
        """Delete a single cached entry by key.

        Args:
            key: The entry key.
            namespace: The namespace the entry belongs to.

        Returns:
            True if the entry was found and deleted.
        """
        async with self._lock:
            ns_data = self._data.get(namespace)
            if ns_data is None or key not in ns_data:
                return False
            del ns_data[key]
            return True

    async def clear(self, namespace: str | None = None) -> int:
        """Clear entries from the cache.

        Args:
            namespace: If provided, clear only this namespace. Otherwise clear all.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            if namespace is not None:
                ns_data = self._data.pop(namespace, {})
                return len(ns_data)
            total = sum(len(v) for v in self._data.values())
            self._data.clear()
            return total

    async def size(self, namespace: str | None = None) -> int:
        """Return the number of cached entries.

        Args:
            namespace: If given, count only this namespace.

        Returns:
            Entry count.
        """
        async with self._lock:
            if namespace is not None:
                return len(self._data.get(namespace, {}))
            return sum(len(v) for v in self._data.values())
