"""Cache storage backends for SemantiCache."""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Any

import numpy as np


class BaseBackend(abc.ABC):
    """Abstract base class for cache storage backends.

    All backends must implement the five core operations:
    store, search, delete, clear, and size.
    """

    @abc.abstractmethod
    async def store(
        self,
        embedding: np.ndarray,
        response: str,
        namespace: str,
        metadata: dict[str, Any],
        ttl: int,
    ) -> str:
        """Store an embedding/response pair.

        Args:
            embedding: The prompt embedding vector.
            response: The LLM response text.
            namespace: Logical partition key.
            metadata: Arbitrary metadata dictionary.
            ttl: Time-to-live in seconds.

        Returns:
            A unique key identifying the stored entry.
        """
        ...

    @abc.abstractmethod
    async def search(
        self,
        embedding: np.ndarray,
        namespace: str,
        threshold: float,
        ttl: int,
    ) -> tuple[str, float, datetime, dict[str, Any]] | None:
        """Find the most similar cached entry above *threshold*.

        Args:
            embedding: The query embedding vector.
            namespace: Logical partition to search within.
            threshold: Minimum cosine similarity to accept.
            ttl: Maximum age in seconds for valid entries.

        Returns:
            A tuple of (response, similarity, cached_at, metadata) for
            the best match, or None if no match exceeds the threshold.
        """
        ...

    @abc.abstractmethod
    async def delete(self, key: str, namespace: str) -> bool:
        """Delete a single cached entry by key.

        Args:
            key: The entry key returned by store().
            namespace: The namespace the entry belongs to.

        Returns:
            True if the entry was deleted, False if it was not found.
        """
        ...

    @abc.abstractmethod
    async def clear(self, namespace: str | None = None) -> int:
        """Remove cached entries.

        Args:
            namespace: If given, only clear this namespace. If None, clear all.

        Returns:
            The number of entries removed.
        """
        ...

    @abc.abstractmethod
    async def size(self, namespace: str | None = None) -> int:
        """Return the number of cached entries.

        Args:
            namespace: If given, count only this namespace. If None, count all.

        Returns:
            Entry count.
        """
        ...


__all__ = ["BaseBackend"]
