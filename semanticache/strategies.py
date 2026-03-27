"""Advanced cache strategies — LRU eviction, frequency scoring, namespaces, batch ops."""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LRU eviction wrapper
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _LRUEntry:
    """Metadata for an LRU-tracked cache entry."""

    key: str
    namespace: str
    access_count: int = 0
    last_access: float = field(default_factory=time.monotonic)


class LRUEvictionPolicy:
    """Least-recently-used eviction policy with optional frequency weighting.

    Wraps around any backend to enforce a maximum number of cached entries.
    When the limit is exceeded, the least-recently-used entry (optionally
    weighted by access frequency) is evicted.

    Args:
        max_entries: Maximum number of entries across all namespaces.
        frequency_weight: Weight given to access frequency when scoring
            eviction candidates.  ``0.0`` = pure LRU, ``1.0`` = frequency
            strongly influences retention.
    """

    def __init__(
        self,
        max_entries: int = 10_000,
        frequency_weight: float = 0.0,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if not 0.0 <= frequency_weight <= 1.0:
            raise ValueError("frequency_weight must be between 0.0 and 1.0")
        self._max_entries = max_entries
        self._frequency_weight = frequency_weight
        # key -> _LRUEntry, ordered by access time
        self._entries: OrderedDict[str, _LRUEntry] = OrderedDict()

    @property
    def max_entries(self) -> int:
        """Maximum number of entries."""
        return self._max_entries

    @property
    def current_size(self) -> int:
        """Current number of tracked entries."""
        return len(self._entries)

    def record_access(self, key: str, namespace: str = "default") -> None:
        """Record an access (hit or insert) for *key*.

        Args:
            key: The cache entry key.
            namespace: Namespace of the entry.
        """
        if key in self._entries:
            entry = self._entries[key]
            entry.access_count += 1
            entry.last_access = time.monotonic()
            self._entries.move_to_end(key)
        else:
            self._entries[key] = _LRUEntry(
                key=key,
                namespace=namespace,
                access_count=1,
                last_access=time.monotonic(),
            )

    def eviction_candidates(self, count: int = 1) -> list[_LRUEntry]:
        """Return the *count* entries most eligible for eviction.

        With ``frequency_weight == 0`` this is pure LRU.  Higher weights
        protect frequently-accessed entries.

        Args:
            count: Number of candidates to return.

        Returns:
            List of ``_LRUEntry`` objects (oldest/least-scored first).
        """
        if self._frequency_weight == 0.0:
            # Pure LRU — take from the front of the OrderedDict
            return [entry for _, entry in zip(range(count), self._entries.values())]

        # Score = (1 - w) * recency_rank + w * frequency_rank
        # Lower score → more eligible for eviction
        entries = list(self._entries.values())
        now = time.monotonic()
        max_freq = max((e.access_count for e in entries), default=1)

        def _score(entry: _LRUEntry) -> float:
            recency = entry.last_access - now  # negative, more negative = older
            freq_norm = entry.access_count / max_freq if max_freq > 0 else 0
            return (1.0 - self._frequency_weight) * recency + self._frequency_weight * freq_norm

        entries.sort(key=_score)
        return entries[:count]

    def should_evict(self) -> bool:
        """Return ``True`` if the cache exceeds ``max_entries``."""
        return len(self._entries) > self._max_entries

    def remove(self, key: str) -> None:
        """Remove *key* from LRU tracking."""
        self._entries.pop(key, None)

    def get_keys_to_evict(self) -> list[str]:
        """Return keys that should be evicted to bring size within limits.

        Returns:
            List of cache keys to evict.
        """
        overflow = len(self._entries) - self._max_entries
        if overflow <= 0:
            return []
        candidates = self.eviction_candidates(overflow)
        return [c.key for c in candidates]

    def reset(self) -> None:
        """Clear all LRU tracking state."""
        self._entries.clear()


# ---------------------------------------------------------------------------
# Namespace manager
# ---------------------------------------------------------------------------


class NamespaceManager:
    """Manage isolated cache namespaces for multi-tenant or multi-environment use.

    Args:
        allowed_namespaces: If set, only these namespaces are permitted.
            ``None`` means any namespace is allowed.
    """

    def __init__(self, allowed_namespaces: set[str] | None = None) -> None:
        self._allowed = allowed_namespaces
        self._active: set[str] = set()

    def validate(self, namespace: str) -> None:
        """Validate that *namespace* is permitted.

        Args:
            namespace: Namespace to validate.

        Raises:
            ValueError: If the namespace is not in the allow-list.
        """
        if self._allowed is not None and namespace not in self._allowed:
            raise ValueError(
                f"Namespace {namespace!r} is not allowed. Allowed: {sorted(self._allowed)}"
            )

    def register(self, namespace: str) -> None:
        """Register *namespace* as active.

        Args:
            namespace: Namespace to register.
        """
        self.validate(namespace)
        self._active.add(namespace)

    def unregister(self, namespace: str) -> None:
        """Remove *namespace* from active set.

        Args:
            namespace: Namespace to unregister.
        """
        self._active.discard(namespace)

    @property
    def active_namespaces(self) -> set[str]:
        """Currently active namespaces."""
        return set(self._active)

    @property
    def allowed_namespaces(self) -> set[str] | None:
        """Configured allow-list, or ``None`` for unrestricted."""
        return set(self._allowed) if self._allowed is not None else None


# ---------------------------------------------------------------------------
# Batch caching
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BatchCacheItem:
    """A single item for batch caching.

    Attributes:
        prompt: The prompt text.
        response: The LLM response text.
        namespace: Target namespace.
        metadata: Optional metadata dict.
    """

    prompt: str
    response: str
    namespace: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


async def batch_cache_put(
    cache: Any,
    items: list[BatchCacheItem],
) -> int:
    """Store multiple prompt/response pairs in one call.

    Args:
        cache: A ``SemantiCache`` instance.
        items: List of ``BatchCacheItem`` to store.

    Returns:
        Number of items successfully cached.
    """
    count = 0
    for item in items:
        try:
            await cache.put(
                prompt=item.prompt,
                response=item.response,
                namespace=item.namespace,
                metadata=item.metadata,
            )
            count += 1
        except Exception:
            logger.exception("Failed to cache item: %r", item.prompt[:80])
    return count


# ---------------------------------------------------------------------------
# Cache warming
# ---------------------------------------------------------------------------


def load_warm_data_from_json(path: str) -> list[BatchCacheItem]:
    """Load cache warm-up data from a JSON file.

    Expected format::

        [
            {"prompt": "...", "response": "...", "namespace": "default", "metadata": {}},
            ...
        ]

    ``namespace`` and ``metadata`` are optional.

    Args:
        path: Path to the JSON file.

    Returns:
        List of ``BatchCacheItem`` objects.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Warm-up JSON must be an array of objects")

    items: list[BatchCacheItem] = []
    for entry in data:
        items.append(
            BatchCacheItem(
                prompt=entry["prompt"],
                response=entry["response"],
                namespace=entry.get("namespace", "default"),
                metadata=entry.get("metadata", {}),
            )
        )
    return items


async def warm_cache(cache: Any, path: str) -> int:
    """Load warm-up data from *path* and populate the cache.

    Args:
        cache: A ``SemantiCache`` instance.
        path: Path to a JSON file with warm-up data.

    Returns:
        Number of items loaded.
    """
    items = load_warm_data_from_json(path)
    return await batch_cache_put(cache, items)
