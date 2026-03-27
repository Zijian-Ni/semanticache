"""Redis-backed cache backend using redis.asyncio."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from semanticache.backends import BaseBackend

logger = logging.getLogger(__name__)

# Redis key layout per entry (all in a namespace hash-set):
#   semanticache:{namespace}:emb:{key}   -> embedding bytes
#   semanticache:{namespace}:data:{key}   -> JSON {response, metadata, cached_at}
# A Redis SET tracks keys per namespace:
#   semanticache:{namespace}:keys         -> set of key strings

_PREFIX = "semanticache"


class RedisBackend(BaseBackend):
    """Cache backend backed by Redis.

    Embeddings are stored as raw numpy bytes.  Responses and metadata are
    serialised as JSON.  TTL is enforced by Redis ``EXPIRE``.

    Args:
        url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
        redis_client: An existing ``redis.asyncio.Redis`` instance.  If
            provided, *url* is ignored.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        redis_client: Any | None = None,
    ) -> None:
        self._url = url
        self._client = redis_client
        self._owned_client = redis_client is None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    async def _get_client(self) -> Any:
        """Lazily create or return the Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as exc:
                raise ImportError(
                    "redis[asyncio] is required for RedisBackend. "
                    "Install it with: pip install redis[asyncio]"
                ) from exc
            self._client = aioredis.from_url(self._url, decode_responses=False)
        return self._client

    async def close(self) -> None:
        """Close the underlying Redis connection if we own it."""
        if self._client is not None and self._owned_client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emb_key(namespace: str, key: str) -> str:
        return f"{_PREFIX}:{namespace}:emb:{key}"

    @staticmethod
    def _data_key(namespace: str, key: str) -> str:
        return f"{_PREFIX}:{namespace}:data:{key}"

    @staticmethod
    def _keys_set(namespace: str) -> str:
        return f"{_PREFIX}:{namespace}:keys"

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

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
        """Store an entry in Redis.

        Args:
            embedding: The prompt embedding vector.
            response: The LLM response text.
            namespace: Logical partition key.
            metadata: Arbitrary metadata dictionary.
            ttl: Time-to-live in seconds.

        Returns:
            A unique key identifying the stored entry.
        """
        client = await self._get_client()
        key = uuid.uuid4().hex

        data_payload = json.dumps(
            {
                "response": response,
                "metadata": metadata,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
        ).encode()

        emb_bytes = embedding.astype(np.float32).tobytes()

        pipe = client.pipeline(transaction=True)
        pipe.set(self._emb_key(namespace, key), emb_bytes, ex=ttl)
        pipe.set(self._data_key(namespace, key), data_payload, ex=ttl)
        pipe.sadd(self._keys_set(namespace), key)
        await pipe.execute()

        logger.debug("Stored key %s in Redis namespace %r (ttl=%ds)", key, namespace, ttl)
        return key

    async def search(
        self,
        embedding: np.ndarray,
        namespace: str,
        threshold: float,
        ttl: int,
    ) -> tuple[str, float, datetime, dict[str, Any]] | None:
        """Search Redis for the best matching cached entry.

        Fetches all keys in the namespace, loads their embeddings, and
        computes cosine similarity.

        Args:
            embedding: The query embedding vector.
            namespace: Logical partition to search within.
            threshold: Minimum cosine similarity to accept.
            ttl: Unused (Redis handles TTL via EXPIRE).

        Returns:
            (response, similarity, cached_at, metadata) or None.
        """
        client = await self._get_client()
        members = await client.smembers(self._keys_set(namespace))

        if not members:
            return None

        best_score: float = -1.0
        best_data: dict[str, Any] | None = None
        stale_keys: list[bytes | str] = []

        for member in members:
            key = member.decode() if isinstance(member, bytes) else member

            emb_bytes = await client.get(self._emb_key(namespace, key))
            data_bytes = await client.get(self._data_key(namespace, key))

            if emb_bytes is None or data_bytes is None:
                # Entry expired but key still in set
                stale_keys.append(member)
                continue

            stored_emb = np.frombuffer(emb_bytes, dtype=np.float32)
            score = self._cosine_similarity(embedding, stored_emb)

            if score > best_score:
                best_score = score
                best_data = json.loads(data_bytes)

        # Clean up stale keys
        if stale_keys:
            await client.srem(self._keys_set(namespace), *stale_keys)

        if best_data is None or best_score < threshold:
            return None

        cached_at = datetime.fromisoformat(best_data["cached_at"])
        return (
            best_data["response"],
            best_score,
            cached_at,
            best_data.get("metadata", {}),
        )

    async def delete(self, key: str, namespace: str) -> bool:
        """Delete a single cached entry from Redis.

        Args:
            key: The entry key.
            namespace: The namespace the entry belongs to.

        Returns:
            True if the entry was deleted.
        """
        client = await self._get_client()
        pipe = client.pipeline(transaction=True)
        pipe.delete(self._emb_key(namespace, key))
        pipe.delete(self._data_key(namespace, key))
        pipe.srem(self._keys_set(namespace), key)
        results = await pipe.execute()
        deleted = sum(int(r) for r in results[:2])
        return deleted > 0

    async def clear(self, namespace: str | None = None) -> int:
        """Clear entries from Redis.

        Args:
            namespace: If provided, clear only this namespace.

        Returns:
            Number of entries removed.
        """
        client = await self._get_client()

        if namespace is not None:
            members = await client.smembers(self._keys_set(namespace))
            if not members:
                return 0
            pipe = client.pipeline(transaction=True)
            for member in members:
                key = member.decode() if isinstance(member, bytes) else member
                pipe.delete(self._emb_key(namespace, key))
                pipe.delete(self._data_key(namespace, key))
            pipe.delete(self._keys_set(namespace))
            await pipe.execute()
            return len(members)

        # Clear all namespaces: scan for our prefix
        count = 0
        async for redis_key in client.scan_iter(match=f"{_PREFIX}:*"):
            await client.delete(redis_key)
            count += 1
        return count

    async def size(self, namespace: str | None = None) -> int:
        """Return the number of cached entries.

        Args:
            namespace: If given, count only this namespace.

        Returns:
            Entry count.
        """
        client = await self._get_client()

        if namespace is not None:
            return await client.scard(self._keys_set(namespace))

        # Sum across all namespace key-sets
        total = 0
        async for redis_key in client.scan_iter(match=f"{_PREFIX}:*:keys"):
            total += await client.scard(redis_key)
        return total
