"""Tests for cache storage backends."""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import pytest

from semanticache.backends.memory import InMemoryBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vector(dim: int = 384, seed: int = 0) -> np.ndarray:
    """Return a deterministic unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# InMemoryBackend
# ---------------------------------------------------------------------------


class TestInMemoryBackendStore:
    async def test_store_returns_key(self):
        backend = InMemoryBackend()
        key = await backend.store(
            embedding=_unit_vector(seed=1),
            response="hello",
            namespace="default",
            metadata={},
            ttl=3600,
        )
        assert isinstance(key, str)
        assert len(key) > 0

    async def test_store_increments_size(self):
        backend = InMemoryBackend()
        assert await backend.size() == 0

        await backend.store(
            embedding=_unit_vector(seed=1),
            response="r1",
            namespace="default",
            metadata={},
            ttl=3600,
        )
        assert await backend.size() == 1

        await backend.store(
            embedding=_unit_vector(seed=2),
            response="r2",
            namespace="default",
            metadata={},
            ttl=3600,
        )
        assert await backend.size() == 2


class TestInMemoryBackendSearch:
    async def test_search_empty_returns_none(self):
        backend = InMemoryBackend()
        result = await backend.search(
            embedding=_unit_vector(seed=1),
            namespace="default",
            threshold=0.90,
            ttl=3600,
        )
        assert result is None

    async def test_search_exact_match(self):
        backend = InMemoryBackend()
        vec = _unit_vector(seed=1)

        await backend.store(
            embedding=vec,
            response="exact match",
            namespace="default",
            metadata={"key": "value"},
            ttl=3600,
        )

        result = await backend.search(
            embedding=vec,
            namespace="default",
            threshold=0.90,
            ttl=3600,
        )

        assert result is not None
        response, similarity, cached_at, metadata = result
        assert response == "exact match"
        assert similarity >= 0.999
        assert isinstance(cached_at, datetime)
        assert metadata == {"key": "value"}

    async def test_search_similar_vector(self):
        backend = InMemoryBackend()
        vec = _unit_vector(seed=1)

        # Create a similar vector with small noise
        noise = np.random.default_rng(42).standard_normal(384).astype(np.float32) * 0.01
        similar_vec = vec + noise
        similar_vec = similar_vec / np.linalg.norm(similar_vec)

        await backend.store(
            embedding=vec,
            response="similar match",
            namespace="default",
            metadata={},
            ttl=3600,
        )

        result = await backend.search(
            embedding=similar_vec,
            namespace="default",
            threshold=0.90,
            ttl=3600,
        )

        assert result is not None
        response, similarity, _, _ = result
        assert response == "similar match"
        assert similarity > 0.90

    async def test_search_below_threshold_returns_none(self):
        backend = InMemoryBackend()

        await backend.store(
            embedding=_unit_vector(seed=1),
            response="stored",
            namespace="default",
            metadata={},
            ttl=3600,
        )

        # Use a very different vector
        result = await backend.search(
            embedding=_unit_vector(seed=999),
            namespace="default",
            threshold=0.99,
            ttl=3600,
        )
        # Similarity of two random unit vectors is unlikely to exceed 0.99
        assert result is None

    async def test_search_wrong_namespace_returns_none(self):
        backend = InMemoryBackend()
        vec = _unit_vector(seed=1)

        await backend.store(
            embedding=vec,
            response="in ns_a",
            namespace="ns_a",
            metadata={},
            ttl=3600,
        )

        result = await backend.search(
            embedding=vec,
            namespace="ns_b",
            threshold=0.90,
            ttl=3600,
        )
        assert result is None

    async def test_search_returns_best_match(self):
        backend = InMemoryBackend()
        query = _unit_vector(seed=1)

        # Store two entries: one close, one far
        close_vec = query + np.random.default_rng(10).standard_normal(384).astype(np.float32) * 0.01
        close_vec = close_vec / np.linalg.norm(close_vec)

        far_vec = _unit_vector(seed=500)

        await backend.store(
            embedding=far_vec, response="far", namespace="default", metadata={}, ttl=3600
        )
        await backend.store(
            embedding=close_vec, response="close", namespace="default", metadata={}, ttl=3600
        )

        result = await backend.search(
            embedding=query, namespace="default", threshold=0.50, ttl=3600
        )
        assert result is not None
        assert result[0] == "close"


class TestInMemoryBackendTTL:
    async def test_expired_entries_not_returned(self):
        backend = InMemoryBackend()
        vec = _unit_vector(seed=1)

        await backend.store(
            embedding=vec,
            response="will expire",
            namespace="default",
            metadata={},
            ttl=1,  # 1 second
        )

        await asyncio.sleep(1.1)

        result = await backend.search(embedding=vec, namespace="default", threshold=0.50, ttl=1)
        assert result is None

    async def test_non_expired_entries_returned(self):
        backend = InMemoryBackend()
        vec = _unit_vector(seed=1)

        await backend.store(
            embedding=vec,
            response="still valid",
            namespace="default",
            metadata={},
            ttl=3600,
        )

        result = await backend.search(embedding=vec, namespace="default", threshold=0.50, ttl=3600)
        assert result is not None
        assert result[0] == "still valid"


class TestInMemoryBackendClear:
    async def test_clear_all(self):
        backend = InMemoryBackend()

        await backend.store(
            embedding=_unit_vector(seed=1),
            response="r1",
            namespace="ns1",
            metadata={},
            ttl=3600,
        )
        await backend.store(
            embedding=_unit_vector(seed=2),
            response="r2",
            namespace="ns2",
            metadata={},
            ttl=3600,
        )

        count = await backend.clear()
        assert count == 2
        assert await backend.size() == 0

    async def test_clear_namespace(self):
        backend = InMemoryBackend()

        await backend.store(
            embedding=_unit_vector(seed=1),
            response="r1",
            namespace="ns1",
            metadata={},
            ttl=3600,
        )
        await backend.store(
            embedding=_unit_vector(seed=2),
            response="r2",
            namespace="ns2",
            metadata={},
            ttl=3600,
        )

        count = await backend.clear(namespace="ns1")
        assert count == 1
        assert await backend.size(namespace="ns1") == 0
        assert await backend.size(namespace="ns2") == 1

    async def test_clear_empty_namespace(self):
        backend = InMemoryBackend()
        count = await backend.clear(namespace="nonexistent")
        assert count == 0


class TestInMemoryBackendSize:
    async def test_size_empty(self):
        backend = InMemoryBackend()
        assert await backend.size() == 0

    async def test_size_all_namespaces(self):
        backend = InMemoryBackend()

        await backend.store(
            embedding=_unit_vector(seed=1),
            response="r1",
            namespace="ns1",
            metadata={},
            ttl=3600,
        )
        await backend.store(
            embedding=_unit_vector(seed=2),
            response="r2",
            namespace="ns2",
            metadata={},
            ttl=3600,
        )

        assert await backend.size() == 2

    async def test_size_specific_namespace(self):
        backend = InMemoryBackend()

        await backend.store(
            embedding=_unit_vector(seed=1),
            response="r1",
            namespace="ns1",
            metadata={},
            ttl=3600,
        )
        await backend.store(
            embedding=_unit_vector(seed=2),
            response="r2",
            namespace="ns2",
            metadata={},
            ttl=3600,
        )

        assert await backend.size(namespace="ns1") == 1
        assert await backend.size(namespace="ns2") == 1
        assert await backend.size(namespace="ns3") == 0


class TestInMemoryBackendDelete:
    async def test_delete_existing(self):
        backend = InMemoryBackend()

        key = await backend.store(
            embedding=_unit_vector(seed=1),
            response="r1",
            namespace="default",
            metadata={},
            ttl=3600,
        )

        assert await backend.delete(key, "default") is True
        assert await backend.size() == 0

    async def test_delete_nonexistent(self):
        backend = InMemoryBackend()
        assert await backend.delete("no-such-key", "default") is False


# ---------------------------------------------------------------------------
# Redis backend (skipped unless Redis is available)
# ---------------------------------------------------------------------------


def _redis_available() -> bool:
    """Check whether a Redis server is reachable on localhost."""
    try:
        import redis

        client = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        client.ping()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _redis_available(), reason="Redis server not available")
class TestRedisBackend:
    """Integration tests for RedisBackend; require a running Redis instance."""

    @pytest.fixture
    async def backend(self):
        from semanticache.backends.redis import RedisBackend

        be = RedisBackend(url="redis://localhost:6379/15")  # Use DB 15 for tests
        yield be
        await be.clear()
        await be.close()

    async def test_store_and_search(self, backend):
        vec = _unit_vector(seed=1)

        await backend.store(
            embedding=vec,
            response="redis response",
            namespace="test",
            metadata={"m": 1},
            ttl=60,
        )

        result = await backend.search(
            embedding=vec,
            namespace="test",
            threshold=0.90,
            ttl=60,
        )
        assert result is not None
        assert result[0] == "redis response"
        assert result[1] >= 0.99

    async def test_clear(self, backend):
        await backend.store(
            embedding=_unit_vector(seed=1),
            response="r",
            namespace="test",
            metadata={},
            ttl=60,
        )
        count = await backend.clear(namespace="test")
        assert count >= 1
        assert await backend.size(namespace="test") == 0

    async def test_delete(self, backend):
        key = await backend.store(
            embedding=_unit_vector(seed=1),
            response="r",
            namespace="test",
            metadata={},
            ttl=60,
        )
        assert await backend.delete(key, "test") is True
