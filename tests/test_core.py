"""Tests for SemantiCache core module."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from semanticache.core import CacheResult, SemantiCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_embedder(vectors: dict[str, np.ndarray] | None = None):
    """Create a mock embedder that returns predetermined vectors.

    If *vectors* is provided it maps prompt text -> embedding.
    Otherwise every call returns a random 384-dim unit vector.
    """
    embedder = AsyncMock()

    if vectors is not None:

        async def _embed(text: str) -> np.ndarray:
            if text in vectors:
                return vectors[text]
            # Return a random vector for unknown prompts
            rng = np.random.default_rng(hash(text) % 2**32)
            v = rng.standard_normal(384).astype(np.float32)
            return v / np.linalg.norm(v)

        embedder.embed = AsyncMock(side_effect=_embed)
    else:

        async def _embed_random(text: str) -> np.ndarray:
            rng = np.random.default_rng(hash(text) % 2**32)
            v = rng.standard_normal(384).astype(np.float32)
            return v / np.linalg.norm(v)

        embedder.embed = AsyncMock(side_effect=_embed_random)

    return embedder


@pytest.fixture
def fixed_vector():
    """A deterministic unit vector of dimension 384."""
    v = np.ones(384, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def similar_vector(fixed_vector):
    """A vector very close to *fixed_vector* (cosine sim > 0.99)."""
    noise = np.random.default_rng(42).standard_normal(384).astype(np.float32) * 0.01
    v = fixed_vector + noise
    return v / np.linalg.norm(v)


@pytest.fixture
def dissimilar_vector():
    """A vector orthogonal-ish to the fixed vector."""
    v = np.zeros(384, dtype=np.float32)
    v[0] = 1.0
    v[1] = -1.0
    rng = np.random.default_rng(99)
    v += rng.standard_normal(384).astype(np.float32) * 0.5
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# CacheResult dataclass
# ---------------------------------------------------------------------------


class TestCacheResult:
    def test_creation(self):
        result = CacheResult(
            response="hello",
            hit=True,
            similarity_score=0.95,
            latency_ms=1.2,
        )
        assert result.response == "hello"
        assert result.hit is True
        assert result.similarity_score == 0.95
        assert result.latency_ms == 1.2
        assert result.cached_at is None

    def test_creation_with_cached_at(self):
        now = datetime.now(timezone.utc)
        result = CacheResult(
            response="world",
            hit=True,
            similarity_score=0.99,
            latency_ms=0.5,
            cached_at=now,
        )
        assert result.cached_at == now

    def test_frozen(self):
        result = CacheResult(response="x", hit=False, similarity_score=0.0, latency_ms=0.0)
        with pytest.raises(AttributeError):
            result.response = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SemantiCache initialisation
# ---------------------------------------------------------------------------


class TestSemantiCacheInit:
    def test_defaults(self):
        """Initialise with explicit mock backend and embedder to avoid importing
        sentence-transformers."""
        from semanticache.backends.memory import InMemoryBackend

        embedder = _make_embedder()
        backend = InMemoryBackend()
        cache = SemantiCache(backend=backend, embedder=embedder)

        assert cache._similarity_threshold == 0.92
        assert cache._ttl == 86400
        assert cache._metrics is not None

    def test_custom_params(self):
        from semanticache.backends.memory import InMemoryBackend

        embedder = _make_embedder()
        backend = InMemoryBackend()
        cache = SemantiCache(
            backend=backend,
            embedder=embedder,
            similarity_threshold=0.85,
            ttl=3600,
            metrics_enabled=False,
        )

        assert cache._similarity_threshold == 0.85
        assert cache._ttl == 3600
        assert cache._metrics is None


# ---------------------------------------------------------------------------
# Put and Get
# ---------------------------------------------------------------------------


class TestPutAndGet:
    @pytest.fixture
    def cache_and_embedder(self, fixed_vector, similar_vector, dissimilar_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {
            "What is Python?": fixed_vector,
            "Tell me about Python": similar_vector,
            "Recipe for chocolate cake": dissimilar_vector,
        }
        embedder = _make_embedder(vectors)
        backend = InMemoryBackend()
        cache = SemantiCache(
            backend=backend,
            embedder=embedder,
            similarity_threshold=0.90,
            ttl=3600,
        )
        return cache, embedder

    async def test_put_and_get_exact(self, cache_and_embedder):
        cache, _ = cache_and_embedder

        await cache.put("What is Python?", "Python is a programming language.")

        result = await cache.get("What is Python?")
        assert result is not None
        assert result.hit is True
        assert result.response == "Python is a programming language."
        assert result.similarity_score >= 0.99

    async def test_get_returns_none_when_empty(self, cache_and_embedder):
        cache, _ = cache_and_embedder
        result = await cache.get("What is Python?")
        assert result is None

    async def test_cache_hit_similar_prompt(self, cache_and_embedder):
        cache, _ = cache_and_embedder

        await cache.put("What is Python?", "Python is a programming language.")

        result = await cache.get("Tell me about Python")
        assert result is not None
        assert result.hit is True
        assert result.similarity_score > 0.90

    async def test_cache_miss_dissimilar_prompt(self, cache_and_embedder):
        cache, _ = cache_and_embedder

        await cache.put("What is Python?", "Python is a programming language.")

        result = await cache.get("Recipe for chocolate cake")
        assert result is None


# ---------------------------------------------------------------------------
# cache() method (with generator)
# ---------------------------------------------------------------------------


class TestCacheMethod:
    async def test_miss_calls_generator(self, fixed_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {"prompt": fixed_vector}
        embedder = _make_embedder(vectors)
        cache = SemantiCache(
            backend=InMemoryBackend(), embedder=embedder, similarity_threshold=0.90
        )

        generator = MagicMock(return_value="generated response")

        result = await cache.cache(prompt="prompt", generator=generator)

        assert result.hit is False
        assert result.response == "generated response"
        assert result.similarity_score == 0.0
        generator.assert_called_once()

    async def test_hit_skips_generator(self, fixed_vector, similar_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {
            "prompt A": fixed_vector,
            "prompt B": similar_vector,
        }
        embedder = _make_embedder(vectors)
        cache = SemantiCache(
            backend=InMemoryBackend(), embedder=embedder, similarity_threshold=0.90
        )

        # Populate cache
        await cache.put("prompt A", "cached response")

        generator = MagicMock(return_value="should not be called")

        result = await cache.cache(prompt="prompt B", generator=generator)

        assert result.hit is True
        assert result.response == "cached response"
        generator.assert_not_called()

    async def test_async_generator(self, fixed_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {"prompt": fixed_vector}
        embedder = _make_embedder(vectors)
        cache = SemantiCache(
            backend=InMemoryBackend(), embedder=embedder, similarity_threshold=0.90
        )

        async def async_gen():
            return "async generated"

        result = await cache.cache(prompt="prompt", generator=async_gen)
        assert result.response == "async generated"


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------


class TestTTLExpiration:
    async def test_expired_entry_not_returned(self, fixed_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {"prompt": fixed_vector}
        embedder = _make_embedder(vectors)
        cache = SemantiCache(
            backend=InMemoryBackend(),
            embedder=embedder,
            similarity_threshold=0.90,
            ttl=1,  # 1 second TTL
        )

        await cache.put("prompt", "response")

        # Wait for expiration
        await asyncio.sleep(1.1)

        result = await cache.get("prompt")
        assert result is None


# ---------------------------------------------------------------------------
# Clear cache
# ---------------------------------------------------------------------------


class TestClearCache:
    async def test_clear_all(self, fixed_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {"p1": fixed_vector}
        embedder = _make_embedder(vectors)
        backend = InMemoryBackend()
        cache = SemantiCache(backend=backend, embedder=embedder)

        await cache.put("p1", "r1")
        assert await backend.size() == 1

        count = await cache.clear()
        assert count == 1
        assert await backend.size() == 0

    async def test_clear_namespace(self):
        from semanticache.backends.memory import InMemoryBackend

        embedder = _make_embedder()
        backend = InMemoryBackend()
        cache = SemantiCache(backend=backend, embedder=embedder)

        await cache.put("p1", "r1", namespace="ns1")
        await cache.put("p2", "r2", namespace="ns2")

        assert await backend.size() == 2

        count = await cache.clear(namespace="ns1")
        assert count == 1
        assert await backend.size(namespace="ns1") == 0
        assert await backend.size(namespace="ns2") == 1


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------


class TestMetrics:
    async def test_metrics_record_miss(self, fixed_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {"prompt": fixed_vector}
        embedder = _make_embedder(vectors)
        cache = SemantiCache(backend=InMemoryBackend(), embedder=embedder)

        await cache.cache(prompt="prompt", generator=lambda: "resp")

        # Access the tracker directly to avoid potential lock issues in to_dict
        assert cache._metrics is not None
        assert cache._metrics.cache_misses >= 1

    async def test_metrics_record_hit(self, fixed_vector, similar_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {"p1": fixed_vector, "p2": similar_vector}
        embedder = _make_embedder(vectors)
        cache = SemantiCache(
            backend=InMemoryBackend(), embedder=embedder, similarity_threshold=0.90
        )

        await cache.put("p1", "resp")
        await cache.get("p2")

        assert cache._metrics is not None
        assert cache._metrics.cache_hits >= 1

    async def test_metrics_disabled(self, fixed_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {"prompt": fixed_vector}
        embedder = _make_embedder(vectors)
        cache = SemantiCache(
            backend=InMemoryBackend(),
            embedder=embedder,
            metrics_enabled=False,
        )

        assert cache.get_metrics() == {}

    async def test_hit_rate(self, fixed_vector, similar_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {"p1": fixed_vector, "p2": similar_vector}
        embedder = _make_embedder(vectors)
        cache = SemantiCache(
            backend=InMemoryBackend(), embedder=embedder, similarity_threshold=0.90
        )

        # 1 miss
        await cache.cache(prompt="p1", generator=lambda: "resp")
        # 1 hit
        await cache.cache(prompt="p2", generator=lambda: "nope")

        assert cache._metrics is not None
        assert cache._metrics.hit_rate == pytest.approx(0.5, abs=0.01)

    async def test_total_requests(self, fixed_vector, similar_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {"p1": fixed_vector, "p2": similar_vector}
        embedder = _make_embedder(vectors)
        cache = SemantiCache(
            backend=InMemoryBackend(), embedder=embedder, similarity_threshold=0.90
        )

        await cache.cache(prompt="p1", generator=lambda: "resp")
        await cache.cache(prompt="p2", generator=lambda: "nope")

        assert cache._metrics is not None
        assert cache._metrics.total_requests == 2


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------


class TestNamespaceIsolation:
    async def test_namespaces_are_isolated(self, fixed_vector):
        from semanticache.backends.memory import InMemoryBackend

        vectors = {"prompt": fixed_vector}
        embedder = _make_embedder(vectors)
        cache = SemantiCache(
            backend=InMemoryBackend(), embedder=embedder, similarity_threshold=0.90
        )

        await cache.put("prompt", "response A", namespace="ns_a")

        # Same prompt but different namespace should miss
        result = await cache.get("prompt", namespace="ns_b")
        assert result is None

        # Same namespace should hit
        result = await cache.get("prompt", namespace="ns_a")
        assert result is not None
        assert result.response == "response A"
