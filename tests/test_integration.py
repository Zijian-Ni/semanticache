"""Integration tests with mock OpenAI responses."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np

from semanticache.backends.memory import InMemoryBackend
from semanticache.core import SemantiCache
from semanticache.security import CacheEncryptor, sanitize_input
from semanticache.strategies import (
    BatchCacheItem,
    LRUEvictionPolicy,
    batch_cache_put,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedder(dim: int = 384) -> MagicMock:
    """Create a mock embedder that returns hash-based deterministic vectors."""
    embedder = MagicMock()

    async def _embed(text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) % 2**32)
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    embedder.embed = AsyncMock(side_effect=_embed)
    return embedder


def _make_similar_embedder(base_text: str, similar_texts: list[str], dim: int = 384) -> MagicMock:
    """Create an embedder where similar_texts return vectors close to base_text."""
    embedder = MagicMock()
    rng = np.random.default_rng(42)
    base_vec = rng.standard_normal(dim).astype(np.float32)
    base_vec = base_vec / np.linalg.norm(base_vec)

    vectors: dict[str, np.ndarray] = {base_text: base_vec}
    for text in similar_texts:
        noise = rng.standard_normal(dim).astype(np.float32) * 0.01
        v = base_vec + noise
        vectors[text] = v / np.linalg.norm(v)

    async def _embed(text: str) -> np.ndarray:
        if text in vectors:
            return vectors[text]
        v = np.random.default_rng(hash(text) % 2**32).standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    embedder.embed = AsyncMock(side_effect=_embed)
    return embedder


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    async def test_cache_miss_then_hit(self) -> None:
        embedder = _make_similar_embedder(
            "What is Python?",
            ["Tell me about Python programming"],
        )
        cache = SemantiCache(
            backend=InMemoryBackend(),
            embedder=embedder,
            similarity_threshold=0.90,
        )

        # First call: miss
        result1 = await cache.cache(
            prompt="What is Python?",
            generator=lambda: "Python is a programming language.",
        )
        assert result1.hit is False
        assert result1.response == "Python is a programming language."

        # Second call with similar prompt: hit
        result2 = await cache.cache(
            prompt="Tell me about Python programming",
            generator=lambda: "Should not be called",
        )
        assert result2.hit is True
        assert result2.response == "Python is a programming language."

    async def test_metrics_tracked_across_operations(self) -> None:
        embedder = _make_similar_embedder("prompt", ["similar prompt"])
        cache = SemantiCache(
            backend=InMemoryBackend(),
            embedder=embedder,
            similarity_threshold=0.90,
        )

        await cache.cache(prompt="prompt", generator=lambda: "response")
        await cache.cache(prompt="similar prompt", generator=lambda: "nope")

        metrics = cache.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["cache_hits"] >= 1
        assert metrics["cache_misses"] >= 1


# ---------------------------------------------------------------------------
# Integration: mock OpenAI
# ---------------------------------------------------------------------------


class TestMockOpenAIIntegration:
    async def test_openai_compat_cache_hit(self) -> None:
        """Simulate OpenAI middleware with cache."""
        embedder = _make_similar_embedder(
            "Explain quantum computing",
            ["What is quantum computing?"],
        )
        cache = SemantiCache(
            backend=InMemoryBackend(),
            embedder=embedder,
            similarity_threshold=0.90,
        )

        # Store initial response
        await cache.put("Explain quantum computing", "Quantum computing uses qubits...")

        # Simulate a similar query hitting the cache
        result = await cache.get("What is quantum computing?")
        assert result is not None
        assert result.hit is True
        assert "qubits" in result.response


# ---------------------------------------------------------------------------
# Integration: security + cache
# ---------------------------------------------------------------------------


class TestSecurityIntegration:
    async def test_sanitized_input_cache(self) -> None:
        """Ensure sanitized input still produces cache hits."""
        embedder = _make_similar_embedder("What is AI?", ["What is AI?"])
        cache = SemantiCache(
            backend=InMemoryBackend(),
            embedder=embedder,
            similarity_threshold=0.90,
        )

        clean = sanitize_input("  What is AI?\x00  ")
        await cache.put(clean, "AI is artificial intelligence.")

        result = await cache.get(clean)
        assert result is not None
        assert result.hit is True

    def test_encrypt_cached_response(self) -> None:
        key = CacheEncryptor.generate_key()
        enc = CacheEncryptor(key)

        response = "This is the LLM response."
        encrypted = enc.encrypt(response)
        decrypted = enc.decrypt(encrypted)
        assert decrypted == response


# ---------------------------------------------------------------------------
# Integration: strategies + cache
# ---------------------------------------------------------------------------


class TestStrategiesIntegration:
    async def test_batch_cache_integration(self) -> None:
        embedder = _make_embedder()
        cache = SemantiCache(
            backend=InMemoryBackend(),
            embedder=embedder,
        )

        items = [
            BatchCacheItem(prompt="p1", response="r1"),
            BatchCacheItem(prompt="p2", response="r2"),
            BatchCacheItem(prompt="p3", response="r3"),
        ]
        count = await batch_cache_put(cache, items)
        assert count == 3

    def test_lru_tracks_access_pattern(self) -> None:
        policy = LRUEvictionPolicy(max_entries=3)
        policy.record_access("a")
        policy.record_access("b")
        policy.record_access("c")
        policy.record_access("d")  # Over limit

        keys = policy.get_keys_to_evict()
        assert "a" in keys  # Oldest should be evicted
