"""Tests for cache strategies — LRU, namespaces, batch caching, warming."""

from __future__ import annotations

import json
import tempfile

import pytest

from semanticache.strategies import (
    BatchCacheItem,
    LRUEvictionPolicy,
    NamespaceManager,
    batch_cache_put,
    load_warm_data_from_json,
    warm_cache,
)


# ---------------------------------------------------------------------------
# LRU Eviction Policy
# ---------------------------------------------------------------------------


class TestLRUEvictionPolicy:
    def test_init_default(self) -> None:
        policy = LRUEvictionPolicy()
        assert policy.max_entries == 10_000
        assert policy.current_size == 0

    def test_init_custom(self) -> None:
        policy = LRUEvictionPolicy(max_entries=100, frequency_weight=0.5)
        assert policy.max_entries == 100

    def test_invalid_max_entries(self) -> None:
        with pytest.raises(ValueError, match="max_entries"):
            LRUEvictionPolicy(max_entries=0)

    def test_invalid_frequency_weight(self) -> None:
        with pytest.raises(ValueError, match="frequency_weight"):
            LRUEvictionPolicy(frequency_weight=1.5)

    def test_record_access_new_key(self) -> None:
        policy = LRUEvictionPolicy(max_entries=10)
        policy.record_access("key1")
        assert policy.current_size == 1

    def test_record_access_existing_key_increments_count(self) -> None:
        policy = LRUEvictionPolicy(max_entries=10)
        policy.record_access("key1")
        policy.record_access("key1")
        entry = policy._entries["key1"]
        assert entry.access_count == 2

    def test_should_evict_under_limit(self) -> None:
        policy = LRUEvictionPolicy(max_entries=5)
        for i in range(5):
            policy.record_access(f"key{i}")
        assert not policy.should_evict()

    def test_should_evict_over_limit(self) -> None:
        policy = LRUEvictionPolicy(max_entries=3)
        for i in range(4):
            policy.record_access(f"key{i}")
        assert policy.should_evict()

    def test_eviction_candidates_pure_lru(self) -> None:
        policy = LRUEvictionPolicy(max_entries=3)
        policy.record_access("old")
        policy.record_access("mid")
        policy.record_access("new")
        candidates = policy.eviction_candidates(1)
        assert len(candidates) == 1
        assert candidates[0].key == "old"

    def test_get_keys_to_evict(self) -> None:
        policy = LRUEvictionPolicy(max_entries=2)
        policy.record_access("a")
        policy.record_access("b")
        policy.record_access("c")
        keys = policy.get_keys_to_evict()
        assert len(keys) == 1
        assert keys[0] == "a"

    def test_get_keys_to_evict_within_limit(self) -> None:
        policy = LRUEvictionPolicy(max_entries=5)
        policy.record_access("a")
        assert policy.get_keys_to_evict() == []

    def test_remove(self) -> None:
        policy = LRUEvictionPolicy(max_entries=10)
        policy.record_access("key1")
        policy.remove("key1")
        assert policy.current_size == 0

    def test_reset(self) -> None:
        policy = LRUEvictionPolicy(max_entries=10)
        for i in range(5):
            policy.record_access(f"key{i}")
        policy.reset()
        assert policy.current_size == 0

    def test_frequency_weighted_eviction(self) -> None:
        policy = LRUEvictionPolicy(max_entries=2, frequency_weight=0.8)
        policy.record_access("frequent")
        # Access "frequent" many times
        for _ in range(10):
            policy.record_access("frequent")
        policy.record_access("rare")
        policy.record_access("also_rare")
        candidates = policy.eviction_candidates(1)
        # The rarely-accessed entry should be first candidate
        assert candidates[0].key != "frequent"


# ---------------------------------------------------------------------------
# Namespace manager
# ---------------------------------------------------------------------------


class TestNamespaceManager:
    def test_unrestricted_allows_any(self) -> None:
        mgr = NamespaceManager()
        mgr.validate("anything")  # Should not raise

    def test_restricted_allows_valid(self) -> None:
        mgr = NamespaceManager(allowed_namespaces={"prod", "staging"})
        mgr.validate("prod")  # Should not raise

    def test_restricted_rejects_invalid(self) -> None:
        mgr = NamespaceManager(allowed_namespaces={"prod", "staging"})
        with pytest.raises(ValueError, match="not allowed"):
            mgr.validate("dev")

    def test_register_and_active(self) -> None:
        mgr = NamespaceManager()
        mgr.register("ns1")
        mgr.register("ns2")
        assert mgr.active_namespaces == {"ns1", "ns2"}

    def test_unregister(self) -> None:
        mgr = NamespaceManager()
        mgr.register("ns1")
        mgr.unregister("ns1")
        assert mgr.active_namespaces == set()

    def test_allowed_namespaces_property(self) -> None:
        mgr = NamespaceManager(allowed_namespaces={"a", "b"})
        assert mgr.allowed_namespaces == {"a", "b"}

    def test_allowed_namespaces_none(self) -> None:
        mgr = NamespaceManager()
        assert mgr.allowed_namespaces is None


# ---------------------------------------------------------------------------
# Batch caching
# ---------------------------------------------------------------------------


class TestBatchCacheItem:
    def test_creation(self) -> None:
        item = BatchCacheItem(prompt="hello", response="world")
        assert item.prompt == "hello"
        assert item.response == "world"
        assert item.namespace == "default"

    def test_custom_namespace(self) -> None:
        item = BatchCacheItem(prompt="p", response="r", namespace="custom")
        assert item.namespace == "custom"


class TestBatchCachePut:
    async def test_batch_put(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        mock_cache = MagicMock()
        mock_cache.put = AsyncMock()

        items = [
            BatchCacheItem(prompt="p1", response="r1"),
            BatchCacheItem(prompt="p2", response="r2"),
        ]
        count = await batch_cache_put(mock_cache, items)
        assert count == 2
        assert mock_cache.put.call_count == 2

    async def test_batch_put_handles_errors(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        mock_cache = MagicMock()
        mock_cache.put = AsyncMock(side_effect=[None, RuntimeError("fail")])

        items = [
            BatchCacheItem(prompt="p1", response="r1"),
            BatchCacheItem(prompt="p2", response="r2"),
        ]
        count = await batch_cache_put(mock_cache, items)
        assert count == 1


# ---------------------------------------------------------------------------
# Cache warming
# ---------------------------------------------------------------------------


class TestLoadWarmData:
    def test_load_valid_json(self) -> None:
        data = [
            {"prompt": "p1", "response": "r1"},
            {"prompt": "p2", "response": "r2", "namespace": "ns2"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            items = load_warm_data_from_json(f.name)

        assert len(items) == 2
        assert items[0].prompt == "p1"
        assert items[0].namespace == "default"
        assert items[1].namespace == "ns2"

    def test_load_invalid_format(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "a list"}, f)
            f.flush()
            with pytest.raises(ValueError, match="array"):
                load_warm_data_from_json(f.name)


class TestWarmCache:
    async def test_warm_cache(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        data = [
            {"prompt": "p1", "response": "r1"},
            {"prompt": "p2", "response": "r2"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            mock_cache = MagicMock()
            mock_cache.put = AsyncMock()
            count = await warm_cache(mock_cache, f.name)

        assert count == 2
