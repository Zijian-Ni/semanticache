"""Distributed caching with Redis backend.

This example demonstrates how to use SemantiCache with a Redis backend
so that cached embeddings and responses are shared across multiple
application instances.

Requirements:
    pip install semanticache[redis]
    # A running Redis server (e.g. redis://localhost:6379)
"""

import asyncio
from semanticache import SemantiCache
from semanticache.backends.redis import RedisBackend


async def main():
    # Connect to a Redis instance
    backend = RedisBackend(url="redis://localhost:6379/0")

    cache = SemantiCache(
        backend=backend,
        similarity_threshold=0.90,
        ttl=7200,  # 2-hour TTL
    )

    # First call - cache miss, calls the generator
    result = await cache.cache(
        prompt="Explain how photosynthesis works",
        generator=lambda: (
            "Photosynthesis is the process by which green plants convert "
            "sunlight, water, and carbon dioxide into glucose and oxygen."
        ),
    )
    print(f"[Miss] Response: {result.response}")

    # Similar prompt - served from Redis cache
    result = await cache.cache(
        prompt="How does photosynthesis work in plants?",
        generator=lambda: "This generator should not be called.",
    )
    print(f"[Hit]  Similarity: {result.similarity_score:.3f}")
    print(f"       Response: {result.response}")

    # Inspect cache size
    size = await backend.size()
    print(f"\nEntries in Redis: {size}")

    # Metrics
    metrics = cache.get_metrics()
    print(f"Hit rate: {metrics['hit_rate']:.1%}")

    # Clean up
    await backend.clear()
    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
