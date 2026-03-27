"""Basic usage of SemantiCache."""
import asyncio
from semanticache import SemantiCache

async def main():
    cache = SemantiCache(similarity_threshold=0.92)

    # First call - cache miss, calls the generator
    result = await cache.cache(
        prompt="What is the capital of France?",
        generator=lambda p: "The capital of France is Paris."
    )
    print(f"Hit: {result.hit}, Response: {result.response}")

    # Second call - similar prompt, cache hit!
    result = await cache.cache(
        prompt="Tell me the capital city of France",
        generator=lambda p: "This won't be called!"
    )
    print(f"Hit: {result.hit}, Similarity: {result.similarity_score:.3f}")
    print(f"Response: {result.response}")

    # Check metrics
    metrics = cache.get_metrics()
    print(f"Hit rate: {metrics['hit_rate']:.1%}")

asyncio.run(main())
