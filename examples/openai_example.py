"""Drop-in OpenAI client replacement with semantic caching."""

import asyncio
from semanticache import SemantiCache
from semanticache.middleware import CachedOpenAI


async def main():
    cache = SemantiCache()
    client = CachedOpenAI(cache=cache)

    # First call - hits OpenAI API
    response = await client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": "Explain quantum computing"}]
    )
    print(response.choices[0].message.content)

    # Similar query - served from cache!
    response = await client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": "What is quantum computing?"}]
    )
    print(response.choices[0].message.content)

    # Check savings
    metrics = cache.get_metrics()
    print(f"Cost saved: ${metrics['cost_saved']:.4f}")


asyncio.run(main())
