<p align="center">
  <h1 align="center">SemantiCache</h1>
  <p align="center">
    <strong>Production-grade semantic caching for LLMs — save 30-60% on API costs</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/semanticache/"><img src="https://img.shields.io/pypi/v/semanticache?color=blue&style=flat-square" alt="PyPI"></a>
    <a href="https://github.com/ZijianNi/semanticache/actions"><img src="https://img.shields.io/github/actions/workflow/status/ZijianNi/semanticache/test.yml?style=flat-square" alt="Tests"></a>
    <img src="https://img.shields.io/pypi/pyversions/semanticache?style=flat-square" alt="Python">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-PolyForm--Shield--1.0.0-green?style=flat-square" alt="License"></a>
  </p>
</p>

---

Stop paying for the same LLM responses twice. **SemantiCache** intercepts your API calls, embeds prompts locally, and returns cached responses for semantically similar queries — no code rewrite needed.

## Why SemantiCache?

| Without cache | With SemantiCache |
|---|---|
| "What is the capital of France?" → API call ($) | "What is the capital of France?" → API call ($) |
| "Tell me France's capital city" → API call ($) | "Tell me France's capital city" → **Cache hit (free)** |
| "Capital of France?" → API call ($) | "Capital of France?" → **Cache hit (free)** |

Real-world workloads see **30-60% cache hit rates**, translating directly to cost savings.

## Architecture

See the full [architecture diagram](docs/architecture.md) for system design details.

```
Client → SemantiCache Core → [Security Layer] → [Embedder] → [Backend]
                ↓                                                 ↓
         [Strategies]                                    [In-Memory / Redis]
         (LRU, Batch, Namespaces)
                ↓
         [Observability]
         (Prometheus, Histograms, Cost Tracking)
```

## Quick Start

```bash
pip install semanticache
```

```python
import asyncio
from semanticache import SemantiCache

async def main():
    cache = SemantiCache(similarity_threshold=0.92)

    result = await cache.cache(
        prompt="What is the capital of France?",
        generator=lambda: "The capital of France is Paris.",
    )
    print(result.response)  # "The capital of France is Paris."
    print(result.hit)        # False (first call)

    result = await cache.cache(
        prompt="Tell me the capital city of France",
        generator=lambda: "This won't be called!",
    )
    print(result.hit)              # True
    print(result.similarity_score) # 0.95+

asyncio.run(main())
```

## Drop-in OpenAI Replacement

Two lines to add caching to your existing OpenAI code:

```python
from semanticache import SemantiCache
from semanticache.middleware import CachedOpenAI

cache = SemantiCache()
client = CachedOpenAI(cache=cache)

# Everything else stays exactly the same
response = await client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)
```

## Security

SemantiCache includes a built-in security layer. See [docs/security.md](docs/security.md).

```python
from semanticache.security import sanitize_input, CacheEncryptor

# Sanitize user input
clean = sanitize_input(user_prompt, max_length=32000)

# Encrypt cached responses at rest
key = CacheEncryptor.generate_key()
encryptor = CacheEncryptor(key)
encrypted = encryptor.encrypt("LLM response")
```

## Cache Strategies

```python
from semanticache.strategies import LRUEvictionPolicy, batch_cache_put, warm_cache

# LRU eviction
policy = LRUEvictionPolicy(max_entries=10000, frequency_weight=0.3)

# Batch caching
items = [BatchCacheItem(prompt="p1", response="r1"), ...]
await batch_cache_put(cache, items)

# Warm cache from file
await warm_cache(cache, "warmup_data.json")
```

## CLI

```bash
pip install semanticache[cli]

semanticache serve              # Start dashboard server
semanticache stats              # Show cache statistics
semanticache stats -f json      # JSON output
semanticache clear --yes        # Clear cache
semanticache benchmark -n 5000  # Performance benchmark
```

## Real-time Dashboard

```bash
pip install semanticache[dashboard]
semanticache serve --port 8080
```

Dark mode, real-time WebSocket updates, cost savings tracker, similarity distribution charts.

## Observability

```python
metrics = cache.get_metrics()

# Prometheus-compatible export
tracker = cache._metrics
print(tracker.to_prometheus())

# CSV export
print(tracker.to_csv())

# Per-model cost tracking
tracker.record_hit(model="gpt-4", similarity_score=0.95)
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `similarity_threshold` | `0.92` | Minimum cosine similarity for a cache hit |
| `ttl` | `86400` (24h) | Time-to-live in seconds |
| `backend` | `InMemoryBackend()` | Storage backend (`InMemoryBackend` or `RedisBackend`) |
| `embedder` | `SentenceTransformerEmbedder()` | Embedding model |
| `metrics_enabled` | `True` | Enable hit/miss/cost tracking |

## Backends

### In-Memory (default)

Zero-config, great for single-process applications:

```python
from semanticache import SemantiCache
cache = SemantiCache()  # uses InMemoryBackend by default
```

### Redis

For distributed caching across multiple workers:

```bash
pip install semanticache[redis]
```

```python
from semanticache import SemantiCache
from semanticache.backends.redis import RedisBackend

backend = RedisBackend(url="redis://localhost:6379")
cache = SemantiCache(backend=backend)
```

## Performance

| Metric | Value |
|---|---|
| Embedding latency | ~5ms (MiniLM-L6-v2, CPU) |
| Cache lookup (1k entries) | <1ms |
| Cache lookup (100k entries) | ~15ms |
| Memory per entry | ~3KB |

See [docs/benchmarks.md](docs/benchmarks.md) for detailed benchmarks.

## Installation Options

```bash
pip install semanticache                  # Core + sentence-transformers
pip install semanticache[openai]          # + OpenAI embeddings
pip install semanticache[redis]           # + Redis backend
pip install semanticache[dashboard]       # + Web dashboard
pip install semanticache[security]        # + AES-256 encryption
pip install semanticache[cli]             # + CLI tool
pip install semanticache[all]             # Everything
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [Security Model](docs/security.md)
- [Benchmarks](docs/benchmarks.md)
- [API Reference](docs/api-reference.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/ZijianNi/semanticache.git
cd semanticache
pip install -e ".[dev]"
pytest
```

## License

[PolyForm Shield License 1.0.0](LICENSE) — Copyright (c) 2026 Zijian Ni
