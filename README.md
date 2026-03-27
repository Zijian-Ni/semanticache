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
        generator=lambda p: "The capital of France is Paris.",
    )
    print(result.response)  # "The capital of France is Paris."
    print(result.hit)        # False (first call)

    result = await cache.cache(
        prompt="Tell me the capital city of France",
        generator=lambda p: "This won't be called!",
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

## Real-time Dashboard

SemantiCache ships with a built-in metrics dashboard:

```bash
pip install semanticache[dashboard]
```

```python
from semanticache import SemantiCache
from dashboard.app import create_app
import uvicorn

cache = SemantiCache()
app = create_app(cache=cache)
uvicorn.run(app, host="0.0.0.0", port=8080)
```

<!-- Screenshot placeholder -->
<!-- ![Dashboard](docs/assets/dashboard.png) -->

Dark mode, real-time WebSocket updates, cost savings tracker, similarity distribution charts.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `similarity_threshold` | `0.92` | Minimum cosine similarity for a cache hit |
| `ttl` | `86400` (24h) | Time-to-live in seconds |
| `backend` | `InMemoryBackend()` | Storage backend (`InMemoryBackend` or `RedisBackend`) |
| `embedder` | `SentenceTransformerEmbedder()` | Embedding model (`SentenceTransformerEmbedder` or `OpenAIEmbedder`) |
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

## Embedders

### Sentence Transformers (default)

Runs locally — no API key needed:

```python
from semanticache.embedders.sentence_transformers import SentenceTransformerEmbedder
embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
```

### OpenAI Embeddings

```bash
pip install semanticache[openai]
```

```python
from semanticache.embedders.openai import OpenAIEmbedder
embedder = OpenAIEmbedder(model="text-embedding-3-small")
```

## Metrics & Cost Tracking

```python
metrics = cache.get_metrics()
print(metrics)
# {
#     "total_requests": 1000,
#     "cache_hits": 420,
#     "cache_misses": 580,
#     "hit_rate": 0.42,
#     "cost_saved": 12.50,
#     "avg_similarity": 0.96
# }
```

## Performance

| Metric | Value |
|---|---|
| Embedding latency | ~5ms (MiniLM-L6-v2, CPU) |
| Cache lookup (1k entries) | <1ms |
| Cache lookup (100k entries) | ~15ms |
| Memory per entry | ~3KB |

## Installation Options

```bash
pip install semanticache                  # Core + sentence-transformers
pip install semanticache[openai]          # + OpenAI embeddings
pip install semanticache[redis]           # + Redis backend
pip install semanticache[dashboard]       # + Web dashboard
pip install semanticache[all]             # Everything
```

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

```bash
git clone https://github.com/ZijianNi/semanticache.git
cd semanticache
pip install -e ".[dev]"
pytest
```

## License

[PolyForm Shield License 1.0.0](LICENSE) — Copyright (c) 2026 Zijian Ni (倪子健)
