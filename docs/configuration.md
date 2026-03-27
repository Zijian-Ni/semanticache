# Configuration Reference

## SemanticCache

The main cache class accepts the following parameters:

```python
from semanticache import SemanticCache

cache = SemanticCache(
    backend=None,               # Storage backend (default: MemoryBackend)
    embedder=None,              # Embedding provider (default: SentenceTransformerEmbedder)
    similarity_threshold=0.85,  # Minimum cosine similarity for a cache hit (0.0-1.0)
    default_ttl=3600,           # Default time-to-live in seconds (default: 1 hour)
    max_entries=10000,          # Maximum number of cached entries (0 = unlimited)
    namespace="default",        # Cache namespace for key isolation
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `Backend` | `MemoryBackend()` | Storage backend instance |
| `embedder` | `Embedder` | `SentenceTransformerEmbedder()` | Embedding provider instance |
| `similarity_threshold` | `float` | `0.85` | Minimum cosine similarity to return a cached result |
| `default_ttl` | `int` | `3600` | Time-to-live for cache entries in seconds |
| `max_entries` | `int` | `10000` | Maximum cached entries before eviction (0 = unlimited) |
| `namespace` | `str` | `"default"` | Namespace for isolating cache entries |

## Backends

### MemoryBackend

In-process memory storage with TTL eviction.

```python
from semanticache.backends.memory import MemoryBackend

backend = MemoryBackend(
    max_size=10000,  # Maximum number of entries (0 = unlimited)
)
```

### RedisBackend

Distributed caching using Redis.

```python
from semanticache.backends.redis import RedisBackend

backend = RedisBackend(
    url="redis://localhost:6379/0",  # Redis connection URL
    key_prefix="semanticache:",      # Key prefix in Redis
    serializer="json",               # Serialization format: "json" or "msgpack"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | `"redis://localhost:6379/0"` | Redis connection URL |
| `key_prefix` | `str` | `"semanticache:"` | Prefix for all Redis keys |
| `serializer` | `str` | `"json"` | Serialization format |

## Embedders

### SentenceTransformerEmbedder

Runs locally using the `sentence-transformers` library. No API key needed.

```python
from semanticache.embedders.sentence_transformer import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(
    model_name="all-MiniLM-L6-v2",  # Model from HuggingFace
    device="cpu",                     # "cpu", "cuda", or "mps"
    batch_size=32,                    # Batch size for encoding
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"all-MiniLM-L6-v2"` | HuggingFace model name |
| `device` | `str` | `"cpu"` | Compute device |
| `batch_size` | `int` | `32` | Encoding batch size |

### OpenAIEmbedder

Uses the OpenAI embeddings API. Requires `OPENAI_API_KEY` environment variable.

```python
from semanticache.embedders.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(
    model="text-embedding-3-small",  # OpenAI embedding model
    api_key=None,                     # Defaults to OPENAI_API_KEY env var
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"text-embedding-3-small"` | OpenAI model name |
| `api_key` | `str \| None` | `None` | API key (falls back to env var) |

## Integrations

### CachedOpenAI

Drop-in replacement for the OpenAI client.

```python
from semanticache.integrations.openai import CachedOpenAI

client = CachedOpenAI(
    cache=None,                  # SemanticCache instance (creates default if None)
    similarity_threshold=0.85,   # Override cache threshold for this client
    **openai_kwargs,             # Passed through to openai.OpenAI()
)
```

### LiteLLM

Use SemantiCache with any LiteLLM-supported provider.

```python
from semanticache.integrations.litellm import CachedLiteLLM

client = CachedLiteLLM(
    cache=None,                  # SemanticCache instance (creates default if None)
    similarity_threshold=0.85,   # Override cache threshold
)

response = client.completion(
    model="anthropic/claude-3-haiku",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Dashboard

### start_dashboard

Launch the real-time metrics dashboard.

```python
from semanticache.dashboard import start_dashboard

start_dashboard(
    cache=cache,        # SemanticCache instance to monitor
    host="0.0.0.0",     # Bind address
    port=8787,          # Port number
    dark_mode=True,     # Enable dark mode UI
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache` | `SemanticCache` | required | Cache instance to monitor |
| `host` | `str` | `"0.0.0.0"` | Bind address |
| `port` | `int` | `8787` | Port number |
| `dark_mode` | `bool` | `True` | Dark mode toggle |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI embeddings and cached OpenAI client |
| `REDIS_URL` | Default Redis URL for RedisBackend |
| `SEMANTICACHE_THRESHOLD` | Default similarity threshold (overridden by code) |
| `SEMANTICACHE_TTL` | Default TTL in seconds (overridden by code) |
| `SEMANTICACHE_LOG_LEVEL` | Logging level: DEBUG, INFO, WARNING, ERROR |
