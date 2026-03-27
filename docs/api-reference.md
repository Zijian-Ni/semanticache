# API Reference

## Core

### `SemantiCache`

The main cache class.

```python
from semanticache import SemantiCache

cache = SemantiCache(
    backend=None,                # BaseBackend (default: InMemoryBackend)
    embedder=None,               # BaseEmbedder (default: SentenceTransformerEmbedder)
    similarity_threshold=0.92,   # Minimum cosine similarity for a hit
    ttl=86400,                   # Time-to-live in seconds
    metrics_enabled=True,        # Enable metrics tracking
)
```

#### `await cache.cache(prompt, generator, namespace="default") -> CacheResult`

Check cache; on miss, call generator and store.

#### `await cache.get(prompt, namespace="default") -> CacheResult | None`

Look up a cached entry. Returns `None` on miss.

#### `await cache.put(prompt, response, namespace="default", metadata=None) -> None`

Store a prompt/response pair.

#### `await cache.clear(namespace=None) -> int`

Clear entries. Returns count removed.

#### `cache.get_metrics() -> dict`

Return a metrics snapshot.

---

### `CacheResult`

Frozen dataclass returned from cache operations.

| Field | Type | Description |
|---|---|---|
| `response` | `str` | The LLM response text |
| `hit` | `bool` | Whether from cache |
| `similarity_score` | `float` | Cosine similarity (0.0 for misses) |
| `latency_ms` | `float` | Time taken in milliseconds |
| `cached_at` | `datetime \| None` | When originally cached |

---

## Security (`semanticache.security`)

### `sanitize_input(text, *, max_length=32000, unicode_form="NFC") -> str`

Sanitize user input (normalize, strip control chars, trim, truncate).

### `validate_prompt_length(text, max_length=32000) -> None`

Raise `ValueError` if text exceeds max_length.

### `hash_cache_key(prompt, namespace="default", salt="") -> str`

SHA-256 hash a cache key. Returns 64-char hex string.

### `CacheEncryptor(key: bytes)`

AES-256-GCM encryption for cached responses.

- `encrypt(plaintext: str) -> bytes`
- `decrypt(data: bytes) -> str`
- `generate_key() -> bytes` (static)

### `RateLimiter(max_requests=60, window_seconds=60)`

Sliding-window rate limiter.

- `is_allowed(client_key: str) -> bool`
- `reset(client_key=None) -> None`

---

## Strategies (`semanticache.strategies`)

### `LRUEvictionPolicy(max_entries=10000, frequency_weight=0.0)`

LRU eviction with optional frequency weighting.

- `record_access(key, namespace="default") -> None`
- `should_evict() -> bool`
- `get_keys_to_evict() -> list[str]`
- `remove(key) -> None`
- `reset() -> None`

### `NamespaceManager(allowed_namespaces=None)`

Manage namespace isolation.

- `validate(namespace) -> None`
- `register(namespace) -> None`
- `unregister(namespace) -> None`

### `BatchCacheItem(prompt, response, namespace="default", metadata={})`

Frozen dataclass for batch operations.

### `await batch_cache_put(cache, items) -> int`

Store multiple items. Returns count stored.

### `await warm_cache(cache, path) -> int`

Load warm-up data from JSON file and populate cache.

---

## Metrics (`semanticache.utils.metrics`)

### `MetricsTracker(...)`

Thread-safe metrics with Prometheus export.

- `record_hit(similarity_score, prompt_tokens, completion_tokens, model, latency_ms)`
- `record_miss(latency_ms)`
- `to_dict() -> dict`
- `to_json(**kwargs) -> str`
- `to_csv() -> str`
- `to_prometheus() -> str`
- `log_structured(event="metrics_snapshot") -> None`
- `reset() -> None`

---

## Backends

### `InMemoryBackend()`

In-memory storage with numpy cosine similarity.

### `RedisBackend(url=None, redis_client=None)`

Distributed Redis backend.

Both implement `BaseBackend` with: `store`, `search`, `delete`, `clear`, `size`.

---

## Embedders

### `SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")`

Local embeddings via sentence-transformers.

### `OpenAIEmbedder(model="text-embedding-3-small", api_key=None)`

OpenAI API embeddings.

Both implement `BaseEmbedder` with: `async embed(text) -> np.ndarray`.

---

## Middleware

### `CachedOpenAI(cache, api_key=None, openai_client=None)`

Drop-in OpenAI replacement. Use `client.chat.completions.create(...)`.

### `CachedLiteLLM(cache=None)`

LiteLLM wrapper. Use `client.completion(...)` or `client.acompletion(...)`.

---

## CLI

```bash
semanticache serve [--host 0.0.0.0] [--port 8080]
semanticache stats [--format table|json]
semanticache clear [--namespace NS] [--yes]
semanticache benchmark [--entries 1000] [--dim 384]
```
