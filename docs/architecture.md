# Architecture

## System Design

SemantiCache is a modular semantic caching middleware for LLMs. The architecture is layered and extensible via pluggable backends, embedders, and strategies.

```mermaid
graph TD
    A[Client Application] -->|prompt| B[SemantiCache Core]
    B --> C{Cache Lookup}
    C -->|HIT| D[Return Cached Response]
    C -->|MISS| E[LLM API Call]
    E --> F[Store in Cache]
    F --> D

    B --> G[Embedder Layer]
    G --> G1[SentenceTransformer]
    G --> G2[OpenAI Embeddings]

    B --> H[Backend Layer]
    H --> H1[In-Memory]
    H --> H2[Redis]

    B --> I[Security Layer]
    I --> I1[Input Sanitization]
    I --> I2[Cache Key Hashing]
    I --> I3[AES-256 Encryption]
    I --> I4[Rate Limiting]

    B --> J[Strategies Layer]
    J --> J1[LRU Eviction]
    J --> J2[Frequency Scoring]
    J --> J3[Namespace Isolation]
    J --> J4[Batch Caching]

    B --> K[Observability]
    K --> K1[Prometheus Metrics]
    K --> K2[Latency Histogram]
    K --> K3[Cost Tracking]
    K --> K4[JSON/CSV Export]

    B --> L[Middleware]
    L --> L1[CachedOpenAI]
    L --> L2[CachedLiteLLM]

    B --> M[Dashboard]
    M --> M1[FastAPI + WebSocket]
    M --> M2[Real-time Metrics]
```

## Request Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant SC as SemantiCache
    participant Sec as Security
    participant Emb as Embedder
    participant Be as Backend
    participant LLM as LLM API

    App->>SC: cache(prompt, generator)
    SC->>Sec: sanitize_input(prompt)
    SC->>Emb: embed(prompt)
    Emb-->>SC: embedding vector
    SC->>Be: search(embedding, threshold)
    alt Cache Hit
        Be-->>SC: (response, similarity, cached_at)
        SC->>SC: record_hit()
        SC-->>App: CacheResult(hit=True)
    else Cache Miss
        Be-->>SC: None
        SC->>LLM: generator()
        LLM-->>SC: response
        SC->>Be: store(embedding, response)
        SC->>SC: record_miss()
        SC-->>App: CacheResult(hit=False)
    end
```

## Module Layout

```
semanticache/
├── __init__.py          # Public API exports
├── core.py              # SemantiCache class, CacheResult
├── security.py          # Sanitization, hashing, encryption, rate limiting
├── strategies.py        # LRU eviction, namespaces, batch ops, warming
├── cli.py               # CLI commands (serve, stats, clear, benchmark)
├── backends/
│   ├── __init__.py      # BaseBackend ABC
│   ├── memory.py        # In-memory backend (numpy cosine similarity)
│   └── redis.py         # Redis backend (distributed)
├── embedders/
│   ├── __init__.py      # BaseEmbedder ABC
│   ├── sentence_transformers.py  # Local embeddings
│   └── openai.py        # OpenAI API embeddings
├── middleware/
│   ├── __init__.py      # Middleware exports
│   ├── openai_compat.py # Drop-in OpenAI wrapper
│   └── litellm_compat.py # LiteLLM wrapper
└── utils/
    ├── __init__.py      # Utility exports
    └── metrics.py       # MetricsTracker with Prometheus, histograms, CSV
```

## Key Design Decisions

1. **Async-first**: All backend and embedder operations are async for maximum throughput.
2. **Pluggable backends**: Abstract base classes allow custom storage implementations.
3. **Thread-safe metrics**: `MetricsTracker` uses `threading.Lock` for safe concurrent access.
4. **Namespace isolation**: Cache entries are partitioned by namespace for multi-tenant use.
5. **Lazy initialization**: Embedder models and clients are loaded on first use to minimize startup time.
6. **Security by default**: Input sanitization prevents cache poisoning; optional encryption protects data at rest.
