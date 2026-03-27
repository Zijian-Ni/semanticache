# Getting Started

## Installation

Install SemantiCache with pip:

```bash
pip install semanticache
```

For OpenAI integration:

```bash
pip install "semanticache[openai]"
```

For all optional dependencies:

```bash
pip install "semanticache[all]"
```

## Quick Start

### Basic Usage

```python
from semanticache import SemanticCache

# Create a cache with default settings (in-memory, sentence-transformers)
cache = SemanticCache()

# Store a response
cache.set("What is the capital of France?", "The capital of France is Paris.")

# Retrieve with semantic matching
result = cache.get("What's France's capital city?")
print(result)  # "The capital of France is Paris."
```

### Drop-in OpenAI Wrapper

Replace your OpenAI client with the cached version for instant savings:

```python
from semanticache.integrations.openai import CachedOpenAI

client = CachedOpenAI()  # wraps openai.OpenAI()

# First call hits the API
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)

# Semantically similar call returns cached response (no API call)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is quantum computing?"}],
)
```

### Using Redis for Distributed Caching

```python
from semanticache import SemanticCache
from semanticache.backends.redis import RedisBackend

cache = SemanticCache(
    backend=RedisBackend(url="redis://localhost:6379/0"),
    similarity_threshold=0.85,
)
```

### Launching the Dashboard

```python
from semanticache.dashboard import start_dashboard

# Starts a FastAPI server with live metrics at http://localhost:8787
start_dashboard(cache, port=8787)
```

## Core Concepts

### Similarity Threshold

The `similarity_threshold` parameter (0.0 to 1.0) controls how similar a query must be to a cached entry to count as a hit. Higher values require closer matches:

- **0.95** -- Very strict, nearly identical queries only
- **0.85** -- Good default, catches rephrasings (recommended)
- **0.75** -- Loose matching, may return less relevant results

### Backends

SemantiCache supports pluggable storage backends:

- **MemoryBackend** -- Default. Fast, single-process, supports TTL.
- **RedisBackend** -- Distributed caching across multiple processes or machines.

### Embedders

Choose how text is converted to vectors:

- **SentenceTransformerEmbedder** -- Default. Runs locally, no API key needed.
- **OpenAIEmbedder** -- Uses OpenAI's embedding API. Requires `OPENAI_API_KEY`.

## Next Steps

- [Configuration Reference](configuration.md) for all available options
- Check the [CHANGELOG](../CHANGELOG.md) for the latest updates
