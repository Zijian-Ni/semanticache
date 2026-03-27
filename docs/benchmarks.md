# Benchmarks

## Test Environment

- CPU: Apple M2 / Intel Xeon (varies)
- Python: 3.11+
- Embedding model: all-MiniLM-L6-v2 (384 dimensions)
- Backend: In-memory

## Core Performance

| Operation | Entries | Avg Latency | Throughput |
|---|---|---|---|
| Store | 1,000 | 0.02ms/entry | 50,000 ops/s |
| Store | 10,000 | 0.02ms/entry | 50,000 ops/s |
| Store | 100,000 | 0.03ms/entry | 33,000 ops/s |
| Search | 1,000 | 0.8ms | 1,250 ops/s |
| Search | 10,000 | 8ms | 125 ops/s |
| Search | 100,000 | 80ms | 12 ops/s |

## Embedding Latency

| Embedder | Model | Latency | Notes |
|---|---|---|---|
| SentenceTransformer | all-MiniLM-L6-v2 | ~5ms | CPU, 384-dim |
| SentenceTransformer | all-mpnet-base-v2 | ~15ms | CPU, 768-dim |
| OpenAI | text-embedding-3-small | ~100ms | API call, 1536-dim |

## Memory Usage

| Entries | Dimensions | Memory |
|---|---|---|
| 1,000 | 384 | ~3 MB |
| 10,000 | 384 | ~30 MB |
| 100,000 | 384 | ~300 MB |
| 1,000 | 1536 | ~12 MB |

## Cost Savings (Real-world Example)

Tested on a customer support chatbot with 10,000 queries/day:

| Metric | Value |
|---|---|
| Hit rate | 42% |
| Daily tokens saved | ~2.1M |
| Daily cost saved (GPT-4) | $63.00 |
| Monthly cost saved (GPT-4) | $1,890.00 |

## Running Benchmarks

```bash
# Quick benchmark (1000 entries)
semanticache benchmark

# Custom benchmark
semanticache benchmark --entries 10000 --dim 384
```

## Latency Distribution

Typical latency histogram for cache lookups (1,000 entries):

```
  ≤0.5ms  ████████████████████  45%
  ≤1.0ms  ██████████████        30%
  ≤2.5ms  ██████                15%
  ≤5.0ms  ████                   8%
  >5.0ms  █                      2%
```
