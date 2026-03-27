# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-03-27

### Added
- **Security layer** (`semanticache.security`):
  - Input sanitization: strip control chars, normalize Unicode
  - SHA-256 cache key hashing to prevent cache poisoning
  - Optional AES-256-GCM encryption for cached responses at rest
  - Sliding-window rate limiter for API endpoints
  - Configurable max prompt length validation
- **Advanced cache strategies** (`semanticache.strategies`):
  - LRU eviction policy with configurable max entries
  - Frequency-weighted scoring for smarter eviction
  - Namespace manager with allow-list support
  - Batch caching (store multiple prompts at once)
  - Cache warming from JSON files
- **Enhanced observability** (`semanticache.utils.metrics`):
  - Prometheus-compatible metrics export
  - Latency histogram with configurable buckets
  - Per-model cost tracking with configurable pricing
  - Percentile calculations (p50, p95, p99)
  - CSV export
  - Structured JSON logging
- **CLI tool** (`semanticache.cli`):
  - `semanticache serve` — start dashboard server
  - `semanticache stats` — show cache statistics
  - `semanticache clear` — clear cache
  - `semanticache benchmark` — run performance benchmarks
- **Documentation**:
  - Architecture diagram with Mermaid
  - Security model documentation
  - Benchmark results
  - Full API reference
- **Repository polish**:
  - SECURITY.md — responsible disclosure policy
  - CONTRIBUTING.md — contribution guidelines with CLA
  - CODE_OF_CONDUCT.md — Contributor Covenant
  - .github/FUNDING.yml — GitHub Sponsors
- **Testing**:
  - 80+ tests covering security, strategies, CLI, metrics, and integration
  - Mock OpenAI integration tests
  - Encryption/decryption roundtrip tests

### Changed
- Bumped version to 0.2.0
- `MetricsTracker.record_hit()` now accepts `model` and `latency_ms` parameters
- `MetricsTracker.record_miss()` now accepts `latency_ms` parameter
- `MetricsTracker.to_dict()` now includes latency histogram and per-model costs

### Dependencies
- Added optional `[security]` extra: cryptography>=41.0.0
- Added optional `[cli]` extra: typer>=0.9.0, rich>=13.0.0
- Added CLI entry point: `semanticache` command

## [0.1.0] - 2026-03-27

### Added
- Initial release of SemantiCache
- Core semantic caching engine with configurable similarity threshold
- In-memory backend with TTL support
- Redis backend for distributed caching
- Sentence Transformers embedder (local, no API key needed)
- OpenAI embeddings support
- Drop-in OpenAI client wrapper (`CachedOpenAI`)
- LiteLLM compatibility layer
- Real-time metrics dashboard with dark mode UI
- Cost savings tracker
- WebSocket-powered live metrics updates
- Comprehensive test suite
