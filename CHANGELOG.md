# Changelog

All notable changes to this project will be documented in this file.

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
