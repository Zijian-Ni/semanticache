"""LLM client middleware wrappers for SemantiCache."""

from semanticache.middleware.litellm_compat import CachedLiteLLM
from semanticache.middleware.openai_compat import CachedOpenAI

__all__ = ["CachedOpenAI", "CachedLiteLLM"]
