"""Cached wrapper around litellm completion functions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from semanticache.core import SemantiCache

logger = logging.getLogger(__name__)


class CachedLiteLLM:
    """Semantic-cache wrapper for ``litellm.completion`` / ``litellm.acompletion``.

    Usage::

        from semanticache.middleware import CachedLiteLLM
        llm = CachedLiteLLM()
        resp = await llm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

    Args:
        cache: A pre-configured SemantiCache instance.  If None a default
            one is created.
    """

    def __init__(self, cache: SemantiCache | None = None) -> None:
        if cache is None:
            cache = SemantiCache()
        self._cache = cache

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_prompt(messages: list[dict[str, str]]) -> str:
        """Build a cache key string from the messages list."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acompletion(self, **kwargs: Any) -> Any:
        """Async completion with semantic cache lookup.

        Accepts the same keyword arguments as ``litellm.acompletion``.

        Returns:
            A litellm ``ModelResponse`` on cache miss, or the cached
            response text wrapped in a compatible structure on hit.
        """
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "litellm is required for CachedLiteLLM. "
                "Install it with: pip install litellm"
            ) from exc

        messages: list[dict[str, str]] = kwargs.get("messages", [])
        prompt_text = self._extract_prompt(messages)
        model = kwargs.get("model", "")
        namespace = f"litellm:{model}"

        cached = await self._cache.get(prompt_text, namespace=namespace)
        if cached is not None:
            logger.debug("LiteLLM cache hit (similarity=%.4f)", cached.similarity_score)
            return self._build_response(cached.response, model=model)

        response = await litellm.acompletion(**kwargs)
        response_text: str = response.choices[0].message.content or ""

        await self._cache.put(prompt_text, response_text, namespace=namespace)
        return response

    def completion(self, **kwargs: Any) -> Any:
        """Synchronous completion with semantic cache lookup.

        Accepts the same keyword arguments as ``litellm.completion``.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.acompletion(**kwargs)).result()

        return asyncio.run(self.acompletion(**kwargs))

    # ------------------------------------------------------------------
    # Response builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response(text: str, model: str = "") -> dict[str, Any]:
        """Build a minimal litellm-compatible response dictionary."""
        return {
            "id": "cached",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "_cache_hit": True,
        }
