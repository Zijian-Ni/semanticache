"""Drop-in cached replacement for the OpenAI Python client."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from semanticache.core import SemantiCache

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Lightweight response wrappers so callers can access .choices etc.
# ------------------------------------------------------------------

@dataclass(slots=True)
class _Message:
    role: str = "assistant"
    content: str = ""


@dataclass(slots=True)
class _Choice:
    index: int = 0
    message: _Message = field(default_factory=_Message)
    finish_reason: str = "stop"


@dataclass(slots=True)
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class _ChatCompletion:
    id: str = "cached"
    object: str = "chat.completion"
    choices: list[_Choice] = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)
    model: str = ""
    _cache_hit: bool = False


# ------------------------------------------------------------------
# Completions-like interface
# ------------------------------------------------------------------

class _CachedCompletions:
    """Mimics ``openai.chat.completions`` with caching."""

    def __init__(self, cache: SemantiCache, client: Any) -> None:
        self._cache = cache
        self._client = client

    async def acreate(self, **kwargs: Any) -> _ChatCompletion:
        """Async version of ``create`` with semantic cache lookup.

        Accepts the same keyword arguments as
        ``openai.chat.completions.create``.
        """
        messages: list[dict[str, str]] = kwargs.get("messages", [])
        prompt_text = self._extract_prompt(messages)
        model = kwargs.get("model", "")
        namespace = f"openai:{model}"

        cached = await self._cache.get(prompt_text, namespace=namespace)
        if cached is not None:
            logger.debug("OpenAI cache hit (similarity=%.4f)", cached.similarity_score)
            return self._wrap_response(cached.response, model=model, cache_hit=True)

        # Cache miss -- call the real client
        response = await self._client.chat.completions.create(**kwargs)
        response_text = response.choices[0].message.content or ""

        await self._cache.put(prompt_text, response_text, namespace=namespace)
        return response

    def create(self, **kwargs: Any) -> _ChatCompletion | Any:
        """Synchronous ``create`` that delegates to :meth:`acreate`.

        If an event loop is already running the call is scheduled on it;
        otherwise a new loop is created.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.acreate(**kwargs)).result()

        return asyncio.run(self.acreate(**kwargs))

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

    @staticmethod
    def _wrap_response(
        text: str,
        model: str = "",
        cache_hit: bool = False,
    ) -> _ChatCompletion:
        return _ChatCompletion(
            choices=[_Choice(message=_Message(content=text))],
            model=model,
            _cache_hit=cache_hit,
        )


class _CachedChat:
    """Namespace object exposing a ``completions`` attribute."""

    def __init__(self, cache: SemantiCache, client: Any) -> None:
        self.completions = _CachedCompletions(cache, client)


class CachedOpenAI:
    """Drop-in replacement for ``openai.OpenAI`` with semantic caching.

    Usage::

        from semanticache.middleware import CachedOpenAI
        client = CachedOpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
        )

    Args:
        cache: A pre-configured SemantiCache instance.  If None a default
            one is created.
        api_key: OpenAI API key (forwarded to the real client).
        openai_client: An existing ``openai.AsyncOpenAI`` instance to
            wrap.  If None one is created automatically.
    """

    def __init__(
        self,
        cache: SemantiCache | None = None,
        api_key: str | None = None,
        openai_client: Any | None = None,
    ) -> None:
        if cache is None:
            cache = SemantiCache()

        if openai_client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "openai is required for CachedOpenAI. "
                    "Install it with: pip install openai"
                ) from exc
            kwargs: dict[str, Any] = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            openai_client = openai.AsyncOpenAI(**kwargs)

        self._cache = cache
        self._client = openai_client
        self.chat = _CachedChat(cache, openai_client)
