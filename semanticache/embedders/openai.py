"""OpenAI embedding provider."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from semanticache.embedders import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using the OpenAI Embeddings API.

    Requires the ``openai`` package and a valid API key (set via the
    ``OPENAI_API_KEY`` environment variable or passed as *api_key*).

    Args:
        model: OpenAI embedding model name.
        api_key: Optional API key override.
        client: An existing ``openai.AsyncOpenAI`` instance.  If provided,
            *api_key* is ignored.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client = client

    async def _get_client(self) -> Any:
        """Lazily create or return the async OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "openai is required for OpenAIEmbedder. Install it with: pip install openai"
                ) from exc

            kwargs: dict[str, Any] = {}
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def embed(self, text: str) -> np.ndarray:
        """Compute an embedding for *text* via the OpenAI API.

        Args:
            text: The input text to embed.

        Returns:
            A 1-D float32 numpy array.
        """
        client = await self._get_client()
        try:
            response = await client.embeddings.create(
                input=text,
                model=self._model,
            )
        except Exception:
            logger.exception("OpenAI embedding request failed (model=%s)", self._model)
            raise

        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)
