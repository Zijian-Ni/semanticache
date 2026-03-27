"""Embedding providers for SemantiCache."""

from __future__ import annotations

import abc

import numpy as np


class BaseEmbedder(abc.ABC):
    """Abstract base class for embedding providers.

    Subclasses must implement the async ``embed`` method which converts
    a text string into a numpy vector.
    """

    @abc.abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Compute an embedding vector for *text*.

        Args:
            text: The input text to embed.

        Returns:
            A 1-D numpy array of float32 values.
        """
        ...


__all__ = ["BaseEmbedder"]
