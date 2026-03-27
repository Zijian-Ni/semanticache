"""Usage metrics tracking for SemantiCache."""

from __future__ import annotations

import json
import threading
from typing import Any


class MetricsTracker:
    """Thread-safe tracker for cache usage metrics.

    Tracks hit/miss counts, estimated token savings, and similarity score
    distributions.

    Args:
        price_per_1k_input: Cost per 1 000 input tokens in USD.
        price_per_1k_output: Cost per 1 000 output tokens in USD.
        avg_prompt_tokens: Assumed average input tokens per request
            (used for savings estimation).
        avg_completion_tokens: Assumed average output tokens per request
            (used for savings estimation).
    """

    def __init__(
        self,
        price_per_1k_input: float = 0.002,
        price_per_1k_output: float = 0.008,
        avg_prompt_tokens: int = 200,
        avg_completion_tokens: int = 300,
    ) -> None:
        self._lock = threading.Lock()

        self._total_requests: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._total_tokens_saved: int = 0
        self._cost_saved: float = 0.0
        self._similarity_scores: list[float] = []

        self._price_per_1k_input = price_per_1k_input
        self._price_per_1k_output = price_per_1k_output
        self._avg_prompt_tokens = avg_prompt_tokens
        self._avg_completion_tokens = avg_completion_tokens

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_requests(self) -> int:
        """Total number of cache lookups."""
        with self._lock:
            return self._total_requests

    @property
    def cache_hits(self) -> int:
        """Number of cache hits."""
        with self._lock:
            return self._cache_hits

    @property
    def cache_misses(self) -> int:
        """Number of cache misses."""
        with self._lock:
            return self._cache_misses

    @property
    def total_tokens_saved(self) -> int:
        """Estimated total tokens saved by cache hits."""
        with self._lock:
            return self._total_tokens_saved

    @property
    def cost_saved(self) -> float:
        """Estimated total cost saved in USD."""
        with self._lock:
            return self._cost_saved

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction in [0, 1].

        Returns 0.0 if no requests have been recorded.
        """
        with self._lock:
            if self._total_requests == 0:
                return 0.0
            return self._cache_hits / self._total_requests

    @property
    def similarity_scores(self) -> list[float]:
        """Copy of all recorded similarity scores from cache hits."""
        with self._lock:
            return list(self._similarity_scores)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_hit(
        self,
        similarity_score: float = 0.0,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        """Record a cache hit.

        Args:
            similarity_score: Cosine similarity of the matched entry.
            prompt_tokens: Actual prompt token count (uses average if None).
            completion_tokens: Actual completion token count (uses average if None).
        """
        pt = prompt_tokens if prompt_tokens is not None else self._avg_prompt_tokens
        ct = completion_tokens if completion_tokens is not None else self._avg_completion_tokens
        tokens = pt + ct
        cost = (pt / 1000.0) * self._price_per_1k_input + (ct / 1000.0) * self._price_per_1k_output

        with self._lock:
            self._total_requests += 1
            self._cache_hits += 1
            self._total_tokens_saved += tokens
            self._cost_saved += cost
            self._similarity_scores.append(similarity_score)

    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._total_requests += 1
            self._cache_misses += 1

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a snapshot of all metrics as a dictionary."""
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": self.hit_rate if self._total_requests > 0 else 0.0,
                "total_tokens_saved": self._total_tokens_saved,
                "cost_saved_usd": round(self._cost_saved, 6),
                "similarity_scores": list(self._similarity_scores),
            }

    def to_json(self, **kwargs: Any) -> str:
        """Return metrics as a JSON string.

        Extra keyword arguments are forwarded to ``json.dumps``.
        """
        return json.dumps(self.to_dict(), **kwargs)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all metrics to zero."""
        with self._lock:
            self._total_requests = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._total_tokens_saved = 0
            self._cost_saved = 0.0
            self._similarity_scores.clear()
