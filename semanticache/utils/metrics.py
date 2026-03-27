"""Usage metrics tracking for SemantiCache.

Includes Prometheus-compatible export, structured JSON logging, latency
histograms, configurable cost models, and CSV export.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default histogram bucket boundaries (milliseconds)
# ---------------------------------------------------------------------------

DEFAULT_LATENCY_BUCKETS: tuple[float, ...] = (
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    float("inf"),
)

# ---------------------------------------------------------------------------
# Per-model token pricing
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4": {"input_per_1k": 0.03, "output_per_1k": 0.06},
    "gpt-4-turbo": {"input_per_1k": 0.01, "output_per_1k": 0.03},
    "gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015},
    "gpt-3.5-turbo": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
    "claude-3-opus": {"input_per_1k": 0.015, "output_per_1k": 0.075},
    "claude-3-sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "claude-3-haiku": {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
}


class MetricsTracker:
    """Thread-safe tracker for cache usage metrics.

    Tracks hit/miss counts, estimated token savings, similarity score
    distributions, latency histograms, and per-model cost savings.

    Args:
        price_per_1k_input: Default cost per 1 000 input tokens in USD.
        price_per_1k_output: Default cost per 1 000 output tokens in USD.
        avg_prompt_tokens: Assumed average input tokens per request
            (used for savings estimation).
        avg_completion_tokens: Assumed average output tokens per request
            (used for savings estimation).
        model_pricing: Per-model pricing overrides.  Keys are model names,
            values are dicts with ``input_per_1k`` and ``output_per_1k``.
        latency_buckets: Histogram bucket boundaries in milliseconds.
    """

    def __init__(
        self,
        price_per_1k_input: float = 0.002,
        price_per_1k_output: float = 0.008,
        avg_prompt_tokens: int = 200,
        avg_completion_tokens: int = 300,
        model_pricing: dict[str, dict[str, float]] | None = None,
        latency_buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS,
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

        # Per-model pricing
        self._model_pricing: dict[str, dict[str, float]] = dict(DEFAULT_MODEL_PRICING)
        if model_pricing:
            self._model_pricing.update(model_pricing)
        self._cost_saved_by_model: dict[str, float] = {}

        # Latency histogram
        self._latency_buckets = latency_buckets
        self._latency_counts: list[int] = [0] * len(latency_buckets)
        self._latency_sum: float = 0.0
        self._latency_count: int = 0
        self._latencies: list[float] = []

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
        model: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Record a cache hit.

        Args:
            similarity_score: Cosine similarity of the matched entry.
            prompt_tokens: Actual prompt token count (uses average if None).
            completion_tokens: Actual completion token count (uses average if None).
            model: Model name for per-model cost tracking.
            latency_ms: Response latency in milliseconds.
        """
        pt = prompt_tokens if prompt_tokens is not None else self._avg_prompt_tokens
        ct = completion_tokens if completion_tokens is not None else self._avg_completion_tokens
        tokens = pt + ct

        # Determine pricing
        if model and model in self._model_pricing:
            pricing = self._model_pricing[model]
            cost = (pt / 1000.0) * pricing["input_per_1k"] + (ct / 1000.0) * pricing[
                "output_per_1k"
            ]
        else:
            cost = (pt / 1000.0) * self._price_per_1k_input + (
                ct / 1000.0
            ) * self._price_per_1k_output

        with self._lock:
            self._total_requests += 1
            self._cache_hits += 1
            self._total_tokens_saved += tokens
            self._cost_saved += cost
            self._similarity_scores.append(similarity_score)

            if model:
                self._cost_saved_by_model[model] = self._cost_saved_by_model.get(model, 0.0) + cost

            if latency_ms is not None:
                self._record_latency(latency_ms)

    def record_miss(self, latency_ms: float | None = None) -> None:
        """Record a cache miss.

        Args:
            latency_ms: Response latency in milliseconds.
        """
        with self._lock:
            self._total_requests += 1
            self._cache_misses += 1
            if latency_ms is not None:
                self._record_latency(latency_ms)

    def _record_latency(self, latency_ms: float) -> None:
        """Record a latency sample into the histogram (must hold lock)."""
        self._latency_sum += latency_ms
        self._latency_count += 1
        self._latencies.append(latency_ms)
        for i, bound in enumerate(self._latency_buckets):
            if latency_ms <= bound:
                self._latency_counts[i] += 1
                break

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a snapshot of all metrics as a dictionary."""
        with self._lock:
            histogram = {
                f"le_{b}" if not math.isinf(b) else "le_inf": c
                for b, c in zip(self._latency_buckets, self._latency_counts)
            }
            return {
                "total_requests": self._total_requests,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": (
                    self._cache_hits / self._total_requests if self._total_requests > 0 else 0.0
                ),
                "total_tokens_saved": self._total_tokens_saved,
                "cost_saved_usd": round(self._cost_saved, 6),
                "cost_saved_by_model": dict(self._cost_saved_by_model),
                "similarity_scores": list(self._similarity_scores),
                "latency_histogram": histogram,
                "latency_avg_ms": (
                    round(self._latency_sum / self._latency_count, 4)
                    if self._latency_count > 0
                    else 0.0
                ),
                "latency_p50_ms": self._percentile(50.0),
                "latency_p95_ms": self._percentile(95.0),
                "latency_p99_ms": self._percentile(99.0),
            }

    def _percentile(self, p: float) -> float:
        """Compute the *p*-th percentile of latency samples (must hold lock)."""
        if not self._latencies:
            return 0.0
        sorted_lat = sorted(self._latencies)
        idx = int(len(sorted_lat) * p / 100.0)
        idx = min(idx, len(sorted_lat) - 1)
        return round(sorted_lat[idx], 4)

    def to_json(self, **kwargs: Any) -> str:
        """Return metrics as a JSON string.

        Extra keyword arguments are forwarded to ``json.dumps``.
        """
        return json.dumps(self.to_dict(), **kwargs)

    def to_csv(self) -> str:
        """Export metrics as a CSV string.

        Returns:
            CSV-formatted string with two columns: ``metric`` and ``value``.
        """
        data = self.to_dict()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["metric", "value"])

        for key, value in data.items():
            if isinstance(value, (dict, list)):
                writer.writerow([key, json.dumps(value)])
            else:
                writer.writerow([key, value])

        return buf.getvalue()

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format.

        Returns:
            Prometheus-compatible metrics string.
        """
        with self._lock:
            lines: list[str] = []

            lines.append("# HELP semanticache_requests_total Total cache lookup requests.")
            lines.append("# TYPE semanticache_requests_total counter")
            lines.append(f"semanticache_requests_total {self._total_requests}")

            lines.append("# HELP semanticache_hits_total Total cache hits.")
            lines.append("# TYPE semanticache_hits_total counter")
            lines.append(f"semanticache_hits_total {self._cache_hits}")

            lines.append("# HELP semanticache_misses_total Total cache misses.")
            lines.append("# TYPE semanticache_misses_total counter")
            lines.append(f"semanticache_misses_total {self._cache_misses}")

            lines.append("# HELP semanticache_hit_rate Cache hit rate.")
            lines.append("# TYPE semanticache_hit_rate gauge")
            hr = self._cache_hits / self._total_requests if self._total_requests > 0 else 0.0
            lines.append(f"semanticache_hit_rate {hr}")

            lines.append("# HELP semanticache_tokens_saved_total Estimated tokens saved.")
            lines.append("# TYPE semanticache_tokens_saved_total counter")
            lines.append(f"semanticache_tokens_saved_total {self._total_tokens_saved}")

            lines.append("# HELP semanticache_cost_saved_usd Estimated cost saved in USD.")
            lines.append("# TYPE semanticache_cost_saved_usd counter")
            lines.append(f"semanticache_cost_saved_usd {round(self._cost_saved, 6)}")

            lines.append("# HELP semanticache_latency_ms Latency histogram in milliseconds.")
            lines.append("# TYPE semanticache_latency_ms histogram")
            cumulative = 0
            for bound, count in zip(self._latency_buckets, self._latency_counts):
                cumulative += count
                label = "+Inf" if math.isinf(bound) else str(bound)
                lines.append(f'semanticache_latency_ms_bucket{{le="{label}"}} {cumulative}')
            lines.append(f"semanticache_latency_ms_sum {round(self._latency_sum, 4)}")
            lines.append(f"semanticache_latency_ms_count {self._latency_count}")

            return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Structured JSON logging
    # ------------------------------------------------------------------

    def log_structured(self, event: str = "metrics_snapshot") -> None:
        """Emit a structured JSON log entry with the current metrics snapshot.

        Args:
            event: Event name for the log entry.
        """
        data = self.to_dict()
        data["event"] = event
        data["timestamp"] = time.time()
        logger.info(json.dumps(data))

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
            self._cost_saved_by_model.clear()
            self._latency_counts = [0] * len(self._latency_buckets)
            self._latency_sum = 0.0
            self._latency_count = 0
            self._latencies.clear()
