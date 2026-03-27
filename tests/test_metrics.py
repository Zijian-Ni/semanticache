"""Tests for the enhanced metrics tracker."""

from __future__ import annotations

import json

import pytest

from semanticache.utils.metrics import MetricsTracker


class TestMetricsTrackerBasic:
    def test_initial_state(self) -> None:
        tracker = MetricsTracker()
        assert tracker.total_requests == 0
        assert tracker.cache_hits == 0
        assert tracker.cache_misses == 0
        assert tracker.hit_rate == 0.0
        assert tracker.total_tokens_saved == 0
        assert tracker.cost_saved == 0.0
        assert tracker.similarity_scores == []

    def test_record_hit(self) -> None:
        tracker = MetricsTracker()
        tracker.record_hit(similarity_score=0.95)
        assert tracker.cache_hits == 1
        assert tracker.total_requests == 1

    def test_record_miss(self) -> None:
        tracker = MetricsTracker()
        tracker.record_miss()
        assert tracker.cache_misses == 1
        assert tracker.total_requests == 1

    def test_hit_rate(self) -> None:
        tracker = MetricsTracker()
        tracker.record_hit()
        tracker.record_miss()
        assert tracker.hit_rate == pytest.approx(0.5)

    def test_reset(self) -> None:
        tracker = MetricsTracker()
        tracker.record_hit()
        tracker.record_miss()
        tracker.reset()
        assert tracker.total_requests == 0


class TestMetricsTrackerLatency:
    def test_record_latency_with_hit(self) -> None:
        tracker = MetricsTracker()
        tracker.record_hit(latency_ms=5.0)
        data = tracker.to_dict()
        assert data["latency_avg_ms"] == 5.0

    def test_record_latency_with_miss(self) -> None:
        tracker = MetricsTracker()
        tracker.record_miss(latency_ms=100.0)
        data = tracker.to_dict()
        assert data["latency_avg_ms"] == 100.0

    def test_latency_percentiles(self) -> None:
        tracker = MetricsTracker()
        for i in range(100):
            tracker.record_hit(latency_ms=float(i), similarity_score=0.9)
        data = tracker.to_dict()
        assert data["latency_p50_ms"] > 0
        assert data["latency_p95_ms"] > data["latency_p50_ms"]
        assert data["latency_p99_ms"] >= data["latency_p95_ms"]

    def test_latency_histogram_populated(self) -> None:
        tracker = MetricsTracker()
        tracker.record_hit(latency_ms=2.0)
        tracker.record_hit(latency_ms=50.0)
        data = tracker.to_dict()
        histogram = data["latency_histogram"]
        assert sum(histogram.values()) >= 2


class TestMetricsTrackerModelPricing:
    def test_per_model_cost_tracking(self) -> None:
        tracker = MetricsTracker()
        tracker.record_hit(model="gpt-4", similarity_score=0.95)
        data = tracker.to_dict()
        assert "gpt-4" in data["cost_saved_by_model"]
        assert data["cost_saved_by_model"]["gpt-4"] > 0

    def test_unknown_model_uses_default_pricing(self) -> None:
        tracker = MetricsTracker()
        tracker.record_hit(model="unknown-model", similarity_score=0.95)
        data = tracker.to_dict()
        assert data["cost_saved_usd"] > 0

    def test_custom_model_pricing(self) -> None:
        tracker = MetricsTracker(
            model_pricing={"my-model": {"input_per_1k": 0.1, "output_per_1k": 0.2}}
        )
        tracker.record_hit(model="my-model", similarity_score=0.95)
        data = tracker.to_dict()
        assert "my-model" in data["cost_saved_by_model"]


class TestMetricsExport:
    def test_to_json(self) -> None:
        tracker = MetricsTracker()
        tracker.record_hit(similarity_score=0.95)
        result = json.loads(tracker.to_json())
        assert result["cache_hits"] == 1

    def test_to_csv(self) -> None:
        tracker = MetricsTracker()
        tracker.record_hit(similarity_score=0.95)
        csv_str = tracker.to_csv()
        assert "metric,value" in csv_str
        assert "cache_hits" in csv_str

    def test_to_prometheus(self) -> None:
        tracker = MetricsTracker()
        tracker.record_hit(similarity_score=0.95, latency_ms=5.0)
        prom = tracker.to_prometheus()
        assert "semanticache_requests_total 1" in prom
        assert "semanticache_hits_total 1" in prom
        assert "semanticache_latency_ms_bucket" in prom
        assert "# TYPE" in prom
        assert "# HELP" in prom

    def test_to_prometheus_empty(self) -> None:
        tracker = MetricsTracker()
        prom = tracker.to_prometheus()
        assert "semanticache_requests_total 0" in prom

    def test_log_structured(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        tracker = MetricsTracker()
        tracker.record_hit(similarity_score=0.95)
        with caplog.at_level(logging.INFO):
            tracker.log_structured("test_event")
        assert "test_event" in caplog.text
