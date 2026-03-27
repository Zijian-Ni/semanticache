"""FastAPI dashboard server for SemantiCache metrics and monitoring."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Rate limiter (simple in-memory, per-IP, 60 req/min)
# ---------------------------------------------------------------------------

_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT = 60
_RATE_WINDOW = 60.0  # seconds


def _is_rate_limited(client_ip: str) -> bool:
    now = time.monotonic()
    timestamps = _rate_store[client_ip]
    # Evict expired entries
    _rate_store[client_ip] = [t for t in timestamps if now - t < _RATE_WINDOW]
    if len(_rate_store[client_ip]) >= _RATE_LIMIT:
        return True
    _rate_store[client_ip].append(now)
    return False


# ---------------------------------------------------------------------------
# Metrics provider protocol
# ---------------------------------------------------------------------------


class _DefaultMetrics:
    """Fallback when no SemantiCache instance is attached."""

    def get_metrics(self) -> dict[str, Any]:
        return {
            "hit_rate": 0.0,
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_size": 0,
            "cost_saved": 0.0,
            "avg_similarity": 0.0,
            "similarity_distribution": [],
        }

    def get_top_queries(self, limit: int = 10) -> list[dict[str, Any]]:
        return []


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    cache: Any | None = None,
    metrics_provider: Any | None = None,
) -> FastAPI:
    """Create the dashboard FastAPI application.

    Parameters
    ----------
    cache:
        A ``SemantiCache`` instance.  If provided, metrics are read directly
        from the cache object.
    metrics_provider:
        Any object that exposes ``get_metrics()`` and optionally
        ``get_top_queries(limit)``.  Takes precedence over *cache* when both
        are supplied.
    """

    provider = metrics_provider or cache or _DefaultMetrics()

    app = FastAPI(
        title="SemantiCache Dashboard",
        docs_url=None,
        redoc_url=None,
    )

    # CORS -------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate-limit middleware ---------------------------------------------------
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        if _is_rate_limited(client_ip):
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status_code=429,
            )
        return await call_next(request)

    # Static files -----------------------------------------------------------
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # WebSocket bookkeeping --------------------------------------------------
    connected_clients: list[WebSocket] = []

    async def _broadcast(data: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for ws in connected_clients:
            try:
                await ws.send_json(data)
            except Exception:
                stale.append(ws)
        for ws in stale:
            connected_clients.remove(ws)

    # Background broadcaster -------------------------------------------------
    async def _metrics_broadcaster() -> None:
        while True:
            await asyncio.sleep(2)
            metrics = provider.get_metrics()
            top_queries: list[dict[str, Any]] = []
            if hasattr(provider, "get_top_queries"):
                top_queries = provider.get_top_queries(limit=10)
            await _broadcast({"type": "update", "metrics": metrics, "top_queries": top_queries})

    @app.on_event("startup")
    async def _start_broadcaster() -> None:
        asyncio.create_task(_metrics_broadcaster())

    # Routes -----------------------------------------------------------------

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"), media_type="text/html")

    @app.get("/api/metrics")
    async def api_metrics() -> dict[str, Any]:
        return provider.get_metrics()

    @app.get("/api/top-queries")
    async def api_top_queries(limit: int = 10) -> list[dict[str, Any]]:
        if hasattr(provider, "get_top_queries"):
            return provider.get_top_queries(limit=limit)
        return []

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        connected_clients.append(ws)
        try:
            # Send initial payload immediately
            metrics = provider.get_metrics()
            top_queries: list[dict[str, Any]] = []
            if hasattr(provider, "get_top_queries"):
                top_queries = provider.get_top_queries(limit=10)
            await ws.send_json({"type": "init", "metrics": metrics, "top_queries": top_queries})
            # Keep alive  receive loop
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            if ws in connected_clients:
                connected_clients.remove(ws)

    return app


# Allow ``uvicorn dashboard.app:app`` for quick development
app = create_app()
