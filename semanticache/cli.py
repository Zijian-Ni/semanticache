"""CLI tool for SemantiCache — serve, stats, clear, benchmark."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError as _exc:
    raise ImportError(
        "CLI dependencies not installed. Install with: pip install semanticache[cli]"
    ) from _exc

app = typer.Typer(
    name="semanticache",
    help="SemantiCache CLI — semantic caching for LLMs.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8080, help="Port number"),
) -> None:
    """Start the SemantiCache metrics dashboard server."""
    try:
        import uvicorn

        from dashboard.app import create_app
    except ImportError:
        console.print(
            "[red]Dashboard dependencies missing.[/red] "
            "Install with: pip install semanticache[dashboard]"
        )
        raise typer.Exit(1)

    from semanticache import SemantiCache

    cache = SemantiCache()
    application = create_app(cache=cache)
    console.print(f"[green]Starting dashboard at http://{host}:{port}[/green]")
    uvicorn.run(application, host=host, port=port)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@app.command()
def stats(
    output_format: str = typer.Option("table", "--format", "-f", help="Output format: table|json"),
) -> None:
    """Show cache statistics from a running SemantiCache instance."""
    from semanticache.utils.metrics import MetricsTracker

    # Create a tracker and show its current state (demo/local mode)
    tracker = MetricsTracker()
    metrics = tracker.to_dict()

    if output_format == "json":
        console.print(json.dumps(metrics, indent=2))
    else:
        table = Table(title="SemantiCache Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Requests", str(metrics["total_requests"]))
        table.add_row("Cache Hits", str(metrics["cache_hits"]))
        table.add_row("Cache Misses", str(metrics["cache_misses"]))
        table.add_row("Hit Rate", f"{metrics['hit_rate']:.2%}")
        table.add_row("Tokens Saved", str(metrics["total_tokens_saved"]))
        table.add_row("Cost Saved (USD)", f"${metrics['cost_saved_usd']:.4f}")

        console.print(table)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


@app.command()
def clear(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Namespace to clear"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Clear the cache (in-memory only for local usage)."""
    if not yes:
        target = f"namespace '{namespace}'" if namespace else "ALL entries"
        confirm = typer.confirm(f"Clear {target}?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    from semanticache import SemantiCache

    cache = SemantiCache()
    count = asyncio.run(cache.clear(namespace=namespace))
    console.print(f"[green]Cleared {count} entries.[/green]")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    num_entries: int = typer.Option(1000, "--entries", "-n", help="Number of entries to test"),
    dim: int = typer.Option(384, "--dim", "-d", help="Embedding dimension"),
) -> None:
    """Run a performance benchmark on the in-memory backend."""
    import numpy as np

    from semanticache.backends.memory import InMemoryBackend

    console.print(f"[cyan]Benchmarking with {num_entries} entries (dim={dim})...[/cyan]")

    async def _run() -> dict[str, float]:
        backend = InMemoryBackend()
        rng = np.random.default_rng(42)

        # --- Store benchmark ---
        start = time.perf_counter()
        for i in range(num_entries):
            vec = rng.standard_normal(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            await backend.store(
                embedding=vec,
                response=f"Response {i}",
                namespace="bench",
                metadata={},
                ttl=3600,
            )
        store_time = time.perf_counter() - start

        # --- Search benchmark ---
        query = rng.standard_normal(dim).astype(np.float32)
        query = query / np.linalg.norm(query)

        start = time.perf_counter()
        search_iters = min(100, num_entries)
        for _ in range(search_iters):
            await backend.search(
                embedding=query,
                namespace="bench",
                threshold=0.90,
                ttl=3600,
            )
        search_time = time.perf_counter() - start

        return {
            "entries": num_entries,
            "dim": dim,
            "store_total_ms": round(store_time * 1000, 2),
            "store_per_entry_ms": round(store_time / num_entries * 1000, 4),
            "search_total_ms": round(search_time * 1000, 2),
            "search_avg_ms": round(search_time / search_iters * 1000, 4),
            "search_iterations": search_iters,
        }

    results = asyncio.run(_run())

    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Entries", str(results["entries"]))
    table.add_row("Dimension", str(results["dim"]))
    table.add_row("Store (total)", f"{results['store_total_ms']}ms")
    table.add_row("Store (per entry)", f"{results['store_per_entry_ms']}ms")
    table.add_row("Search (total)", f"{results['search_total_ms']}ms")
    table.add_row("Search (avg)", f"{results['search_avg_ms']}ms")
    table.add_row("Search iterations", str(results["search_iterations"]))

    console.print(table)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
