import asyncio
import sys
from pathlib import Path

import httpx

from fake_vllm.server import create_app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_gen import measure, percentiles  # noqa: E402


def run(stream: bool) -> dict:
    app = create_app("chat", "m", tokens_per_second=2000, max_concurrency=4)
    transport = httpx.ASGITransport(app=app)
    return asyncio.run(measure("http://fake", "chat", concurrency=4, duration=0.5, max_tokens=8, stream=stream,
                               transport=transport))  # fmt: skip


def test_percentiles():
    out = percentiles([float(i) for i in range(1, 101)])
    assert (out["p50_s"], out["p90_s"], out["p95_s"], out["p99_s"]) == (50.5, 90.1, 95.05, 99.01)
    assert percentiles([2.0]) == {"p50_s": 2.0, "p90_s": 2.0, "p95_s": 2.0, "p99_s": 2.0}
    assert percentiles([0.002, 0.004], "tpot_", scale=1000.0)["tpot_p50_ms"] == 3.0


def test_measure_without_streaming():
    summary = run(stream=False)
    assert summary["requests"] > 0 and summary["errors"] == 0
    assert summary["tokens_per_s"] > 0
    assert summary["p50_s"] <= summary["p90_s"] <= summary["p95_s"] <= summary["p99_s"]
    assert "ttft_p50_s" not in summary


def test_measure_with_streaming_reports_ttft_and_tpot():
    summary = run(stream=True)
    assert summary["requests"] > 0 and summary["errors"] == 0, summary["first_error"]
    assert summary["tokens_per_s"] > 0  # usage arrives in the final streamed chunk
    for prefix, unit in (("ttft_", "_s"), ("tpot_", "_ms")):
        values = [summary[f"{prefix}p{p}{unit}"] for p in (50, 90, 95, 99)]
        assert all(v is not None for v in values) and values == sorted(values)
    # httpx's in-process ASGI transport delivers a streamed body all at once, so here the first
    # token lands with the last one; against a real server TTFT is well below request latency.
