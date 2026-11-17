"""The control plane's own Prometheus metrics, and the PromQL it runs for model metrics."""

import math

import httpx
from prometheus_client import Counter, Histogram

_BUCKETS = (1, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900, 1800)

ROLLOUT_SECONDS = Histogram(
    "icp_rollout_duration_seconds",
    "Time from rollout start to active or failed",
    ["backend", "result"],
    buckets=_BUCKETS,
)
PHASE_SECONDS = Histogram(
    "icp_rollout_phase_seconds",
    "Time spent in each rollout phase (queue, startup, gate)",
    ["backend", "phase"],
    buckets=_BUCKETS,
)
ROLLOUTS = Counter("icp_rollouts", "Finished rollouts by result", ["backend", "result"])
ROLLBACKS = Counter("icp_rollbacks", "Automatic rollbacks by failure reason", ["backend", "reason"])


def model_queries(name: str) -> dict[str, str]:
    """vLLM labels its metrics with the served model name, which is the deployment name."""
    sel = f'model_name="{name}"'  # name is validated as a DNS label, so this is safe to inline
    return {
        "queue_depth": f"sum(vllm:num_requests_waiting{{{sel}}})",
        "running": f"sum(vllm:num_requests_running{{{sel}}})",
        "tokens_per_second": f"sum(rate(vllm:generation_tokens_total{{{sel}}}[1m]))",
        "p99_latency_seconds": (
            f"histogram_quantile(0.99, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{{{sel}}}[5m])))"
        ),
        "p99_ttft_seconds": (
            f"histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{{{sel}}}[5m])))"
        ),
    }


def _number(value: str) -> float | None:
    number = float(value)
    return None if math.isnan(number) or math.isinf(number) else number


class PrometheusClient:
    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=10, transport=transport)

    async def __aenter__(self) -> "PrometheusClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def _result(self, path: str, params: dict) -> list[dict]:
        response = await self._client.get(path, params=params)
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "success":
            raise ValueError(body.get("error", "query failed"))
        return body["data"]["result"]

    async def query(self, promql: str) -> float | None:
        result = await self._result("/api/v1/query", {"query": promql})
        return _number(result[0]["value"][1]) if result else None

    async def query_range(self, promql: str, start: float, end: float, step: float) -> list[tuple[float, float | None]]:
        result = await self._result("/api/v1/query_range", {"query": promql, "start": start, "end": end, "step": step})
        return [(float(t), _number(v)) for t, v in result[0]["values"]] if result else []
