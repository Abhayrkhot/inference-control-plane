import asyncio
import time
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from icp_api.metrics import PrometheusClient, model_queries
from icp_api.routes.deployments import SessionDep, find_deployment

router = APIRouter(tags=["metrics"])
CHART_KEYS = ("queue_depth", "tokens_per_second", "p99_latency_seconds")


@router.get("/deployments/{name}/metrics")
async def deployment_metrics(
    name: str, request: Request, s: SessionDep, minutes: Annotated[int, Query(ge=1, le=1440)] = 15
) -> dict[str, Any]:
    """Current values and recent series for queue depth, token throughput and latency."""
    find_deployment(s, name)
    url = request.app.state.settings.prometheus_url
    queries = model_queries(name)
    end = time.time()
    step = max(5, minutes * 60 // 120)
    try:
        async with PrometheusClient(url) as prom:
            current = await asyncio.gather(*(prom.query(q) for q in queries.values()))
            series = await asyncio.gather(
                *(prom.query_range(queries[k], end - minutes * 60, end, step) for k in CHART_KEYS)
            )
    except (httpx.HTTPError, ValueError, KeyError) as e:
        raise HTTPException(503, f"could not query Prometheus at {url}: {e}") from e
    return {
        "queries": queries,
        "current": dict(zip(queries, current, strict=True)),
        "series": dict(zip(CHART_KEYS, series, strict=True)),
    }
