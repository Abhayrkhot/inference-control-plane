"""A stand-in for vLLM's OpenAI-compatible server: the same endpoints and metric names,
with simulated latency and queueing, and no GPU."""

import asyncio
import random
import time
import uuid

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

# Close to vLLM's own buckets, so quantiles interpolate the way they would on a real server.
_E2E_BUCKETS = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0)
_TTFT_BUCKETS = (0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0)


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str] = ""
    max_tokens: int = 16


class ChatRequest(BaseModel):
    model: str
    messages: list[dict] = []
    max_tokens: int = 16


def create_app(
    served_name: str,
    model: str,
    startup_delay: float = 0.0,
    fail: str = "",
    max_concurrency: int = 4,
    tokens_per_second: float = 100.0,
) -> FastAPI:
    registry = CollectorRegistry()
    labels = ["model_name"]

    def metric(kind, name: str, doc: str, **kwargs):
        return kind(name, doc, labels, registry=registry, **kwargs).labels(served_name)

    running = metric(Gauge, "vllm:num_requests_running", "Requests currently running")
    waiting = metric(Gauge, "vllm:num_requests_waiting", "Requests waiting for a slot")
    generation_tokens = metric(Counter, "vllm:generation_tokens", "Generated tokens")
    prompt_tokens = metric(Counter, "vllm:prompt_tokens", "Prompt tokens")
    e2e = metric(Histogram, "vllm:e2e_request_latency_seconds", "Request latency", buckets=_E2E_BUCKETS)
    ttft = metric(Histogram, "vllm:time_to_first_token_seconds", "Time to first token", buckets=_TTFT_BUCKETS)

    started = time.monotonic()
    slots = asyncio.Semaphore(max_concurrency)
    app = FastAPI(title="fake-vllm")

    @app.get("/health")
    def health() -> Response:
        healthy = fail != "health" and time.monotonic() - started >= startup_delay
        return Response(status_code=200 if healthy else 503)

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": served_name, "object": "model", "root": model}]}

    async def generate(requested: str, prompt_len: int, max_tokens: int) -> tuple[str, int]:
        if requested != served_name:
            raise HTTPException(404, f"The model `{requested}` does not exist.")
        if fail == "completions":
            raise HTTPException(500, "fake-vllm: injected completion failure")
        t0 = time.perf_counter()
        waiting.inc()
        try:
            await slots.acquire()
        finally:
            waiting.dec()
        running.inc()
        try:
            n = max(1, min(max_tokens, 256))
            await asyncio.sleep(0.01 + random.random() * 0.02)  # prefill
            ttft.observe(time.perf_counter() - t0)
            await asyncio.sleep(n / tokens_per_second)  # decode
        finally:
            running.dec()
            slots.release()
        e2e.observe(time.perf_counter() - t0)
        generation_tokens.inc(n)
        prompt_tokens.inc(prompt_len)
        return " ".join(["lorem"] * n), n

    def usage(prompt_len: int, n: int) -> dict:
        return {"prompt_tokens": prompt_len, "completion_tokens": n, "total_tokens": prompt_len + n}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> dict:
        prompt = req.prompt if isinstance(req.prompt, str) else " ".join(req.prompt)
        prompt_len = len(prompt.split())
        text, n = await generate(req.model, prompt_len, req.max_tokens)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": served_name,
            "choices": [{"index": 0, "text": text, "finish_reason": "length", "logprobs": None}],
            "usage": usage(prompt_len, n),
        }

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest) -> dict:
        prompt_len = sum(len(str(m.get("content", "")).split()) for m in req.messages)
        text, n = await generate(req.model, prompt_len, req.max_tokens)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}],
            "usage": usage(prompt_len, n),
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app
