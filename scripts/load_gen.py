"""Send concurrent completions to a deployment and report latency and throughput.

    kubectl -n icp-models port-forward svc/demo 8000:8000
    python scripts/load_gen.py --url http://localhost:8000 --model demo --concurrency 8 --duration 60

The model name is the deployment name (it is vLLM's --served-model-name). Every request
asks for exactly --max-tokens tokens (ignore_eos), so runs against different models do the
same amount of work. Add --json for one machine-readable line, e.g. to compare models.

Latency percentiles are measured on the client, per request, so they include the network
path to the model (for Slurm deployments, the SSH tunnel). With few requests, high
percentiles are just the slowest requests: p99 needs at least ~100 requests to mean much.
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx

PROMPT = "Explain, step by step, how a GPU schedules thousands of threads."
PERCENTILES = (50, 90, 95, 99)


def percentiles(samples: list[float], prefix: str = "", scale: float = 1.0, digits: int = 3) -> dict:
    """p50/p90/p95/p99 of samples (times scale), interpolated within the observed range."""
    suffix = "_s" if scale == 1.0 else "_ms"
    if len(samples) < 2:
        value = round(samples[0] * scale, digits) if samples else None
        return {f"{prefix}p{p}{suffix}": value for p in PERCENTILES}
    cuts = statistics.quantiles(samples, n=100, method="inclusive")
    return {f"{prefix}p{p}{suffix}": round(cuts[p - 1] * scale, digits) for p in PERCENTILES}


async def _request(client: httpx.AsyncClient, body: dict) -> tuple[float, float | None, int]:
    """One completion. Returns (latency, time to first token or None, generated tokens)."""
    started = time.perf_counter()
    if not body.get("stream"):
        response = await client.post("/v1/completions", json=body)
        response.raise_for_status()
        return time.perf_counter() - started, None, response.json().get("usage", {}).get("completion_tokens", 0)
    first_token, tokens = None, 0
    async with client.stream("POST", "/v1/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if first_token is None and any(c.get("text") for c in chunk.get("choices") or []):
                first_token = time.perf_counter() - started
            if chunk.get("usage"):
                tokens = chunk["usage"].get("completion_tokens", 0)
    return time.perf_counter() - started, first_token, tokens


async def _worker(client: httpx.AsyncClient, body: dict, deadline: float, results: dict) -> None:
    while time.monotonic() < deadline:
        try:
            latency, ttft, tokens = await _request(client, body)
        except (httpx.HTTPError, ValueError) as e:
            results["errors"].append(e)
            await asyncio.sleep(0.5)
            continue
        results["latencies"].append(latency)
        results["tokens"] += tokens
        if ttft is not None:
            results["ttfts"].append(ttft)
            if tokens > 1:  # time per output token after the first
                results["tpots"].append((latency - ttft) / (tokens - 1))


async def measure(
    url: str,
    model: str,
    concurrency: int,
    duration: float,
    max_tokens: int = 128,
    stream: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """Run one load test and return its summary (also used by scripts/benchmark.py).

    With stream=True each response is streamed, which adds per-request time to first token
    (ttft_*) and time per output token (tpot_*, in ms) to the summary.
    """
    body = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,  # vLLM extension: always generate max_tokens, for a fixed amount of work
    }
    if stream:
        body |= {"stream": True, "stream_options": {"include_usage": True}}
    results: dict = {"latencies": [], "ttfts": [], "tpots": [], "tokens": 0, "errors": []}
    started = time.monotonic()
    async with httpx.AsyncClient(base_url=url, timeout=300, transport=transport) as client:
        await asyncio.gather(*(_worker(client, body, started + duration, results) for _ in range(concurrency)))
    elapsed = time.monotonic() - started
    summary = {
        "model": model,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "seconds": round(elapsed, 1),
        "requests": len(results["latencies"]),
        "errors": len(results["errors"]),
        "first_error": repr(results["errors"][0]) if results["errors"] else None,
        "requests_per_s": round(len(results["latencies"]) / elapsed, 2),
        "tokens_per_s": round(results["tokens"] / elapsed, 1),
        **percentiles(results["latencies"]),
    }
    if stream:
        summary |= percentiles(results["ttfts"], "ttft_", digits=4)
        summary |= percentiles(results["tpots"], "tpot_", scale=1000.0, digits=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--stream", action="store_true", help="stream responses: adds time to first token")
    parser.add_argument("--json", action="store_true", help="print one JSON line")
    args = parser.parse_args()
    summary = asyncio.run(
        measure(args.url, args.model, args.concurrency, args.duration, args.max_tokens, stream=args.stream)
    )
    if args.json:
        print(json.dumps(summary))
        return
    print(
        f"{summary['requests']} ok, {summary['errors']} errors in {summary['seconds']}s at concurrency "
        f"{args.concurrency}: {summary['requests_per_s']} req/s, {summary['tokens_per_s']} generated tokens/s"
    )
    print("request latency:      " + "  ".join(f"p{p} {summary[f'p{p}_s']}s" for p in PERCENTILES))
    if args.stream:
        print("time to first token:  " + "  ".join(f"p{p} {summary[f'ttft_p{p}_s']}s" for p in PERCENTILES))
        print("time per output token: " + "  ".join(f"p{p} {summary[f'tpot_p{p}_ms']}ms" for p in PERCENTILES))
    if summary["first_error"]:
        print(f"first error: {summary['first_error']}")


if __name__ == "__main__":
    main()
