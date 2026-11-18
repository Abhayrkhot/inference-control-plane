"""Send concurrent completions to a deployment and report latency and throughput.

    kubectl -n icp-models port-forward svc/demo 8000:8000
    python scripts/load_gen.py --url http://localhost:8000 --model demo --concurrency 8 --duration 60

The model name is the deployment name (it is vLLM's --served-model-name). Every request
asks for exactly --max-tokens tokens (ignore_eos), so runs against different models do the
same amount of work. Add --json for one machine-readable line, e.g. to compare models.
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx

PROMPT = "Explain, step by step, how a GPU schedules thousands of threads."


async def worker(client: httpx.AsyncClient, args: argparse.Namespace, deadline: float, results: dict) -> None:
    body = {
        "model": args.model,
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,  # vLLM extension: always generate max_tokens, for a fixed amount of work
    }
    while time.monotonic() < deadline:
        started = time.perf_counter()
        try:
            response = await client.post("/v1/completions", json=body)
            response.raise_for_status()
            results["latencies"].append(time.perf_counter() - started)
            results["tokens"] += response.json().get("usage", {}).get("completion_tokens", 0)
        except (httpx.HTTPError, ValueError) as e:
            results["errors"].append(e)
            await asyncio.sleep(0.5)


async def run(args: argparse.Namespace) -> None:
    results: dict = {"latencies": [], "tokens": 0, "errors": []}
    started = time.monotonic()
    deadline = started + args.duration
    async with httpx.AsyncClient(base_url=args.url, timeout=300) as client:
        await asyncio.gather(*(worker(client, args, deadline, results) for _ in range(args.concurrency)))
    elapsed = time.monotonic() - started
    lat = sorted(results["latencies"])
    q = statistics.quantiles(lat, n=100) if len(lat) >= 2 else [float("nan")] * 99
    summary = {
        "model": args.model,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "seconds": round(elapsed, 1),
        "requests": len(lat),
        "errors": len(results["errors"]),
        "requests_per_s": round(len(lat) / elapsed, 2),
        "tokens_per_s": round(results["tokens"] / elapsed, 1),
        "p50_s": round(q[49], 3),
        "p99_s": round(q[98], 3),
    }
    if args.json:
        print(json.dumps(summary))
    else:
        print(
            f"{summary['requests']} ok, {summary['errors']} errors in {summary['seconds']}s at concurrency "
            f"{args.concurrency}: {summary['requests_per_s']} req/s, {summary['tokens_per_s']} generated tokens/s"
        )
        print(f"client-side latency: p50 {summary['p50_s']:.3f}s  p99 {summary['p99_s']:.3f}s")
    if results["errors"]:
        print(f"first error: {results['errors'][0]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--json", action="store_true", help="print one JSON line")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
