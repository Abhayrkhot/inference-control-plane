"""Send concurrent completions to a deployment so the metrics have traffic to show.

    kubectl -n icp-models port-forward svc/demo 8000:8000
    python scripts/load_gen.py --url http://localhost:8000 --model demo --concurrency 8 --duration 60

The model name is the deployment name (it is vLLM's --served-model-name).
"""

import argparse
import asyncio
import statistics
import time

import httpx


async def worker(client: httpx.AsyncClient, args: argparse.Namespace, deadline: float, latencies: list, errors: list):
    while time.monotonic() < deadline:
        started = time.perf_counter()
        try:
            response = await client.post(
                "/v1/completions",
                json={"model": args.model, "prompt": "Write a haiku about GPUs.", "max_tokens": args.max_tokens},
            )
            response.raise_for_status()
            latencies.append(time.perf_counter() - started)
        except httpx.HTTPError as e:
            errors.append(e)
            await asyncio.sleep(0.5)


async def run(args: argparse.Namespace) -> None:
    latencies: list[float] = []
    errors: list[Exception] = []
    deadline = time.monotonic() + args.duration
    async with httpx.AsyncClient(base_url=args.url, timeout=120) as client:
        await asyncio.gather(*(worker(client, args, deadline, latencies, errors) for _ in range(args.concurrency)))
    print(f"{len(latencies)} ok, {len(errors)} errors in {args.duration}s at concurrency {args.concurrency}")
    if len(latencies) >= 2:
        q = statistics.quantiles(latencies, n=100)
        print(f"client-side latency: p50 {q[49]:.3f}s  p99 {q[98]:.3f}s")
    if errors:
        print(f"first error: {errors[0]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=64)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
