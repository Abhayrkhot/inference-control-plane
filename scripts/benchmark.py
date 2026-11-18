"""Benchmark deployments with identical load and record p50/p90/p95/p99 latencies.

    kubectl --context kind-icp -n icp-models port-forward svc/qwen32b 18004:8000 &
    python scripts/benchmark.py --target qwen32b=http://localhost:18004 --out benchmarks/run.jsonl

For each target and concurrency level it runs scripts/load_gen.py's test (fixed 128-token
requests), then reads vLLM's own request-latency and time-to-first-token percentiles from
Prometheus over the same window. Targets run one at a time so they don't compete for the
network path. Two kinds of percentile are recorded:

- client_pXX_s: per request, timed by this script; includes the network path to the model.
- vllm_latency_pXX_s / vllm_ttft_pXX_s: from vLLM's histograms; excludes the network path,
  but is interpolated within histogram buckets.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_gen import PERCENTILES, measure  # noqa: E402

HISTOGRAMS = {
    "latency": "vllm:e2e_request_latency_seconds_bucket",
    "ttft": "vllm:time_to_first_token_seconds_bucket",
}
SCRAPE_WAIT_S = 10  # let Prometheus scrape the end of each run


def prom_percentiles(prom_url: str, model: str, window_s: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    with httpx.Client(base_url=prom_url, timeout=15) as client:
        for kind, metric in HISTOGRAMS.items():
            for p in PERCENTILES:
                query = (
                    f'histogram_quantile({p / 100}, sum by (le) (rate({metric}{{model_name="{model}"}}[{window_s}s])))'
                )
                try:
                    result = client.get("/api/v1/query", params={"query": query}).json()["data"]["result"]
                    value = float(result[0]["value"][1]) if result else None
                    out[f"vllm_{kind}_p{p}_s"] = None if value is None or value != value else round(value, 4)
                except (httpx.HTTPError, KeyError, ValueError):
                    out[f"vllm_{kind}_p{p}_s"] = None
    return out


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", action="append", required=True, help="name=url, repeatable (name = deployment)")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--duration", type=int, nargs="+", default=[30, 60], help="seconds per concurrency level")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--label", action="append", default=[], help="key=value added to every row, e.g. gpu=A100")
    parser.add_argument("--out", type=Path, help="append JSON lines here")
    args = parser.parse_args()
    durations = (
        args.duration if len(args.duration) == len(args.concurrency) else [args.duration[0]] * len(args.concurrency)
    )
    labels = dict(item.split("=", 1) for item in args.label)

    rows = []
    for target in args.target:
        name, _, url = target.partition("=")
        for concurrency, duration in zip(args.concurrency, durations, strict=True):
            print(f"== {name}: {concurrency} client(s) for {duration}s", flush=True)
            started = time.time()
            row = asyncio.run(measure(url, name, concurrency, duration, args.max_tokens))
            for p in PERCENTILES:
                row[f"client_p{p}_s"] = row.pop(f"p{p}_s")
            time.sleep(SCRAPE_WAIT_S)
            window = int(time.time() - started) + 1
            row = {"deployment": row.pop("model"), **labels, **row, **prom_percentiles(args.prometheus, name, window)}
            rows.append(row)
            if args.out:
                with args.out.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    header = ["deployment", "clients", "req/s", "tokens/s", "errors"]
    for label in ("request", "TTFT"):
        header += [f"{label} p{p}" for p in PERCENTILES]
    print("\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in rows:
        cells = [r["deployment"], r["concurrency"], r["requests_per_s"], r["tokens_per_s"], r["errors"]]
        cells += [fmt(r[f"client_p{p}_s"]) for p in PERCENTILES]
        cells += [fmt(r[f"vllm_ttft_p{p}_s"]) for p in PERCENTILES]
        print("| " + " | ".join(str(c) for c in cells) + " |")
    return 1 if any(r["errors"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
