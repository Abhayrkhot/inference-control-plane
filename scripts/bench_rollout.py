"""Roll out an existing deployment N times and report the timings the control plane measured.
Each run sets a different env var, so every run is a real new revision.

    python scripts/bench_rollout.py demo --runs 5

Use this before and after a change (cached weights, preloaded images, probe tuning) to get
numbers you can stand behind.
"""

import argparse
import os
import statistics
import sys
import time

import httpx

PHASES = ("queue_seconds", "startup_seconds", "gate_seconds", "total_seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    base = os.environ.get("ICP_URL", "http://localhost:8080").rstrip("/")
    results: list[dict] = []
    with httpx.Client(base_url=f"{base}/api", timeout=30) as client:
        current = client.get(f"/deployments/{args.name}").raise_for_status().json()
        spec = (current["active_revision"] or current["latest_revision"])["spec"]
        for run in range(args.runs):
            body = spec | {"env": spec.get("env", {}) | {"ICP_BENCH_RUN": f"{time.time():.0f}-{run}"}}
            number = client.post(f"/deployments/{args.name}/rollouts", json=body).raise_for_status().json()["number"]
            while True:
                rev = next(r for r in client.get(f"/deployments/{args.name}/revisions").json() if r["number"] == number)
                if rev["status"] in ("active", "failed") and rev["total_seconds"] is not None:
                    break
                time.sleep(1)
            results.append(rev)
            cells = "  ".join(f"{(rev[p] or 0):7.1f}" for p in PHASES)
            print(f"r{number:<4} {rev['status']:<7} {cells}")

    ok = [r for r in results if r["status"] == "active"]
    if not ok:
        print("no successful rollouts")
        return 1
    print(f"\nmedian over {len(ok)} successful rollouts (seconds):")
    for phase in PHASES:
        values = [r[phase] for r in ok if r[phase] is not None]
        if values:
            print(f"  {phase.removesuffix('_seconds'):<8} {statistics.median(values):7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
