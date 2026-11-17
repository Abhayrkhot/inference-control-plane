"""Roll out known-bad revisions on top of a healthy fake-vLLM deployment, and check that each
one is caught, given the right failure reason, and rolled back while the healthy revision
keeps serving.

    python scripts/inject_failures.py                 # all scenarios
    python scripts/inject_failures.py --only gated,oom

Needs the API running against a cluster that has the icp-fake-vllm:dev image.
"""

import argparse
import os
import sys
import time

import httpx

# name -> (spec overrides, expected failure reason)
SCENARIOS: dict[str, tuple[dict, str]] = {
    "cuda_oom": ({"env": {"FAKE_VLLM_FAIL": "cuda_oom"}}, "gpu_oom"),
    "kv_cache": ({"env": {"FAKE_VLLM_FAIL": "kv_cache"}}, "gpu_oom"),
    "gated_model": ({"env": {"FAKE_VLLM_FAIL": "gated"}}, "model_access"),
    "bad_args": ({"env": {"FAKE_VLLM_FAIL": "bad_args"}}, "bad_args"),
    "crash": ({"env": {"FAKE_VLLM_FAIL": "crash"}}, "crash_loop"),
    "oom_killed": ({"env": {"FAKE_VLLM_FAIL": "oom"}, "resources": {"memory": "256Mi"}}, "oom_killed"),
    "bad_image": ({"image": "icp-image-that-does-not-exist:dev"}, "image_pull"),
    "smoke_test": ({"env": {"FAKE_VLLM_FAIL": "completions"}}, "smoke_test"),
}
HEALTHY = {"model": "facebook/opt-125m", "backend": "kubernetes"}


def wait(client: httpx.Client, name: str, number: int, timeout: float = 300) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rev = next(r for r in client.get(f"/deployments/{name}/revisions").json() if r["number"] == number)
        if rev["status"] in ("active", "failed", "superseded") and rev["total_seconds"] is not None:
            return rev
        time.sleep(1)
    raise TimeoutError(f"r{number} did not finish in {timeout:.0f}s")


def active_number(client: httpx.Client, name: str) -> int | None:
    active = client.get(f"/deployments/{name}").json()["active_revision"]
    return active["number"] if active else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="faults")
    parser.add_argument("--only", help="comma-separated scenario names")
    args = parser.parse_args()
    chosen = args.only.split(",") if args.only else list(SCENARIOS)

    base = os.environ.get("ICP_URL", "http://localhost:8080").rstrip("/")
    with httpx.Client(base_url=f"{base}/api", timeout=30) as client:
        if client.get(f"/deployments/{args.name}").status_code == 404:
            client.post("/deployments", json={"name": args.name, "spec": HEALTHY}).raise_for_status()
            rev = wait(client, args.name, 1)
            if rev["status"] != "active":
                print(f"healthy baseline failed: {rev['failure_reason']}: {rev['message']}")
                return 1
        serving = active_number(client, args.name)
        if serving is None:
            print(f"{args.name} has nothing serving; fix or delete it first")
            return 1

        failures = 0
        print(f"{'scenario':<14} {'expected':<14} {'got':<14} {'rollback':<9} {'time':>7}  detail")
        for name in chosen:
            overrides, expected = SCENARIOS[name]
            response = client.post(f"/deployments/{args.name}/rollouts", json=HEALTHY | overrides)
            response.raise_for_status()
            rev = wait(client, args.name, response.json()["number"])
            still_serving = active_number(client, args.name) == serving
            ok = rev["status"] == "failed" and rev["failure_reason"] == expected and still_serving
            failures += not ok
            print(
                f"{name:<14} {expected:<14} {rev['failure_reason'] or rev['status']:<14} "
                f"{'ok' if still_serving else 'BROKEN':<9} {rev['total_seconds']:>6.1f}s  "
                f"{'PASS' if ok else 'FAIL'}  {(rev['message'] or '')[:70]}"
            )
    print(f"\n{len(chosen) - failures}/{len(chosen)} scenarios behaved as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
