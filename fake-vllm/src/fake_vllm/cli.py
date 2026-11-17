"""A `vllm serve MODEL [options]` look-alike, so one Kubernetes manifest runs either the real
vLLM image or this one. vLLM options it does not know are accepted and ignored.

Failure injection, for testing rollouts. Set FAKE_VLLM_FAIL to:
  cuda_oom, kv_cache, gated, bad_args, crash   print vLLM's error for that case and exit 1
  oom                                          allocate memory until the container is OOM-killed
  health                                       start, but never report healthy
  completions                                  report healthy, but fail every completion

Other knobs: FAKE_VLLM_STARTUP_DELAY (seconds before /health is 200, default 3),
FAKE_VLLM_TOKENS_PER_SECOND (default 100), FAKE_VLLM_MAX_CONCURRENCY (default 4).
"""

import argparse
import os
import sys
import time

import uvicorn

from fake_vllm.server import create_app

STARTUP_ERRORS = {
    "cuda_oom": "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. (fake-vllm)",
    "kv_cache": (
        "ValueError: The model's max seq len (32768) is larger than the maximum number of tokens "
        "that can be stored in KV cache (8192). (fake-vllm)"
    ),
    "gated": "huggingface_hub.errors.GatedRepoError: 401 Client Error. Cannot access gated repo. (fake-vllm)",
    "bad_args": "vllm serve: error: unrecognized arguments: --not-a-real-flag (fake-vllm)",
    "crash": "RuntimeError: engine core initialization failed (fake-vllm)",
}


def _eat_memory() -> None:
    chunks = []
    while True:
        chunks.append(bytearray(b"\x01") * (64 * 1024 * 1024))  # writes the pages, so RSS really grows
        time.sleep(0.05)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vllm")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="serve a (fake) model")
    serve.add_argument("model")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--served-model-name", default=None)
    args, ignored = parser.parse_known_args(argv)

    served = args.served_model_name or args.model
    fail = os.environ.get("FAKE_VLLM_FAIL", "")
    print(f"fake-vllm: serving {args.model} as {served!r} on {args.host}:{args.port}; ignoring {ignored}", flush=True)
    if fail in STARTUP_ERRORS:
        time.sleep(float(os.environ.get("FAKE_VLLM_EXIT_DELAY", "1")))
        print(STARTUP_ERRORS[fail], file=sys.stderr, flush=True)
        sys.exit(1)
    if fail == "oom":
        _eat_memory()

    app = create_app(
        served,
        args.model,
        startup_delay=float(os.environ.get("FAKE_VLLM_STARTUP_DELAY", "3")),
        fail=fail,
        max_concurrency=int(os.environ.get("FAKE_VLLM_MAX_CONCURRENCY", "4")),
        tokens_per_second=float(os.environ.get("FAKE_VLLM_TOKENS_PER_SECOND", "100")),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
