# Inference Control Plane

A control plane for deploying vLLM models, with a FastAPI service, a React UI and a Python
CLI. It runs models in two places behind one interface:

- **Kubernetes**: each revision is its own Deployment, with startup, readiness and liveness
  checks on vLLM's `/health`.
- **Slurm** (an HPC cluster such as Northeastern's Explorer): each revision is an `sbatch`
  job running vLLM in Apptainer on a GPU node. A gateway pod in Kubernetes reaches it over
  an SSH tunnel.

Rollouts are blue/green. A new revision starts next to the one that is serving. It only
takes traffic after it is ready and answers a test completion. If anything fails on the
way, the control plane names the reason (out of memory, CUDA OOM, gated model, bad
arguments, image pull, time limit...), tears the new revision down, and the old one keeps
serving. Prometheus collects vLLM's queue depth, token throughput and latency, plus the
control plane's own rollout timings.

```
Mac: kind cluster                                   HPC cluster: Slurm
  FastAPI + rollout engine ──ssh: sbatch/squeue/sacct/scancel──▶ login node
  React UI · Prometheus                                              │
  vLLM pods (Kubernetes backend)                                     ▼
  gateway pod + Service ──────────ssh tunnel─────────────▶ GPU node: vllm serve
  stable Service per deployment → active revision          (Apptainer job)
```

## Layout

| Path | What it is |
|---|---|
| `api/` | FastAPI service: API routes, rollout engine, Kubernetes and Slurm backends, failure classifier |
| `cli/` | `icp`, a Typer CLI: `deploy`, `status`, `history`, `events`, `logs`, `rollback`, `metrics`, `delete` |
| `web/` | React + Vite UI: deployments, revisions, rollout events, live logs, metrics |
| `fake-vllm/` | Stand-in for vLLM (same endpoints and metric names, no GPU) with failure injection |
| `gateway/` | Image for the SSH tunnel pod that fronts a Slurm job |
| `deploy/` | kind config, namespaces, RBAC, model cache, in-cluster API, Prometheus config |
| `hpc/` | One-time HPC setup: SSH key, Apptainer image, manual test |
| `scripts/` | Failure injection, rollout benchmark, load generator |

## Quick start (laptop, no GPU)

Needs Python 3.11, Node 20+, Docker, kind, kubectl and helm.

```bash
make setup          # .venv, Python packages, web dependencies, .env
make test           # unit and API tests, no cluster needed
make cluster-up     # kind cluster "icp" with namespaces, RBAC and the model cache
make images         # build the fake vLLM, gateway and API images and load them into kind
```

Then run these in separate terminals:

```bash
make proxy          # kubectl proxy, so the API on your Mac can reach Services in kind
make api            # API on :8080
make web            # UI on http://localhost:5173
```

Deploy something:

```bash
.venv/bin/icp deploy demo --model facebook/opt-125m
.venv/bin/icp deploy demo --arg=--max-model-len=2048         # new revision, blue/green
.venv/bin/icp deploy demo --env FAKE_VLLM_FAIL=cuda_oom       # fails, rolls back
.venv/bin/icp history demo
.venv/bin/icp rollback demo
.venv/bin/python scripts/inject_failures.py                   # every failure scenario
```

For metrics, run `make monitoring` (installs a trimmed kube-prometheus-stack) and
`make prom-forward`, then send traffic with `scripts/load_gen.py`. Prometheus starts
scraping the model pods a minute or two after install, once its config reloads.

The default image is the fake server. For real vLLM on CPU, pass an image that has
`vllm` on its path with `--image`, and give it room with `--cpu` and `--memory`.

## Real GPUs on the HPC cluster

See [hpc/README.md](hpc/README.md): set up key-based SSH, pull the vLLM image into
scratch, check a model by hand, then:

```bash
.venv/bin/icp deploy qwen --backend slurm --model Qwen/Qwen2.5-1.5B-Instruct \
  --partition gpu-short --gpu-type a100 --constraint a100@80g --time-limit 01:00:00
```

## Benchmarks

- [One H200, four models](benchmarks/2026-11-18-h200.md): Qwen2.5 1.5B / 7B / 32B and gpt-oss-120b benchmarked one after another on a single H200, from the GPU node, with p50/p90/p95/p99 request latency, time to first token and time per output token (`hpc/benchmark-models.sbatch`).
- [Through the control plane](benchmarks/2026-11-18-explorer.md): the same models deployed by the Slurm backend on A100s and an H200 and measured through the SSH gateway.

## How a rollout works

1. `POST /api/deployments/{name}/rollouts` stores a new revision with an immutable spec.
2. The backend starts it: a Deployment and Service on Kubernetes, or an `sbatch` job on Slurm.
3. The engine polls the backend. Phases are **pending** (waiting for resources), **starting**
   (model loading) and **ready**. A pod or job that fails is classified from its status and
   log. Time in pending and time starting each have their own timeout.
4. A one-token test completion goes to the new revision's own Service.
5. The deployment's stable Service is repointed at the new revision, and the old revision
   is torn down.

If any step fails, the new revision is torn down and the stable Service never moves. Each
revision records how long it spent in each phase (queue, startup, test, total). Those
are the numbers to quote, and `scripts/bench_rollout.py` collects them.

## Configuration

Everything is an `ICP_*` environment variable (or a line in `.env`); see
[api/src/icp_api/config.py](api/src/icp_api/config.py).

## Known gaps

- Rollouts run inside the API process. If it restarts mid-rollout, that revision is marked
  failed and may need cleaning up by hand.
- One API replica only. State is SQLite; moving to Postgres is a `ICP_DATABASE_URL` change.
- If Slurm requeues a job onto another node, the gateway is not repointed yet.
- Blue/green needs room for two copies during a rollout, on either backend.
- No authentication on the API.
