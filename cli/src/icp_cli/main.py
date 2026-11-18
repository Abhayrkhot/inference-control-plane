"""`icp`: deploy and operate vLLM models through the control plane API.

Set ICP_URL to point at the API (default http://localhost:8080).
"""

import copy
import os
import time
from datetime import datetime
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

app = typer.Typer(
    help="Deploy and operate vLLM models through the inference control plane.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

TERMINAL = {"active", "failed", "superseded", "stopped"}
EVENT_STYLES = {
    "failed": "red",
    "teardown_failed": "red",
    "retire_failed": "red",
    "rollback": "yellow",
    "status_error": "yellow",
    "ready": "green",
    "smoke_test_passed": "green",
    "activated": "bold green",
    "retired": "dim",
}


# --- helpers ---------------------------------------------------------------------------------


def api() -> httpx.Client:
    base = os.environ.get("ICP_URL", "http://localhost:8080").rstrip("/")
    return httpx.Client(base_url=f"{base}/api", timeout=30)


def check(response: httpx.Response) -> httpx.Response:
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        console.print(f"[red]error {response.status_code}:[/] {escape(str(detail))}")
        raise typer.Exit(1)
    return response


def parse_env(pairs: list[str]) -> dict[str, str]:
    env = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"expected KEY=VALUE, got {pair!r}", param_hint="--env")
        env[key] = value
    return env


def build_spec(base: dict[str, Any] | None, **flags: Any) -> dict[str, Any]:
    """Overlay CLI flags on the current spec, so `icp deploy` only changes what you pass."""
    spec = copy.deepcopy(base) if base else {}
    for key in ("model", "backend", "image", "replicas"):
        if flags.get(key) is not None:
            spec[key] = flags[key]
    if flags.get("clear_args"):
        spec["vllm_args"] = []
    if flags.get("args"):
        spec["vllm_args"] = list(flags["args"])
    if flags.get("env"):
        spec["env"] = {**spec.get("env", {}), **parse_env(flags["env"])}
    sections = {
        "resources": {"cpu": "cpu", "memory": "memory", "gpu": "gpu"},
        "slurm": {
            "partition": "partition",
            "gpus": "gpus",
            "gpu_type": "gpu_type",
            "constraint": "constraint",
            "cpus": "slurm_cpus",
            "mem": "slurm_mem",
            "time_limit": "time_limit",
        },
    }
    for section, fields in sections.items():
        given = {field: flags[flag] for field, flag in fields.items() if flags.get(flag) is not None}
        if given:
            spec[section] = {**spec.get(section, {}), **given}
    return spec


def seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}s"


def timing_summary(rev: dict[str, Any]) -> str:
    parts = [
        ("queue", rev.get("queue_seconds")),
        ("startup", rev.get("startup_seconds")),
        ("test", rev.get("gate_seconds")),
        ("total", rev.get("total_seconds")),
    ]
    return " · ".join(f"{name} {seconds(value)}" for name, value in parts if value is not None)


def print_event(event: dict[str, Any]) -> None:
    at = datetime.fromisoformat(event["at"].replace("Z", "+00:00")).astimezone().strftime("%H:%M:%S")
    style = EVENT_STYLES.get(event["kind"], "cyan")
    console.print(
        f"[dim]{at}[/] r{event['revision']} [{style}]{event['kind']:<18}[/] {escape(event['message'])}",
        highlight=False,
    )


def get_revision(c: httpx.Client, name: str, number: int) -> dict[str, Any]:
    return next(r for r in check(c.get(f"/deployments/{name}/revisions")).json() if r["number"] == number)


def follow_rollout(c: httpx.Client, name: str, number: int, poll: float = 1.0) -> int:
    """Print the rollout's events until it finishes. Returns a process exit code."""
    after = 0

    def drain() -> None:
        nonlocal after
        events = check(c.get(f"/deployments/{name}/events", params={"after": after, "revision": number})).json()
        for event in events:
            print_event(event)
            after = event["id"]

    while True:
        drain()
        if get_revision(c, name, number)["status"] in TERMINAL:
            break
        time.sleep(poll)
    time.sleep(poll)  # let the last events (retire, rollback) land
    drain()
    rev = get_revision(c, name, number)
    if rev["status"] == "failed":
        console.print(f"[red]✗ r{number} failed ({rev['failure_reason']})[/]  {timing_summary(rev)}")
        return 1
    console.print(f"[green]✓ r{number} is serving[/]  {timing_summary(rev)}")
    return 0


# --- commands --------------------------------------------------------------------------------


@app.command()
def deploy(
    name: Annotated[str, typer.Argument(help="Deployment name: lowercase letters, digits, dashes")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Hugging Face model id")] = None,
    backend: Annotated[str | None, typer.Option("--backend", "-b", help="kubernetes or slurm")] = None,
    image: Annotated[str | None, typer.Option(help="Container image (kubernetes) or .sif path (slurm)")] = None,
    args: Annotated[
        list[str] | None, typer.Option("--arg", help="vLLM argument, repeatable: --arg=--max-model-len=4096")
    ] = None,
    clear_args: Annotated[bool, typer.Option("--clear-args", help="Drop the current vLLM arguments")] = False,
    env: Annotated[list[str] | None, typer.Option("--env", "-e", help="KEY=VALUE, repeatable")] = None,
    replicas: Annotated[int | None, typer.Option(help="Kubernetes replicas")] = None,
    cpu: Annotated[str | None, typer.Option(help="Kubernetes CPU request, e.g. 2 or 500m")] = None,
    memory: Annotated[str | None, typer.Option(help="Kubernetes memory, e.g. 8Gi")] = None,
    gpu: Annotated[int | None, typer.Option(help="Kubernetes GPUs per pod")] = None,
    partition: Annotated[str | None, typer.Option(help="Slurm partition")] = None,
    gpus: Annotated[int | None, typer.Option(help="Slurm GPUs")] = None,
    gpu_type: Annotated[str | None, typer.Option(help="Slurm GPU type, e.g. a100")] = None,
    constraint: Annotated[str | None, typer.Option(help="Slurm node features, e.g. a100@80g")] = None,
    slurm_cpus: Annotated[int | None, typer.Option(help="Slurm CPUs per task")] = None,
    slurm_mem: Annotated[str | None, typer.Option(help="Slurm memory, e.g. 32G")] = None,
    time_limit: Annotated[str | None, typer.Option(help="Slurm time limit, e.g. 02:00:00")] = None,
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Follow the rollout until it finishes")] = True,
) -> None:
    """Create a deployment, or roll out a new revision of an existing one."""
    flags = dict(
        model=model, backend=backend, image=image, args=args, clear_args=clear_args, env=env,
        replicas=replicas, cpu=cpu, memory=memory, gpu=gpu, partition=partition, gpus=gpus,
        gpu_type=gpu_type, constraint=constraint, slurm_cpus=slurm_cpus, slurm_mem=slurm_mem, time_limit=time_limit,
    )  # fmt: skip
    with api() as c:
        existing = c.get(f"/deployments/{name}")
        if existing.status_code == 404:
            if not model:
                console.print("[red]--model is required for a new deployment[/]")
                raise typer.Exit(2)
            body = check(c.post("/deployments", json={"name": name, "spec": build_spec(None, **flags)})).json()
            number = body["latest_revision"]["number"]
        else:
            current = check(existing).json()
            base = current["active_revision"] or current["latest_revision"]
            spec = build_spec(base["spec"] if base else None, **flags)
            number = check(c.post(f"/deployments/{name}/rollouts", json=spec)).json()["number"]
        console.print(f"rollout started: [bold]{name} r{number}[/]")
        if wait:
            raise typer.Exit(follow_rollout(c, name, number))


@app.command()
def status(name: Annotated[str | None, typer.Argument(help="Show one deployment")] = None) -> None:
    """What each deployment is serving, and how its latest rollout went."""
    with api() as c:
        items = [check(c.get(f"/deployments/{name}")).json()] if name else check(c.get("/deployments")).json()
    table = Table("NAME", "SERVING", "BACKEND", "MODEL", "LATEST ROLLOUT", "REASON")
    for d in items:
        active, latest = d["active_revision"], d["latest_revision"]
        shown = active or latest or {}
        table.add_row(
            d["name"],
            f"r{active['number']}" if active else "[dim]nothing[/]",
            shown.get("backend", "-"),
            shown.get("spec", {}).get("model", "-"),
            f"r{latest['number']} {latest['status']}" if latest else "-",
            (latest or {}).get("failure_reason") or "",
        )
    console.print(table)


@app.command()
def history(name: str) -> None:
    """Every revision of a deployment, with rollout timings."""
    with api() as c:
        revisions = check(c.get(f"/deployments/{name}/revisions")).json()
    table = Table("REV", "STATUS", "BACKEND", "MODEL", "ARGS", "QUEUE", "STARTUP", "TOTAL", "REASON")
    for r in revisions:
        source = f" (from r{r['source_revision']})" if r["source_revision"] else ""
        table.add_row(
            f"r{r['number']}{source}",
            r["status"],
            r["backend"],
            r["spec"]["model"],
            " ".join(r["spec"]["vllm_args"]),
            seconds(r["queue_seconds"]),
            seconds(r["startup_seconds"]),
            seconds(r["total_seconds"]),
            r["failure_reason"] or "",
        )
    console.print(table)


@app.command()
def events(name: str, revision: Annotated[int | None, typer.Option("--revision", "-r")] = None) -> None:
    """The rollout event log."""
    params = {"revision": revision} if revision else {}
    with api() as c:
        for event in check(c.get(f"/deployments/{name}/events", params=params)).json():
            print_event(event)


@app.command()
def logs(
    name: str,
    follow: Annotated[bool, typer.Option("--follow", "-f")] = False,
    revision: Annotated[int | None, typer.Option("--revision", "-r", help="Default: the serving revision")] = None,
    tail: Annotated[int, typer.Option(help="Lines of history")] = 200,
) -> None:
    """Model server logs (pod logs on Kubernetes, the job output file on Slurm)."""
    params: dict[str, Any] = {"follow": follow, "tail": tail}
    if revision:
        params["revision"] = revision
    with api() as c, c.stream("GET", f"/deployments/{name}/logs", params=params, timeout=None) as response:
        if response.is_error:
            response.read()
            check(response)
        for chunk in response.iter_text():
            print(chunk, end="", flush=True)


@app.command()
def rollback(
    name: str,
    to: Annotated[int | None, typer.Option("--to", help="Revision to go back to (default: the previous one)")] = None,
    wait: Annotated[bool, typer.Option("--wait/--no-wait")] = True,
) -> None:
    """Roll out an earlier revision's spec again."""
    with api() as c:
        rev = check(c.post(f"/deployments/{name}/rollback", json={"revision": to})).json()
        console.print(f"rolling back: [bold]{name} r{rev['number']}[/] with the spec of r{rev['source_revision']}")
        if wait:
            raise typer.Exit(follow_rollout(c, name, rev["number"]))


@app.command()
def metrics(name: str) -> None:
    """Current queue depth, throughput and latency, from Prometheus."""
    with api() as c:
        data = check(c.get(f"/deployments/{name}/metrics")).json()
    current = data["current"]
    rows = [
        ("queue_depth", "Queue depth", "{:.0f}"),
        ("running", "Running requests", "{:.0f}"),
        ("tokens_per_second", "Generated tokens/s", "{:.1f}"),
    ]
    table = Table("METRIC", "VALUE")
    for key, label, fmt in rows:
        value = current.get(key)
        table.add_row(label, "-" if value is None else fmt.format(value))
    console.print(table)

    def cell(key: str) -> str:
        value = current.get(key)
        return "-" if value is None else f"{value * 1000:.0f} ms" if value < 1 else f"{value:.2f} s"

    latency = Table("LATENCY (last 5 min)", "p50", "p90", "p95", "p99")
    for kind, label in (("latency", "Request"), ("ttft", "Time to first token")):
        latency.add_row(label, *(cell(f"p{p}_{kind}_seconds") for p in (50, 90, 95, 99)))
    console.print(latency)


@app.command()
def stop(name: str) -> None:
    """Stop serving and free the GPU, keeping the deployment's history (rollback restarts it)."""
    with api() as c:
        check(c.post(f"/deployments/{name}/stop"))
    console.print(f"stopped {name}; its history is kept (`icp rollback {name}` serves it again)")


@app.command()
def delete(name: str, yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask")] = False) -> None:
    """Tear down a deployment and permanently forget its history (use `stop` to keep it)."""
    if not yes:
        typer.confirm(f"Delete {name}, tear down what it serves, and erase its history?", abort=True)
    with api() as c:
        check(c.delete(f"/deployments/{name}"))
    console.print(f"deleted {name}")


if __name__ == "__main__":
    app()
