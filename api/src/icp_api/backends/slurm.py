"""Slurm backend. Each revision is an sbatch job running vLLM in Apptainer on a GPU node.

Compute nodes are not reachable from outside the cluster, so once the job publishes its
node:port, a small gateway pod in Kubernetes opens an SSH tunnel to it through the login
node. The gateway carries the same labels as a Kubernetes revision, so readiness checks,
the stable Service and Prometheus treat a Slurm revision like any other.
"""

import shlex
from typing import Any

import asyncssh
import jinja2

from icp_api.backends.base import BackendStatus, DeployError, Phase, RevisionRef
from icp_api.backends.kube import KubeOps
from icp_api.backends.kubernetes import (
    PORT,
    crash_logs,
    evaluate,
    http_probe,
    render_revision_service,
    render_stable_service,
    revision_labels,
    revision_selector,
    selector_string,
    service_url,
)
from icp_api.backends.ssh import Runner
from icp_api.config import Settings
from icp_api.failures import (
    FATAL_PENDING_REASONS,
    LIMIT_PENDING_REASONS,
    FailureReason,
    classify_sbatch_error,
    classify_slurm,
    match_log,
    slurm_state_name,
)

q = shlex.quote
_templates = jinja2.Environment(
    loader=jinja2.PackageLoader("icp_api.backends", "templates"),
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=True,
    autoescape=False,  # shell script, not HTML; values are shell-quoted with the `q` filter
)
_templates.filters["q"] = shlex.quote


def run_dir(settings: Settings, rev: RevisionRef) -> str:
    return f"{settings.slurm_root}/runs/{rev.deployment}/r{rev.number}"


def render_sbatch(rev: RevisionRef, settings: Settings) -> str:
    root = settings.slurm_root
    return _templates.get_template("vllm.sbatch.j2").render(
        job_name=f"icp-{rev.object_name}",
        deployment=rev.deployment,
        number=rev.number,
        slurm=rev.spec.slurm,
        run_dir=run_dir(settings, rev),
        root=root,
        sif_path=rev.spec.image or f"{root}/vllm.sif",
        hf_cache=f"{root}/hf/hub",
        module_load=settings.slurm_module_load,
        cuda_compat=settings.slurm_cuda_compat,
        env=sorted(rev.spec.env.items()),
        model=rev.spec.model,
        vllm_args=rev.spec.vllm_args,
    )


def render_gateway(rev: RevisionRef, target_host: str, target_port: int, settings: Settings) -> list[dict[str, Any]]:
    labels = revision_labels(rev, "slurm")
    container = {
        "name": "tunnel",
        "image": settings.gateway_image,
        "imagePullPolicy": "IfNotPresent",
        "env": [
            {"name": "TARGET_HOST", "value": target_host},
            {"name": "TARGET_PORT", "value": str(target_port)},
            {"name": "LOGIN_HOST", "value": settings.slurm_login_host},
            {"name": "SSH_USER", "value": settings.slurm_ssh_user},
        ],
        "ports": [{"name": "http", "containerPort": PORT}],
        # Checks vLLM's /health through the tunnel, so "ready" means the model answers.
        "readinessProbe": http_probe(5, 3),
        "resources": {"requests": {"cpu": "50m", "memory": "32Mi"}, "limits": {"memory": "64Mi"}},
        "volumeMounts": [{"name": "ssh", "mountPath": "/etc/icp-ssh", "readOnly": True}],
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": rev.object_name,
            "labels": labels,
            "annotations": {"icp/slurm-job": rev.handle or "", "icp/target": f"{target_host}:{target_port}"},
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": revision_selector(rev)},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [container],
                    "volumes": [
                        {"name": "ssh", "secret": {"secretName": settings.gateway_ssh_secret, "defaultMode": 0o400}}
                    ],
                },
            },
        },
    }
    return [deployment, render_revision_service(rev, labels)]


def parse_squeue(output: str) -> tuple[str, str, str] | None:
    """Parse `squeue -h -j ID -o '%T|%r|%N'` into (state, reason, nodes)."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    state, reason, nodes = (lines[0].split("|") + ["", ""])[:3]
    return state, reason, nodes


def parse_sacct(output: str) -> str | None:
    """First state from `sacct -n -P -X -j ID -o State`."""
    return next((line.strip() for line in output.splitlines() if line.strip()), None)


class SlurmBackend:
    name = "slurm"

    def __init__(self, settings: Settings, kube: KubeOps, runner: Runner):
        self.settings = settings
        self.kube = kube
        self.runner = runner

    def _log_path(self, rev: RevisionRef) -> str:
        return f"{run_dir(self.settings, rev)}/slurm-{rev.handle}.out"

    async def deploy(self, rev: RevisionRef) -> str:
        for setting in ("slurm_ssh_user", "slurm_ssh_key_path"):
            if not getattr(self.settings, setting):
                raise DeployError(FailureReason.BACKEND_ERROR, f"ICP_{setting.upper()} is not set")
        d = q(run_dir(self.settings, rev))
        try:
            result = await self.runner.run(
                f"mkdir -p {d} && rm -f {d}/endpoint && cat > {d}/job.sbatch && sbatch --parsable {d}/job.sbatch",
                input=render_sbatch(rev, self.settings),
            )
        except (OSError, asyncssh.Error) as e:
            raise DeployError(FailureReason.SSH, f"cannot reach {self.settings.slurm_login_host}: {e}") from e
        job_id = result.stdout.strip().split(";")[0]  # --parsable prints "jobid[;cluster]"
        if result.exit_status != 0 or not job_id.isdigit():
            message = result.stderr.strip() or result.stdout.strip() or "sbatch failed"
            raise DeployError(classify_sbatch_error(message), message)
        return job_id

    async def status(self, rev: RevisionRef) -> BackendStatus:
        job = rev.handle or ""
        result = await self.runner.run(f"squeue -h -j {q(job)} -o '%T|%r|%N'")
        queued = parse_squeue(result.stdout) if result.exit_status == 0 else None
        if queued:
            state, reason, nodes = queued
            if state in ("PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED"):
                return self._pending(job, reason)
            if state == "RUNNING":
                return await self._running(rev, nodes)
        # Not in the queue any more (or COMPLETING): ask the accounting database.
        return await self._finished(rev)

    def _pending(self, job: str, reason: str) -> BackendStatus:
        if reason in FATAL_PENDING_REASONS:
            return BackendStatus(
                Phase.FAILED, f"job {job} can never start: {reason}", FailureReason.UNSCHEDULABLE, source=job
            )
        message = f"job {job} queued ({reason or 'no reason given'})"
        if reason in LIMIT_PENDING_REASONS:
            message += "; you are at a per-user limit, possibly held by the revision serving now"
        return BackendStatus(Phase.PENDING, message, source=job)

    async def _running(self, rev: RevisionRef, nodes: str) -> BackendStatus:
        job = rev.handle or ""
        result = await self.runner.run(f"cat {q(run_dir(self.settings, rev))}/endpoint")
        endpoint = result.stdout.strip()
        host, _, port = endpoint.rpartition(":")
        if result.exit_status != 0 or not host or not port.isdigit():
            return BackendStatus(
                Phase.STARTING, f"job {job} running on {nodes}; waiting for it to publish its port", source=job
            )

        deployment, pods = await self.kube.read(rev.object_name, selector_string(rev))
        if deployment is None:
            for manifest in render_gateway(rev, host, int(port), self.settings):
                await self.kube.apply(manifest)
            return BackendStatus(Phase.STARTING, f"job {job} on {endpoint}; starting gateway", source=job)
        # TODO: if Slurm requeues the job onto another node, re-point the gateway.

        gateway = evaluate(deployment, pods, desired=1)
        if gateway.phase is Phase.FAILED:
            found = match_log(await crash_logs(self.kube, gateway.source)) if gateway.source else None
            if found:
                gateway.failure, line = found
                gateway.message = line
            gateway.message = f"gateway {gateway.source}: {gateway.message}"
            return gateway
        if gateway.phase is Phase.READY:
            return BackendStatus(
                Phase.READY,
                f"job {job} on {endpoint}, reachable through the gateway",
                endpoint=service_url(self.settings, rev.object_name),
                source=job,
            )
        return BackendStatus(Phase.STARTING, f"job {job} on {endpoint}; model loading", source=job)

    async def _finished(self, rev: RevisionRef) -> BackendStatus:
        job = rev.handle or ""
        acct = await self.runner.run(f"sacct -n -P -X -j {q(job)} -o State")
        state = parse_sacct(acct.stdout) or "UNKNOWN"
        base = slurm_state_name(state)
        if base == "PENDING":  # squeue hiccup; the job is still queued
            return self._pending(job, "")
        if base == "RUNNING":
            return await self._running(rev, "?")
        tail = await self.runner.run(f"tail -n 200 {q(self._log_path(rev))}")
        reason, line = classify_slurm(state, tail.stdout)
        message = f"job {job} ended: {state}" + (f" | {line}" if line else "")
        return BackendStatus(Phase.FAILED, message, reason, source=job)

    async def activate(self, rev: RevisionRef) -> None:
        await self.kube.apply(render_stable_service(rev))

    async def deactivate(self, deployment: str) -> None:
        await self.kube.delete_service(deployment)

    async def teardown(self, rev: RevisionRef) -> None:
        if rev.handle:
            await self.runner.run(f"scancel {q(rev.handle)}")
        await self.kube.delete_selected(selector_string(rev))

    async def logs(self, rev: RevisionRef, follow: bool = False, tail: int = 200):
        if not rev.handle:
            yield "[icp] this revision has no Slurm job yet\n"
            return
        flag = " -F" if follow else ""
        async for line in self.runner.stream(f"tail -n {int(tail)}{flag} {q(self._log_path(rev))}"):
            yield line
