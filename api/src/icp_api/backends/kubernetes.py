"""Kubernetes backend. Each revision gets its own Deployment and Service; a stable Service
named after the deployment points at whichever revision is active (blue/green)."""

from typing import Any

from icp_api.backends.base import BackendStatus, Phase, RevisionRef
from icp_api.backends.kube import KubeOps
from icp_api.config import Settings
from icp_api.failures import FailureReason, classify_pod, match_log

PORT = 8000


def revision_labels(rev: RevisionRef, backend: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/managed-by": "icp",
        "icp/managed": "true",  # Prometheus PodMonitor selects on this
        "icp/backend": backend,
        "icp/deployment": rev.deployment,
        "icp/revision": str(rev.number),
    }


def revision_selector(rev: RevisionRef) -> dict[str, str]:
    return {"icp/deployment": rev.deployment, "icp/revision": str(rev.number)}


def selector_string(rev: RevisionRef) -> str:
    return ",".join(f"{k}={v}" for k, v in revision_selector(rev).items())


def service_url(settings: Settings, service: str) -> str:
    return settings.service_url_template.format(service=service, namespace=settings.namespace)


def http_probe(period: int, failure_threshold: int) -> dict[str, Any]:
    return {
        "httpGet": {"path": "/health", "port": "http"},
        "periodSeconds": period,
        "failureThreshold": failure_threshold,
    }


def _service(name: str, labels: dict[str, str], selector: dict[str, str], **metadata: Any) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "labels": labels, **metadata},
        "spec": {"selector": selector, "ports": [{"name": "http", "port": PORT, "targetPort": "http"}]},
    }


def render_revision_service(rev: RevisionRef, labels: dict[str, str]) -> dict[str, Any]:
    """Reaches only this revision's pods. The rollout's test completion goes through it."""
    return _service(rev.object_name, labels, revision_selector(rev))


def render_stable_service(rev: RevisionRef) -> dict[str, Any]:
    """The Service clients use. Activating a revision repoints its selector."""
    labels = {"app.kubernetes.io/managed-by": "icp", "icp/deployment": rev.deployment, "icp/role": "stable"}
    return _service(
        rev.deployment,
        labels,
        revision_selector(rev),
        annotations={"icp/active-revision": str(rev.number)},
    )


def render_manifests(rev: RevisionRef, settings: Settings) -> list[dict[str, Any]]:
    spec = rev.spec
    labels = revision_labels(rev, "kubernetes")
    env: list[dict[str, Any]] = [{"name": "HF_HOME", "value": "/models/hf"}]
    env += [{"name": k, "value": v} for k, v in sorted(spec.env.items())]
    env.append(
        {
            "name": "HF_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": settings.hf_token_secret, "key": "token", "optional": True}},
        }
    )
    requests = {"cpu": spec.resources.cpu, "memory": spec.resources.memory}
    limits = {"memory": spec.resources.memory}
    if spec.resources.gpu:
        limits["nvidia.com/gpu"] = str(spec.resources.gpu)
    # The startup probe outlasts the engine's startup timeout, so the engine (not the kubelet)
    # decides when loading has taken too long and records why.
    startup_threshold = int(settings.startup_timeout_s // 5) + 12

    container = {
        "name": "vllm",
        "image": spec.image or settings.default_image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["vllm", "serve"],
        "args": [
            spec.model,
            "--served-model-name",
            rev.deployment,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            *spec.vllm_args,
        ],
        "ports": [{"name": "http", "containerPort": PORT}],
        "env": env,
        "resources": {"requests": requests, "limits": limits},
        "startupProbe": http_probe(5, startup_threshold),
        "readinessProbe": http_probe(5, 3),
        "livenessProbe": http_probe(10, 6),
        "volumeMounts": [
            {"name": "model-cache", "mountPath": "/models"},
            {"name": "shm", "mountPath": "/dev/shm"},
        ],
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": rev.object_name, "labels": labels},
        "spec": {
            "replicas": spec.replicas,
            "selector": {"matchLabels": revision_selector(rev)},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [container],
                    "volumes": [
                        {"name": "model-cache", "persistentVolumeClaim": {"claimName": settings.model_cache_pvc}},
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}},
                    ],
                },
            },
        },
    }
    return [deployment, render_revision_service(rev, labels)]


def _unscheduled(pod: dict[str, Any]) -> str | None:
    for cond in (pod.get("status") or {}).get("conditions") or []:
        if cond.get("type") == "PodScheduled" and cond.get("status") == "False":
            return cond.get("message") or cond.get("reason") or "pod not scheduled"
    return None


def evaluate(deployment: dict[str, Any] | None, pods: list[dict[str, Any]], desired: int) -> BackendStatus:
    """Decide a revision's phase from its Deployment and pods (camelCase API dicts)."""
    if deployment is None:
        return BackendStatus(Phase.FAILED, "Deployment object is missing", FailureReason.BACKEND_ERROR)
    for pod in pods:
        found = classify_pod(pod)
        if found:
            reason, message = found
            return BackendStatus(Phase.FAILED, message, reason, source=pod["metadata"]["name"])
    status = deployment.get("status") or {}
    ready = status.get("readyReplicas") or 0
    observed = (status.get("observedGeneration") or 0) >= (deployment["metadata"].get("generation") or 0)
    if observed and ready >= desired:
        return BackendStatus(Phase.READY, f"{ready}/{desired} replicas ready")
    unscheduled = [msg for msg in map(_unscheduled, pods) if msg]
    if not pods or unscheduled:
        return BackendStatus(Phase.PENDING, unscheduled[0] if unscheduled else "waiting for pods")
    return BackendStatus(Phase.STARTING, f"{ready}/{desired} replicas ready; model loading")


async def crash_logs(kube: KubeOps, pod: str) -> str:
    """Logs of the crashed container. Read both: right after a restart the API can answer the
    `previous` request with "unable to retrieve container logs", and the current container
    may itself be the one that just crashed."""
    return await kube.pod_log(pod, previous=True) + "\n" + await kube.pod_log(pod)


class KubernetesBackend:
    name = "kubernetes"

    def __init__(self, settings: Settings, kube: KubeOps):
        self.settings = settings
        self.kube = kube

    async def deploy(self, rev: RevisionRef) -> str:
        for manifest in render_manifests(rev, self.settings):
            await self.kube.apply(manifest)
        return rev.object_name

    async def status(self, rev: RevisionRef) -> BackendStatus:
        deployment, pods = await self.kube.read(rev.object_name, selector_string(rev))
        status = evaluate(deployment, pods, rev.spec.replicas)
        if status.failure is FailureReason.CRASH_LOOP and status.source:
            found = match_log(await crash_logs(self.kube, status.source))
            if found:
                status.failure, line = found
                status.message = f"{status.source}: {line}"
        if status.phase is Phase.READY:
            status.endpoint = service_url(self.settings, rev.object_name)
        return status

    async def activate(self, rev: RevisionRef) -> None:
        await self.kube.apply(render_stable_service(rev))

    async def deactivate(self, deployment: str) -> None:
        await self.kube.delete_service(deployment)

    async def teardown(self, rev: RevisionRef) -> None:
        await self.kube.delete_selected(selector_string(rev))

    def logs(self, rev: RevisionRef, follow: bool = False, tail: int = 200):
        return self.kube.stream_logs(selector_string(rev), follow, tail)
