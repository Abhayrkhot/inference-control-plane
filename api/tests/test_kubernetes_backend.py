import asyncio

from fakes import FakeKube, deployment_obj, fast_settings, pod

from icp_api.backends.base import Phase, RevisionRef
from icp_api.backends.kubernetes import KubernetesBackend, evaluate, render_manifests, render_stable_service
from icp_api.failures import FailureReason
from icp_api.schemas import DeploymentSpec, Resources


def ref(**spec) -> RevisionRef:
    return RevisionRef("chat", 3, DeploymentSpec(model="facebook/opt-125m", **spec))


def test_revision_manifests():
    settings = fast_settings()
    deployment, service = render_manifests(ref(vllm_args=["--max-model-len=2048"], env={"A": "1"}), settings)

    assert deployment["metadata"]["name"] == "chat-r3"
    labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert labels["icp/deployment"] == "chat"
    assert labels["icp/revision"] == "3"
    assert labels["icp/managed"] == "true"
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["vllm", "serve"]
    assert container["args"][:3] == ["facebook/opt-125m", "--served-model-name", "chat"]
    assert container["args"][-1] == "--max-model-len=2048"
    assert container["image"] == "icp-fake-vllm:dev"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["startupProbe"]["failureThreshold"] * 5 > settings.startup_timeout_s
    assert {"name": "A", "value": "1"} in container["env"]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert service["metadata"]["name"] == "chat-r3"
    assert service["spec"]["selector"] == {"icp/deployment": "chat", "icp/revision": "3"}


def test_gpu_limit():
    deployment, _ = render_manifests(ref(resources=Resources(gpu=1)), fast_settings())
    assert deployment["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"


def test_stable_service_points_at_the_revision():
    service = render_stable_service(ref())
    assert service["metadata"]["name"] == "chat"
    assert service["spec"]["selector"] == {"icp/deployment": "chat", "icp/revision": "3"}
    # Must not match the revision selector, or tearing down a revision would delete it.
    assert "icp/revision" not in service["metadata"]["labels"]


def test_phases():
    assert evaluate(None, [], 1).phase is Phase.FAILED
    assert evaluate(deployment_obj(), [], 1).phase is Phase.PENDING
    unscheduled = pod(conditions=[{"type": "PodScheduled", "status": "False", "message": "Insufficient memory"}])
    status = evaluate(deployment_obj(), [unscheduled], 1)
    assert status.phase is Phase.PENDING
    assert "Insufficient memory" in status.message
    assert evaluate(deployment_obj(), [pod(phase="Running")], 1).phase is Phase.STARTING
    assert evaluate(deployment_obj(ready=1), [pod(phase="Running")], 1).phase is Phase.READY
    # The controller has not caught up with the latest spec yet.
    assert evaluate(deployment_obj(ready=1, generation=2, observed=1), [pod()], 1).phase is Phase.STARTING


def test_failures():
    pull = pod(containerStatuses=[{"state": {"waiting": {"reason": "ImagePullBackOff", "message": "pull denied"}}}])
    status = evaluate(deployment_obj(), [pull], 1)
    assert (status.phase, status.failure) == (Phase.FAILED, FailureReason.IMAGE_PULL)
    assert "pull denied" in status.message

    oom = pod(
        containerStatuses=[
            {"restartCount": 1, "state": {"running": {}}, "lastState": {"terminated": {"reason": "OOMKilled"}}}
        ]
    )
    assert evaluate(deployment_obj(), [oom], 1).failure is FailureReason.OOM_KILLED

    crash = crash_pod("chat-r3-x")
    status = evaluate(deployment_obj(), [crash], 1)
    assert status.failure is FailureReason.CRASH_LOOP
    assert status.source == "chat-r3-x"


def crash_pod(name: str) -> dict:
    return pod(
        name,
        containerStatuses=[
            {
                "restartCount": 1,
                "state": {"waiting": {"reason": "CrashLoopBackOff"}},
                "lastState": {"terminated": {"reason": "Error", "exitCode": 1}},
            }
        ],
    )


def test_crash_is_explained_by_the_container_log():
    kube = FakeKube()
    kube.deployment = deployment_obj()
    kube.pods = [crash_pod("chat-r3-x")]
    kube.logs["chat-r3-x"] = "INFO loading\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n"

    status = asyncio.run(KubernetesBackend(fast_settings(), kube).status(ref()))

    assert status.failure is FailureReason.GPU_OOM
    assert "Tried to allocate" in status.message


def test_crash_log_is_found_when_the_previous_log_is_unavailable():
    # Seen on kind: after a fast restart, `previous=true` returns this text instead of an error.
    kube = FakeKube()
    kube.deployment = deployment_obj()
    kube.pods = [crash_pod("chat-r3-x")]
    kube.previous_logs["chat-r3-x"] = "unable to retrieve container logs for containerd://abc123"
    kube.logs["chat-r3-x"] = "huggingface_hub.errors.GatedRepoError: 401 Client Error. Cannot access gated repo.\n"

    status = asyncio.run(KubernetesBackend(fast_settings(), kube).status(ref()))

    assert status.failure is FailureReason.MODEL_ACCESS


def test_backend_lifecycle():
    kube = FakeKube()
    backend = KubernetesBackend(fast_settings(), kube)

    assert asyncio.run(backend.deploy(ref())) == "chat-r3"
    assert [m["kind"] for m in kube.applied] == ["Deployment", "Service"]

    kube.deployment = deployment_obj(ready=1)
    kube.pods = [pod(phase="Running")]
    status = asyncio.run(backend.status(ref()))
    assert status.phase is Phase.READY
    assert status.endpoint == "http://chat-r3.icp-models.svc.cluster.local:8000"

    asyncio.run(backend.activate(ref()))
    assert kube.applied[-1]["metadata"]["name"] == "chat"

    asyncio.run(backend.teardown(ref()))
    assert kube.deleted == ["icp/deployment=chat,icp/revision=3"]
