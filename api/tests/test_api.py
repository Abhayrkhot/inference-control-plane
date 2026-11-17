import time

from icp_api.backends.base import BackendStatus, Phase
from icp_api.failures import FailureReason

SPEC = {"model": "facebook/opt-125m"}
TERMINAL = {"active", "failed", "superseded"}


def wait_for(client, name: str, number: int, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        revisions = client.get(f"/api/deployments/{name}/revisions").json()
        rev = next((r for r in revisions if r["number"] == number), None)
        if rev and rev["status"] in TERMINAL and rev["total_seconds"] is not None:
            return rev
        time.sleep(0.02)
    raise AssertionError(f"r{number} of {name} did not finish")


def create(client, name: str = "chat") -> None:
    response = client.post("/api/deployments", json={"name": name, "spec": SPEC})
    assert response.status_code == 201, response.text


def test_create_roll_out_and_roll_back(client, backend):
    create(client)
    assert wait_for(client, "chat", 1)["status"] == "active"

    response = client.post("/api/deployments/chat/rollouts", json=SPEC | {"vllm_args": ["--max-model-len=2048"]})
    assert response.status_code == 202, response.text
    assert wait_for(client, "chat", 2)["status"] == "active"
    assert client.get("/api/deployments/chat").json()["active_revision"]["number"] == 2
    assert backend.torn_down == [1]

    response = client.post("/api/deployments/chat/rollback", json={})
    assert response.status_code == 202, response.text
    assert response.json()["source_revision"] == 1
    rev3 = wait_for(client, "chat", 3)
    assert rev3["status"] == "active"
    assert rev3["spec"]["vllm_args"] == []
    statuses = {r["number"]: r["status"] for r in client.get("/api/deployments/chat/revisions").json()}
    assert statuses == {1: "superseded", 2: "superseded", 3: "active"}


def test_failed_rollout_keeps_previous_revision_serving(client, backend):
    create(client)
    wait_for(client, "chat", 1)
    backend.plans[2] = [
        BackendStatus(Phase.FAILED, "torch.OutOfMemoryError: CUDA out of memory", FailureReason.GPU_OOM)
    ]

    client.post("/api/deployments/chat/rollouts", json=SPEC)
    rev = wait_for(client, "chat", 2)

    assert rev["status"] == "failed"
    assert rev["failure_reason"] == "gpu_oom"
    assert client.get("/api/deployments/chat").json()["active_revision"]["number"] == 1
    assert backend.activated == [1]
    assert backend.torn_down == [2]
    kinds = [e["kind"] for e in client.get("/api/deployments/chat/events", params={"revision": 2}).json()]
    assert kinds[-2:] == ["failed", "rollback"]


def test_failed_test_completion_rolls_back(client, backend, smoke_errors):
    create(client)
    wait_for(client, "chat", 1)
    backend.plans[2] = [BackendStatus(Phase.READY, "ready", endpoint="http://broken")]
    smoke_errors["http://broken"] = "HTTP 500: boom"

    client.post("/api/deployments/chat/rollouts", json=SPEC)
    rev = wait_for(client, "chat", 2)

    assert rev["failure_reason"] == "smoke_test"
    assert "HTTP 500" in rev["message"]
    assert client.get("/api/deployments/chat").json()["active_revision"]["number"] == 1


def test_rejects_conflicts_and_bad_input(client, backend):
    backend.plans[1] = [BackendStatus(Phase.PENDING, "queued")]
    create(client)
    assert client.post("/api/deployments", json={"name": "chat", "spec": SPEC}).status_code == 409
    assert client.post("/api/deployments/chat/rollouts", json=SPEC).status_code == 409  # r1 still in flight
    assert client.post("/api/deployments/chat/rollback", json={}).status_code == 400  # nothing to go back to
    assert client.post("/api/deployments", json={"name": "Bad_Name", "spec": SPEC}).status_code == 422

    injection = {"model": "m", "backend": "slurm", "slurm": {"partition": "gpu\n#SBATCH --account=x"}}
    assert client.post("/api/deployments", json={"name": "x", "spec": injection}).status_code == 422
    # This app was built with only the Kubernetes backend.
    slurm = {"model": "m", "backend": "slurm"}
    assert client.post("/api/deployments", json={"name": "y", "spec": slurm}).status_code == 400
    assert client.get("/api/deployments/nope").status_code == 404


def test_events_logs_and_delete(client, backend):
    create(client)
    wait_for(client, "chat", 1)

    events = client.get("/api/deployments/chat/events").json()
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "rollout_started"
    assert {"submitted", "starting", "ready", "smoke_test_passed", "activated"} <= set(kinds)
    assert events[0]["at"].endswith("Z")
    assert client.get("/api/deployments/chat/events", params={"after": events[-1]["id"]}).json() == []

    assert client.get("/api/deployments/chat/logs").text == "fake log for r1\n"

    assert client.delete("/api/deployments/chat").status_code == 204
    assert backend.deactivated == ["chat"]
    assert backend.torn_down == [1]
    assert client.get("/api/deployments").json() == []


def test_metrics_endpoints(client):
    assert client.get("/healthz").json() == {"ok": True}
    create(client)
    wait_for(client, "chat", 1)

    response = client.get("/api/deployments/chat/metrics")
    assert response.status_code == 503
    assert "Prometheus" in response.json()["detail"]

    text = client.get("/metrics").text
    assert 'icp_rollouts_total{backend="kubernetes",result="active"}' in text
    assert "icp_rollout_phase_seconds_bucket" in text
