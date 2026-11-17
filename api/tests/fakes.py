"""Test doubles for the backends, the Kubernetes wrapper and the SSH runner."""

from typing import Any

from icp_api.backends.base import BackendStatus, DeployError, Phase, RevisionRef
from icp_api.backends.ssh import CommandResult
from icp_api.config import Settings

READY = BackendStatus(Phase.READY, "1/1 replicas ready", endpoint="http://fake")
STARTING = BackendStatus(Phase.STARTING, "model loading")


def fast_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": "sqlite://",
        "poll_interval_s": 0.01,
        "queue_timeout_s": 5,
        "startup_timeout_s": 5,
        "smoke_timeout_s": 1,
        "prometheus_url": "http://127.0.0.1:9",  # nothing listens here
        "slurm_ssh_user": "tester",
    }
    return Settings(_env_file=None, **(values | overrides))


class FakeBackend:
    """Scripted backend. plans[n] lists the statuses revision n reports, one per poll;
    the last one repeats. Without a plan, a revision starts and then becomes ready."""

    def __init__(self, name: str = "kubernetes"):
        self.name = name
        self.plans: dict[int, list[BackendStatus]] = {}
        self.deploy_errors: dict[int, DeployError] = {}
        self.deployed: list[int] = []
        self.activated: list[int] = []
        self.torn_down: list[int] = []
        self.deactivated: list[str] = []
        self._polls: dict[int, int] = {}

    async def deploy(self, rev: RevisionRef) -> str:
        if rev.number in self.deploy_errors:
            raise self.deploy_errors[rev.number]
        self.deployed.append(rev.number)
        return f"fake-{rev.object_name}"

    async def status(self, rev: RevisionRef) -> BackendStatus:
        plan = self.plans.get(rev.number, [STARTING, READY])
        poll = self._polls.get(rev.number, 0)
        self._polls[rev.number] = poll + 1
        return plan[min(poll, len(plan) - 1)]

    async def activate(self, rev: RevisionRef) -> None:
        self.activated.append(rev.number)

    async def deactivate(self, deployment: str) -> None:
        self.deactivated.append(deployment)

    async def teardown(self, rev: RevisionRef) -> None:
        self.torn_down.append(rev.number)

    async def logs(self, rev: RevisionRef, follow: bool = False, tail: int = 200):
        yield f"fake log for r{rev.number}\n"


class FakeKube:
    def __init__(self) -> None:
        self.applied: list[dict] = []
        self.deleted: list[str] = []
        self.deleted_services: list[str] = []
        self.deployment: dict | None = None
        self.pods: list[dict] = []
        self.logs: dict[str, str] = {}
        self.previous_logs: dict[str, str] = {}

    async def apply(self, manifest: dict) -> None:
        self.applied.append(manifest)

    async def read(self, name: str, selector: str) -> tuple[dict | None, list[dict]]:
        return self.deployment, self.pods

    async def delete_selected(self, selector: str) -> None:
        self.deleted.append(selector)

    async def delete_service(self, name: str) -> None:
        self.deleted_services.append(name)

    async def pod_log(self, pod: str, previous: bool = False, tail: int = 200) -> str:
        return (self.previous_logs if previous else self.logs).get(pod, "")


class FakeRunner:
    """Answers commands by prefix, and records every command and its stdin."""

    def __init__(self, responses: dict[str, CommandResult] | None = None):
        self.responses = responses or {}
        self.commands: list[str] = []
        self.inputs: list[str | None] = []

    async def run(self, command: str, input: str | None = None) -> CommandResult:
        self.commands.append(command)
        self.inputs.append(input)
        for prefix, result in self.responses.items():
            if command.startswith(prefix):
                return result
        return CommandResult(1, "", f"no fake response for: {command}")

    async def stream(self, command: str):
        self.commands.append(command)
        yield "fake slurm log\n"


def ok(stdout: str = "") -> CommandResult:
    return CommandResult(0, stdout, "")


def err(stderr: str) -> CommandResult:
    return CommandResult(1, "", stderr)


def pod(name: str = "chat-r1-abc", **status: Any) -> dict:
    return {"metadata": {"name": name}, "status": status}


def deployment_obj(ready: int = 0, generation: int = 1, observed: int = 1) -> dict:
    return {
        "metadata": {"name": "chat-r1", "generation": generation},
        "status": {"readyReplicas": ready, "observedGeneration": observed},
    }
