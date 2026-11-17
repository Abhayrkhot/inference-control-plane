"""What every runtime backend (Kubernetes, Slurm) provides to the rollout engine."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from icp_api.failures import FailureReason
from icp_api.schemas import DeploymentSpec


class Phase(StrEnum):
    PENDING = "pending"  # waiting for resources: unscheduled pods, queued Slurm job
    STARTING = "starting"  # process is up, model is loading
    READY = "ready"  # passing readiness checks and reachable from the control plane
    FAILED = "failed"  # will not become ready without a new revision


@dataclass(frozen=True)
class RevisionRef:
    """Everything a backend needs to know about one revision."""

    deployment: str
    number: int
    spec: DeploymentSpec
    handle: str | None = None  # set after deploy(): object name or Slurm job id

    @property
    def object_name(self) -> str:
        return f"{self.deployment}-r{self.number}"


@dataclass
class BackendStatus:
    phase: Phase
    message: str = ""
    failure: FailureReason | None = None
    endpoint: str | None = None  # base URL the control plane can reach, once READY
    source: str | None = None  # pod name or job id this status came from


class DeployError(Exception):
    """The backend refused the revision outright (bad partition, invalid manifest...)."""

    def __init__(self, reason: FailureReason, message: str):
        super().__init__(message)
        self.reason = reason


class Backend(Protocol):
    name: str

    async def deploy(self, rev: RevisionRef) -> str:
        """Start the revision next to whatever is serving now. Returns its handle."""
        ...

    async def status(self, rev: RevisionRef) -> BackendStatus: ...

    async def activate(self, rev: RevisionRef) -> None:
        """Point the deployment's stable Service at this revision."""
        ...

    async def deactivate(self, deployment: str) -> None:
        """Remove the stable Service (when the deployment is deleted)."""
        ...

    async def teardown(self, rev: RevisionRef) -> None:
        """Stop the revision and delete what it created. Must be safe to call twice."""
        ...

    def logs(self, rev: RevisionRef, follow: bool = False, tail: int = 200) -> AsyncIterator[str]: ...
