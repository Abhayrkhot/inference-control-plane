import asyncio

from fakes import READY, STARTING, FakeBackend, fast_settings
from sqlmodel import Session, col, select

from icp_api.backends.base import BackendStatus, DeployError, Phase
from icp_api.db import init_db, make_engine
from icp_api.failures import FailureReason
from icp_api.models import Deployment, Revision, RolloutEvent
from icp_api.rollout.engine import RolloutEngine
from icp_api.schemas import DeploymentSpec

PENDING = BackendStatus(Phase.PENDING, "queued")


class FakeClock:
    """Time only moves when the engine sleeps (or the fake test completion runs)."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


class Harness:
    def __init__(self, backend: FakeBackend | None = None, smoke_error: str | None = None):
        self.backend = backend or FakeBackend()
        self.clock = FakeClock()
        self.db = make_engine("sqlite://")
        init_db(self.db)
        settings = fast_settings(poll_interval_s=1, queue_timeout_s=30, startup_timeout_s=60)

        async def smoke(endpoint: str, model: str, timeout: float) -> str | None:
            self.clock.now += 0.5
            return smoke_error

        self.engine = RolloutEngine(
            self.db, {"kubernetes": self.backend}, settings, smoke=smoke, clock=self.clock, sleep=self.clock.sleep
        )

    def rollout(self, number: int) -> Revision:
        with Session(self.db) as s:
            deployment = s.exec(select(Deployment)).first() or Deployment(name="chat")
            s.add(deployment)
            s.commit()
            s.refresh(deployment)
            spec = DeploymentSpec(model="m").model_dump(mode="json")
            rev = Revision(deployment_id=deployment.id, number=number, backend="kubernetes", spec=spec)
            s.add(rev)
            s.commit()
            s.refresh(rev)
            revision_id = rev.id
        asyncio.run(self.engine.run(revision_id))
        return self.revision(revision_id)

    def revision(self, revision_id: int) -> Revision:
        with Session(self.db) as s:
            return s.get(Revision, revision_id)

    def active_number(self) -> int | None:
        with Session(self.db) as s:
            deployment = s.exec(select(Deployment)).one()
            active = s.get(Revision, deployment.active_revision_id) if deployment.active_revision_id else None
            return active.number if active else None

    def events(self, revision_id: int) -> list[str]:
        with Session(self.db) as s:
            query = select(RolloutEvent).where(RolloutEvent.revision_id == revision_id).order_by(col(RolloutEvent.id))
            return [e.kind for e in s.exec(query)]


def test_first_rollout_becomes_active_and_records_timings():
    h = Harness()
    h.backend.plans[1] = [PENDING, PENDING, STARTING, STARTING, READY]

    rev = h.rollout(1)

    assert rev.status == "active"
    assert h.active_number() == 1
    assert (rev.queue_seconds, rev.startup_seconds, rev.gate_seconds, rev.total_seconds) == (2, 2, 0.5, 4.5)
    assert h.events(rev.id) == [
        "rollout_started",
        "submitted",
        "pending",
        "starting",
        "ready",
        "smoke_test_passed",
        "activated",
    ]


def test_new_revision_replaces_and_retires_the_old_one():
    h = Harness()
    first = h.rollout(1)
    second = h.rollout(2)

    assert second.status == "active"
    assert h.revision(first.id).status == "superseded"
    assert h.active_number() == 2
    assert h.backend.activated == [1, 2]
    assert h.backend.torn_down == [1]
    assert h.events(second.id)[-1] == "retired"


def test_backend_failure_rolls_back_to_the_serving_revision():
    h = Harness()
    h.rollout(1)
    h.backend.plans[2] = [STARTING, BackendStatus(Phase.FAILED, "CUDA out of memory", FailureReason.GPU_OOM)]

    rev = h.rollout(2)

    assert rev.status == "failed"
    assert rev.failure_reason == "gpu_oom"
    assert "CUDA" in rev.message
    assert h.active_number() == 1
    assert h.backend.torn_down == [2]
    assert h.events(rev.id)[-2:] == ["failed", "rollback"]


def test_failed_test_completion_never_takes_traffic():
    h = Harness(smoke_error="HTTP 500")
    rev = h.rollout(1)

    assert rev.failure_reason == "smoke_test"
    assert h.active_number() is None
    assert h.backend.activated == []


def test_startup_timeout():
    h = Harness()
    h.backend.plans[1] = [STARTING]
    rev = h.rollout(1)
    assert rev.failure_reason == "startup_timeout"
    assert rev.queue_seconds == 0
    assert rev.startup_seconds is None


def test_queue_timeout():
    h = Harness()
    h.backend.plans[1] = [PENDING]
    rev = h.rollout(1)
    assert rev.failure_reason == "unschedulable"
    assert "still pending" in rev.message


def test_rejected_deploy_is_cleaned_up():
    h = Harness()
    h.backend.deploy_errors[1] = DeployError(FailureReason.UNSCHEDULABLE, "sbatch: error: invalid partition specified")
    rev = h.rollout(1)
    assert rev.failure_reason == "unschedulable"
    assert "invalid partition" in rev.message
    assert h.backend.torn_down == [1]


class FlakyBackend(FakeBackend):
    def __init__(self, failures: int):
        super().__init__()
        self.failures = failures

    async def status(self, rev):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("ssh connection dropped")
        return await super().status(rev)


def test_transient_status_errors_are_retried():
    h = Harness(FlakyBackend(failures=2))
    rev = h.rollout(1)
    assert rev.status == "active"
    assert h.events(rev.id).count("status_error") == 2


def test_persistent_status_errors_fail_the_rollout():
    h = Harness(FlakyBackend(failures=100))
    rev = h.rollout(1)
    assert rev.failure_reason == "backend_error"
    assert "lost contact" in rev.message
