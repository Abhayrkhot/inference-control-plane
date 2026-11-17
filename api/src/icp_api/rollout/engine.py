"""Blue/green rollouts.

A new revision starts next to the active one. It only takes traffic after the backend
reports it ready and it answers a test completion. If anything fails on the way, the new
revision is torn down and the active revision keeps serving: that is the rollback.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select

from icp_api import metrics
from icp_api.backends.base import Backend, BackendStatus, DeployError, Phase, RevisionRef
from icp_api.config import Settings
from icp_api.failures import FailureReason
from icp_api.models import Deployment, Revision, RevisionStatus, RolloutEvent, utcnow
from icp_api.rollout.smoke import smoke_test
from icp_api.schemas import DeploymentSpec

log = logging.getLogger("icp.rollout")

SmokeFn = Callable[[str, str, float], Awaitable[str | None]]
MAX_STATUS_ERRORS = 5  # consecutive failed status calls before giving up on the backend


class RolloutFailed(Exception):
    def __init__(self, reason: FailureReason, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


def revision_ref(rev: Revision, deployment_name: str) -> RevisionRef:
    return RevisionRef(deployment_name, rev.number, DeploymentSpec.model_validate(rev.spec), rev.handle)


class RolloutEngine:
    def __init__(
        self,
        db: Engine,
        backends: dict[str, Backend],
        settings: Settings,
        smoke: SmokeFn | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ):
        self.db = db
        self.backends = backends
        self.settings = settings
        self.smoke = smoke or smoke_test
        self.clock = clock
        self.sleep = sleep
        self._tasks: set[asyncio.Task] = set()

    # --- task management ---------------------------------------------------------------

    def start(self, revision_id: int) -> asyncio.Task:
        task = asyncio.create_task(self.run(revision_id), name=f"rollout-{revision_id}")
        self._tasks.add(task)  # keep a reference so the task is not garbage-collected
        task.add_done_callback(self._tasks.discard)
        return task

    async def shutdown(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # --- the rollout ---------------------------------------------------------------------

    async def run(self, revision_id: int) -> str:
        with Session(self.db) as s:
            rev = s.get(Revision, revision_id)
            deployment = s.get(Deployment, rev.deployment_id)
            rev.status = RevisionStatus.ROLLING_OUT
            rev.started_at = utcnow()
            s.add(rev)
            s.commit()
            ref = revision_ref(rev, deployment.name)
            backend = self.backends[rev.backend]

        timings: dict[str, float] = {}
        started = self.clock()
        self._event(revision_id, "rollout_started", f"r{ref.number}: {ref.spec.model} on {backend.name}")
        try:
            try:
                handle = await backend.deploy(ref)
            except DeployError as e:
                raise RolloutFailed(e.reason, str(e)) from e
            ref = replace(ref, handle=handle)
            self._save(revision_id, handle=handle)
            self._event(revision_id, "submitted", f"{backend.name} accepted it as {handle}")

            status = await self._wait_until_ready(revision_id, backend, ref, timings)

            gate_started = self.clock()
            error = await self.smoke(status.endpoint or "", ref.deployment, self.settings.smoke_timeout_s)
            timings["gate"] = self.clock() - gate_started
            if error:
                raise RolloutFailed(FailureReason.SMOKE_TEST, f"test completion failed: {error}")
            self._event(revision_id, "smoke_test_passed", "answered a test completion")

            await backend.activate(ref)
        except asyncio.CancelledError:
            raise
        except RolloutFailed as e:
            await self._fail(revision_id, backend, ref, e.reason, e.message, started, timings)
            return RevisionStatus.FAILED
        except Exception as e:
            log.exception("rollout of revision %s crashed", revision_id)
            reason, message = FailureReason.BACKEND_ERROR, f"{type(e).__name__}: {e}"
            await self._fail(revision_id, backend, ref, reason, message, started, timings)
            return RevisionStatus.FAILED

        # Traffic has moved; from here on, problems are logged but never undo the switch.
        previous = self._promote(revision_id)
        self._event(revision_id, "activated", f"r{ref.number} is now serving traffic")
        if previous:
            await self._retire(revision_id, *previous)
        self._finish(revision_id, backend.name, "active", started, timings)
        return RevisionStatus.ACTIVE

    async def _wait_until_ready(
        self, revision_id: int, backend: Backend, ref: RevisionRef, timings: dict[str, float]
    ) -> BackendStatus:
        s = self.settings
        pending_since = self.clock()
        starting_since: float | None = None
        last: tuple[Phase, str] | None = None
        errors = 0
        while True:
            try:
                status = await backend.status(ref)
                errors = 0
            except Exception as e:
                errors += 1
                self._event(revision_id, "status_error", f"{type(e).__name__}: {e}")
                if errors >= MAX_STATUS_ERRORS:
                    raise RolloutFailed(FailureReason.BACKEND_ERROR, f"lost contact with {backend.name}: {e}") from e
                await self.sleep(s.poll_interval_s)
                continue

            now = self.clock()
            if status.phase is Phase.FAILED:
                raise RolloutFailed(status.failure or FailureReason.UNKNOWN, status.message)
            if (status.phase, status.message) != last:
                last = (status.phase, status.message)
                self._event(revision_id, status.phase.value, status.message)
            if status.phase in (Phase.STARTING, Phase.READY) and starting_since is None:
                starting_since = now
                timings["queue"] = now - pending_since
            if status.phase is Phase.READY:
                timings["startup"] = now - (starting_since or now)
                if not status.endpoint:
                    raise RolloutFailed(FailureReason.BACKEND_ERROR, "backend reported ready without an endpoint")
                return status
            if starting_since is None and now - pending_since > s.queue_timeout_s:
                raise RolloutFailed(
                    FailureReason.UNSCHEDULABLE, f"still pending after {s.queue_timeout_s:.0f}s: {status.message}"
                )
            if starting_since is not None and now - starting_since > s.startup_timeout_s:
                raise RolloutFailed(
                    FailureReason.STARTUP_TIMEOUT,
                    f"not ready {s.startup_timeout_s:.0f}s after starting: {status.message}",
                )
            await self.sleep(s.poll_interval_s)

    def _promote(self, revision_id: int) -> tuple[str, RevisionRef] | None:
        """Mark the revision active. Returns the revision it replaced, to tear down."""
        with Session(self.db) as s:
            rev = s.get(Revision, revision_id)
            deployment = s.get(Deployment, rev.deployment_id)
            previous = s.get(Revision, deployment.active_revision_id) if deployment.active_revision_id else None
            deployment.active_revision_id = rev.id
            rev.status = RevisionStatus.ACTIVE
            replaced = None
            if previous is not None and previous.id != rev.id:
                previous.status = RevisionStatus.SUPERSEDED
                replaced = (previous.backend, revision_ref(previous, deployment.name))
                s.add(previous)
            s.add(rev)
            s.add(deployment)
            s.commit()
            return replaced

    async def _retire(self, revision_id: int, backend_name: str, ref: RevisionRef) -> None:
        try:
            await self.backends[backend_name].teardown(ref)
            self._event(revision_id, "retired", f"tore down r{ref.number}")
        except Exception as e:
            log.exception("could not tear down r%s of %s", ref.number, ref.deployment)
            self._event(revision_id, "retire_failed", f"could not tear down r{ref.number}: {e}")

    async def _fail(
        self,
        revision_id: int,
        backend: Backend,
        ref: RevisionRef,
        reason: FailureReason,
        message: str,
        started: float,
        timings: dict[str, float],
    ) -> None:
        log.warning("r%s of %s failed: %s: %s", ref.number, ref.deployment, reason.value, message)
        self._event(revision_id, "failed", f"{reason.value}: {message}")
        try:
            await backend.teardown(ref)
        except Exception as e:
            self._event(revision_id, "teardown_failed", f"{type(e).__name__}: {e}")
        with Session(self.db) as s:
            rev = s.get(Revision, revision_id)
            rev.failure_reason = reason.value
            rev.message = message[:2000]
            s.add(rev)
            s.commit()
            deployment = s.get(Deployment, rev.deployment_id)
            active = s.get(Revision, deployment.active_revision_id) if deployment.active_revision_id else None
            active_number = active.number if active else None
        if active_number is not None:
            self._event(revision_id, "rollback", f"tore down r{ref.number}; r{active_number} kept serving traffic")
        else:
            self._event(revision_id, "rollback", f"tore down r{ref.number}; no earlier revision to fall back to")
        metrics.ROLLBACKS.labels(backend.name, reason.value).inc()
        self._finish(revision_id, backend.name, "failed", started, timings)

    def _finish(self, revision_id: int, backend: str, result: str, started: float, timings: dict[str, float]) -> None:
        total = self.clock() - started
        with Session(self.db) as s:
            rev = s.get(Revision, revision_id)
            if result == "failed":
                rev.status = RevisionStatus.FAILED
            rev.finished_at = utcnow()
            rev.queue_seconds = timings.get("queue")
            rev.startup_seconds = timings.get("startup")
            rev.gate_seconds = timings.get("gate")
            rev.total_seconds = total
            s.add(rev)
            s.commit()
        metrics.ROLLOUTS.labels(backend, result).inc()
        metrics.ROLLOUT_SECONDS.labels(backend, result).observe(total)
        for phase, seconds in timings.items():
            metrics.PHASE_SECONDS.labels(backend, phase).observe(seconds)

    # --- persistence helpers -------------------------------------------------------------

    def _event(self, revision_id: int, kind: str, message: str) -> None:
        log.info("revision %s %s: %s", revision_id, kind, message)
        with Session(self.db) as s:
            rev = s.get(Revision, revision_id)
            s.add(
                RolloutEvent(
                    revision_id=revision_id, deployment_id=rev.deployment_id, kind=kind, message=message[:2000]
                )
            )
            s.commit()

    def _save(self, revision_id: int, **fields: Any) -> None:
        with Session(self.db) as s:
            rev = s.get(Revision, revision_id)
            for key, value in fields.items():
                setattr(rev, key, value)
            s.add(rev)
            s.commit()


def recover_interrupted(db: Engine) -> int:
    """Rollouts run inside the API process, so a restart orphans any in flight. Mark them
    failed so the deployment is not stuck. TODO: resume them, or tear them down."""
    in_flight = [RevisionStatus.PENDING, RevisionStatus.ROLLING_OUT]
    with Session(db) as s:
        stuck = s.exec(select(Revision).where(col(Revision.status).in_(in_flight))).all()
        for rev in stuck:
            rev.status = RevisionStatus.FAILED
            rev.failure_reason = FailureReason.BACKEND_ERROR.value
            rev.message = "control plane restarted during the rollout; the revision may still be running"
            rev.finished_at = utcnow()
            s.add(rev)
            s.add(RolloutEvent(revision_id=rev.id, deployment_id=rev.deployment_id, kind="failed", message=rev.message))
        s.commit()
        return len(stuck)
