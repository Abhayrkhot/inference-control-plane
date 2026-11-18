from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


class RevisionStatus(StrEnum):
    PENDING = "pending"  # created, rollout not started yet
    ROLLING_OUT = "rolling_out"
    ACTIVE = "active"  # serving traffic
    FAILED = "failed"  # never took traffic; torn down
    SUPERSEDED = "superseded"  # served traffic, then replaced; torn down
    STOPPED = "stopped"  # served traffic until the operator stopped it; torn down, history kept


class Deployment(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=utcnow)
    # Plain int rather than a foreign key, to avoid a deployment <-> revision cycle.
    active_revision_id: int | None = None


class Revision(SQLModel, table=True):
    """One immutable spec for a deployment, and what happened when it was rolled out."""

    __table_args__ = (UniqueConstraint("deployment_id", "number"),)

    id: int | None = Field(default=None, primary_key=True)
    deployment_id: int = Field(foreign_key="deployment.id", index=True)
    number: int
    backend: str
    spec: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    status: str = RevisionStatus.PENDING
    handle: str | None = None  # backend id: Kubernetes object name or Slurm job id
    failure_reason: str | None = None
    message: str | None = None
    source_revision: int | None = None  # set when this revision is a rollback to an older spec
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # Measured rollout phases, in seconds. These are the numbers worth reporting.
    queue_seconds: float | None = None
    startup_seconds: float | None = None
    gate_seconds: float | None = None
    total_seconds: float | None = None


class RolloutEvent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    deployment_id: int = Field(foreign_key="deployment.id", index=True)
    revision_id: int = Field(foreign_key="revision.id", index=True)
    at: datetime = Field(default_factory=utcnow)
    kind: str
    message: str = ""
