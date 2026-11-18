"""Request and response models. The spec fields feed straight into Kubernetes manifests and
sbatch scripts, so anything that lands unquoted in them is validated here."""

import re
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator

# DNS-1035 label, short enough to leave room for "-r<number>" suffixes.
NAME_PATTERN = r"^[a-z]([-a-z0-9]{0,30}[a-z0-9])?$"
_SLURM_TOKEN = r"^[A-Za-z0-9_.\-]+$"
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _as_utc(value: datetime) -> datetime:
    # SQLite drops tzinfo; everything is stored in UTC.
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


UTCDateTime = Annotated[datetime, AfterValidator(_as_utc)]


class Resources(BaseModel):
    """Per-pod resources for the Kubernetes backend. Defaults fit the fake vLLM server."""

    cpu: str = Field("500m", pattern=r"^\d+(\.\d+)?m?$")
    memory: str = Field("1Gi", pattern=r"^\d+(\.\d+)?(Ki|Mi|Gi|Ti|K|M|G|T)?$")
    gpu: int = Field(0, ge=0, le=8)


class SlurmOptions(BaseModel):
    partition: str = Field("gpu", pattern=_SLURM_TOKEN)
    gpus: int = Field(1, ge=1, le=8)
    gpu_type: str | None = Field(None, pattern=_SLURM_TOKEN)
    # Node features, e.g. "a100@80g" for Explorer's 80 GB A100s (see `sinfo -o "%N %f"`).
    constraint: str | None = Field(None, pattern=r"^[A-Za-z0-9_.@&|\-]+$")
    cpus: int = Field(8, ge=1, le=128)
    mem: str = Field("32G", pattern=r"^\d+[KMGT]?$")
    time_limit: str = Field("02:00:00", pattern=r"^(\d+-)?\d{1,2}(:\d{2}){0,2}$")


class DeploymentSpec(BaseModel):
    model: str = Field(min_length=1, description="Hugging Face model id or local path")
    backend: Literal["kubernetes", "slurm"] = "kubernetes"
    image: str | None = Field(None, description="Container image (kubernetes) or .sif path (slurm)")
    vllm_args: list[str] = []
    env: dict[str, str] = {}
    replicas: int = Field(1, ge=1, le=8, description="Kubernetes only")
    resources: Resources = Resources()
    slurm: SlurmOptions = SlurmOptions()

    @field_validator("env")
    @classmethod
    def _env_keys(cls, env: dict[str, str]) -> dict[str, str]:
        bad = [key for key in env if not _ENV_KEY.match(key)]
        if bad:
            raise ValueError(f"invalid environment variable names: {bad}")
        return env


class DeploymentCreate(BaseModel):
    name: str = Field(pattern=NAME_PATTERN)
    spec: DeploymentSpec


class RollbackRequest(BaseModel):
    revision: int | None = Field(None, description="Default: the revision that served before the current one")


class RevisionOut(BaseModel):
    number: int
    backend: str
    status: str
    spec: DeploymentSpec
    handle: str | None
    failure_reason: str | None
    message: str | None
    source_revision: int | None
    created_at: UTCDateTime
    started_at: UTCDateTime | None
    finished_at: UTCDateTime | None
    queue_seconds: float | None
    startup_seconds: float | None
    gate_seconds: float | None
    total_seconds: float | None


class DeploymentOut(BaseModel):
    name: str
    created_at: UTCDateTime
    active_revision: RevisionOut | None
    latest_revision: RevisionOut | None


class EventOut(BaseModel):
    id: int
    revision: int
    at: UTCDateTime
    kind: str
    message: str
