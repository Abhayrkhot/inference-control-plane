"""Turn raw failure signals (pod status, Slurm job state, log lines) into a short list of
reasons that the UI, CLI and metrics can group by."""

import re
from enum import StrEnum
from typing import Any


class FailureReason(StrEnum):
    IMAGE_PULL = "image_pull"
    OOM_KILLED = "oom_killed"  # host memory: container OOMKilled, Slurm OUT_OF_MEMORY
    GPU_OOM = "gpu_oom"  # CUDA OOM, or the KV cache does not fit at this max-model-len
    MODEL_ACCESS = "model_access"  # gated or missing model on Hugging Face
    BAD_ARGS = "bad_args"
    CRASH_LOOP = "crash_loop"  # exited for a reason none of the patterns recognise
    UNSCHEDULABLE = "unschedulable"  # never got resources, or asked for impossible ones
    STARTUP_TIMEOUT = "startup_timeout"
    SMOKE_TEST = "smoke_test"
    TIME_LIMIT = "time_limit"  # Slurm job hit its wall-clock limit
    SSH = "ssh"  # could not reach or authenticate to the cluster
    BACKEND_ERROR = "backend_error"  # Kubernetes / Slurm call failed
    UNKNOWN = "unknown"


# Checked in order; the first pattern that matches any line wins.
_LOG_PATTERNS = [
    (FailureReason.GPU_OOM, r"CUDA out of memory|torch\.OutOfMemoryError|CUDA error: out of memory"),
    (
        FailureReason.GPU_OOM,
        r"No available memory for the cache blocks|stored in KV cache|larger than the available KV cache memory",
    ),
    (
        FailureReason.MODEL_ACCESS,
        r"GatedRepoError|Cannot access gated repo|401 Client Error|RepositoryNotFoundError"
        r"|404 Client Error|not a valid model identifier",
    ),
    (
        FailureReason.BAD_ARGS,
        r"error: unrecognized arguments|error: argument |error: the following arguments are required",
    ),
    (FailureReason.SSH, r"Permission denied \(publickey|Host key verification failed|Could not resolve hostname"),
]
_COMPILED = [(reason, re.compile(pattern, re.IGNORECASE)) for reason, pattern in _LOG_PATTERNS]


def match_log(text: str) -> tuple[FailureReason, str] | None:
    """Find the first known failure in a log. Returns the reason and the line that matched."""
    lines = text.splitlines()
    for reason, pattern in _COMPILED:
        for line in reversed(lines):  # the line closest to the crash
            if pattern.search(line):
                return reason, line.strip()[:300]
    return None


# --- Kubernetes ----------------------------------------------------------------------------

_IMAGE_PULL = {"ErrImagePull", "ImagePullBackOff", "InvalidImageName", "ErrImageNeverPull"}


def classify_pod(pod: dict[str, Any]) -> tuple[FailureReason, str] | None:
    """Spot pod failures that will not fix themselves. `pod` is a camelCase API dict.

    A model server should never exit during a rollout, so any non-zero exit counts as a crash;
    the backend then reads the container log to say why.
    """
    name = (pod.get("metadata") or {}).get("name", "?")
    for cs in (pod.get("status") or {}).get("containerStatuses") or []:
        state = cs.get("state") or {}
        waiting = state.get("waiting") or {}
        if waiting.get("reason") in _IMAGE_PULL:
            return FailureReason.IMAGE_PULL, f"{name}: {waiting.get('message') or waiting['reason']}"
        if waiting.get("reason") == "CreateContainerConfigError":
            return FailureReason.BACKEND_ERROR, f"{name}: {waiting.get('message', 'container config error')}"
        terminated = state.get("terminated") or (cs.get("lastState") or {}).get("terminated") or {}
        if terminated.get("reason") == "OOMKilled":
            return FailureReason.OOM_KILLED, f"{name}: container was OOM-killed (raise resources.memory)"
        if terminated.get("exitCode") not in (None, 0):
            return FailureReason.CRASH_LOOP, f"{name}: container exited with code {terminated['exitCode']}"
    return None


# --- Slurm ---------------------------------------------------------------------------------

_SLURM_END_STATES = {
    "OUT_OF_MEMORY": FailureReason.OOM_KILLED,
    "TIMEOUT": FailureReason.TIME_LIMIT,
    "DEADLINE": FailureReason.TIME_LIMIT,
    "NODE_FAIL": FailureReason.BACKEND_ERROR,
    "PREEMPTED": FailureReason.BACKEND_ERROR,
}

# Pending reasons that mean the job can never start as submitted.
FATAL_PENDING_REASONS = frozenset(
    {
        "PartitionTimeLimit",
        "PartitionNodeLimit",
        "PartitionConfig",
        "InvalidAccount",
        "InvalidQOS",
        "BadConstraints",
        "AccountNotAllowed",
        "QOSMaxWallDurationPerJobLimit",
        "QOSMaxGRESPerJob",
    }
)
# Pending reasons that mean you are at a per-user limit. With blue/green rollouts the
# revision that is serving now may be the job holding that limit.
LIMIT_PENDING_REASONS = frozenset(
    {"QOSMaxGRESPerUser", "QOSMaxJobsPerUserLimit", "AssocGrpGRES", "AssocMaxJobsLimit", "QOSMaxCpuPerUserLimit"}
)


def slurm_state_name(state: str) -> str:
    """'CANCELLED by 1234' -> 'CANCELLED', 'FAILED+' -> 'FAILED'."""
    return state.split()[0].rstrip("+") if state.strip() else "UNKNOWN"


def classify_slurm(state: str, log_tail: str = "") -> tuple[FailureReason, str]:
    """Classify a job that has ended. Returns the reason and the most useful log line."""
    found = match_log(log_tail)
    line = found[1] if found else ""
    base = slurm_state_name(state)
    if base in _SLURM_END_STATES:
        return _SLURM_END_STATES[base], line
    if found:
        return found
    return FailureReason.UNKNOWN, line


def classify_sbatch_error(stderr: str) -> FailureReason:
    text = stderr.lower()
    impossible = (
        "invalid partition",
        "node configuration is not available",
        "time limit",
        "invalid generic resource",
        "invalid gres",
        "invalid account",
        "invalid qos",
        "more processors than permitted",
    )
    if any(phrase in text for phrase in impossible):
        return FailureReason.UNSCHEDULABLE
    return FailureReason.BACKEND_ERROR
