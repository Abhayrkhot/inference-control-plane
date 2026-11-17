import pytest

from icp_api.failures import FailureReason, classify_sbatch_error, classify_slurm, match_log


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.00 GiB", FailureReason.GPU_OOM),
        (
            "ValueError: The model's max seq len (32768) is larger than the maximum number of tokens "
            "that can be stored in KV cache (9000).",
            FailureReason.GPU_OOM,
        ),
        ("ValueError: No available memory for the cache blocks.", FailureReason.GPU_OOM),
        ("GatedRepoError: 401 Client Error. Cannot access gated repo for url", FailureReason.MODEL_ACCESS),
        ("RepositoryNotFoundError: 404 Client Error.", FailureReason.MODEL_ACCESS),
        ("vllm serve: error: unrecognized arguments: --max-model-length 2048", FailureReason.BAD_ARGS),
        ("user@login.example: Permission denied (publickey,gssapi-keyex).", FailureReason.SSH),
    ],
)
def test_match_log(line, reason):
    assert match_log(f"INFO starting\n{line}\nINFO shutting down\n") == (reason, line)


def test_match_log_without_a_known_failure():
    assert match_log("INFO all good\n") is None


def test_classify_slurm():
    assert classify_slurm("OUT_OF_MEMORY")[0] is FailureReason.OOM_KILLED
    assert classify_slurm("TIMEOUT")[0] is FailureReason.TIME_LIMIT
    assert classify_slurm("CANCELLED by 1234")[0] is FailureReason.UNKNOWN
    assert classify_slurm("FAILED", "x\ntorch.OutOfMemoryError: CUDA out of memory\n") == (
        FailureReason.GPU_OOM,
        "torch.OutOfMemoryError: CUDA out of memory",
    )


def test_classify_sbatch_error():
    assert classify_sbatch_error("sbatch: error: invalid partition specified: gpux") is FailureReason.UNSCHEDULABLE
    assert (
        classify_sbatch_error(
            "sbatch: error: Batch job submission failed: Requested node configuration is not available"
        )
        is FailureReason.UNSCHEDULABLE
    )
    assert classify_sbatch_error("sbatch: error: Slurm controller not responding") is FailureReason.BACKEND_ERROR
