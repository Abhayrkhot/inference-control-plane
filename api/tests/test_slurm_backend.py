import asyncio

import pytest
from fakes import FakeKube, FakeRunner, deployment_obj, err, fast_settings, ok, pod
from pydantic import ValidationError

from icp_api.backends.base import DeployError, Phase, RevisionRef
from icp_api.backends.slurm import SlurmBackend, parse_squeue, render_sbatch
from icp_api.failures import FailureReason
from icp_api.schemas import DeploymentSpec, SlurmOptions


def ref(handle: str | None = None, model: str = "Qwen/Qwen2.5-7B-Instruct", **spec) -> RevisionRef:
    return RevisionRef("chat", 2, DeploymentSpec(model=model, backend="slurm", **spec), handle)


def status(runner: FakeRunner, kube: FakeKube | None = None):
    backend = SlurmBackend(fast_settings(), kube or FakeKube(), runner)
    return asyncio.run(backend.status(ref(handle="4242")))


def test_sbatch_script():
    script = render_sbatch(
        ref(
            vllm_args=["--max-model-len", "8192"],
            env={"VLLM_LOGGING_LEVEL": "DEBUG"},
            slurm=SlurmOptions(gpu_type="a100", time_limit="04:00:00"),
        ),
        fast_settings(),
    )
    assert "#SBATCH --partition=gpu\n" in script
    assert "#SBATCH --gres=gpu:a100:1\n" in script
    assert "#SBATCH --time=04:00:00\n" in script
    assert "#SBATCH --output=/scratch/tester/icp/runs/chat/r2/slurm-%j.out\n" in script
    assert "export VLLM_LOGGING_LEVEL=DEBUG\n" in script
    assert "SIF=/scratch/tester/icp/vllm.sif\n" in script
    assert "vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0" in script
    assert script.rstrip().endswith("--served-model-name chat --max-model-len 8192")


def test_sbatch_script_quotes_untrusted_values():
    script = render_sbatch(ref(model="$(touch /tmp/pwned)", vllm_args=["--x=`id`"]), fast_settings())
    assert "'$(touch /tmp/pwned)'" in script
    assert "'--x=`id`'" in script


def test_slurm_options_reject_injection():
    with pytest.raises(ValidationError):
        SlurmOptions(partition="gpu\n#SBATCH --account=other")
    with pytest.raises(ValidationError):
        SlurmOptions(time_limit="2h")


def test_deploy_uploads_and_submits_the_script():
    runner = FakeRunner({"mkdir": ok("4242\n")})
    backend = SlurmBackend(fast_settings(), FakeKube(), runner)

    assert asyncio.run(backend.deploy(ref())) == "4242"
    assert "sbatch --parsable /scratch/tester/icp/runs/chat/r2/job.sbatch" in runner.commands[0]
    assert "rm -f /scratch/tester/icp/runs/chat/r2/endpoint" in runner.commands[0]
    assert runner.inputs[0].startswith("#!/bin/bash")


def test_deploy_rejected_by_slurm():
    runner = FakeRunner({"mkdir": err("sbatch: error: invalid partition specified: gpux")})
    backend = SlurmBackend(fast_settings(), FakeKube(), runner)
    with pytest.raises(DeployError) as info:
        asyncio.run(backend.deploy(ref()))
    assert info.value.reason is FailureReason.UNSCHEDULABLE


def test_deploy_needs_a_cluster_user():
    backend = SlurmBackend(fast_settings(slurm_ssh_user=""), FakeKube(), FakeRunner())
    with pytest.raises(DeployError, match="ICP_SLURM_SSH_USER"):
        asyncio.run(backend.deploy(ref()))


def test_queued_job():
    result = status(FakeRunner({"squeue": ok("PENDING|Priority|\n")}))
    assert result.phase is Phase.PENDING
    assert "Priority" in result.message


def test_queued_behind_a_per_user_limit():
    result = status(FakeRunner({"squeue": ok("PENDING|QOSMaxGRESPerUser|\n")}))
    assert result.phase is Phase.PENDING
    assert "per-user limit" in result.message


def test_job_that_can_never_start():
    result = status(FakeRunner({"squeue": ok("PENDING|PartitionTimeLimit|\n")}))
    assert (result.phase, result.failure) == (Phase.FAILED, FailureReason.UNSCHEDULABLE)


def test_running_job_before_it_publishes_its_port():
    result = status(FakeRunner({"squeue": ok("RUNNING|None|d1001\n"), "cat": err("No such file")}))
    assert result.phase is Phase.STARTING
    assert "publish its port" in result.message


def test_running_job_gets_a_gateway_then_becomes_ready():
    kube = FakeKube()
    runner = FakeRunner({"squeue": ok("RUNNING|None|d1001\n"), "cat": ok("d1001:43123\n")})

    assert status(runner, kube).phase is Phase.STARTING
    gateway = kube.applied[0]
    container = gateway["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container["env"]}
    assert (env["TARGET_HOST"], env["TARGET_PORT"], env["SSH_USER"]) == ("d1001", "43123", "tester")
    assert gateway["spec"]["template"]["metadata"]["labels"]["icp/backend"] == "slurm"
    assert kube.applied[1]["kind"] == "Service"

    kube.deployment = deployment_obj(ready=1)
    kube.pods = [pod(phase="Running")]
    result = status(runner, kube)
    assert result.phase is Phase.READY
    assert result.endpoint == "http://chat-r2.icp-models.svc.cluster.local:8000"


def test_gateway_that_cannot_log_in():
    kube = FakeKube()
    kube.deployment = deployment_obj()
    kube.pods = [
        pod(
            "chat-r2-gw",
            containerStatuses=[{"restartCount": 1, "lastState": {"terminated": {"exitCode": 255}}, "state": {}}],
        )
    ]
    kube.logs["chat-r2-gw"] = "user@login.example: Permission denied (publickey).\n"
    result = status(FakeRunner({"squeue": ok("RUNNING|None|d1001\n"), "cat": ok("d1001:43123\n")}), kube)
    assert result.failure is FailureReason.SSH
    assert "Permission denied" in result.message


def ended(state: str, log: str = "") -> FakeRunner:
    return FakeRunner(
        {
            "squeue": err("slurm_load_jobs error: Invalid job id specified"),
            "sacct": ok(f"{state}\n"),
            "tail": ok(log),
        }
    )


def test_job_out_of_memory():
    assert status(ended("OUT_OF_MEMORY")).failure is FailureReason.OOM_KILLED


def test_job_hit_its_time_limit():
    assert status(ended("TIMEOUT")).failure is FailureReason.TIME_LIMIT


def test_failed_job_is_explained_by_its_log():
    result = status(ended("FAILED", "INFO loading\ntorch.OutOfMemoryError: CUDA out of memory\n"))
    assert result.failure is FailureReason.GPU_OOM
    assert "CUDA out of memory" in result.message


def test_teardown_cancels_the_job_and_removes_the_gateway():
    runner, kube = FakeRunner({"scancel": ok()}), FakeKube()
    asyncio.run(SlurmBackend(fast_settings(), kube, runner).teardown(ref(handle="4242")))
    assert "scancel 4242" in runner.commands
    assert kube.deleted == ["icp/deployment=chat,icp/revision=2"]


def test_parse_squeue():
    assert parse_squeue("") is None
    assert parse_squeue("RUNNING|None|d1001\n") == ("RUNNING", "None", "d1001")
