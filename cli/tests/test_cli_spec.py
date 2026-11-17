import pytest
import typer

from icp_cli.main import build_spec, parse_env


def test_new_spec_from_flags():
    spec = build_spec(
        None,
        model="m",
        backend="slurm",
        args=["--max-model-len=4096"],
        env=["A=1"],
        partition="gpu",
        gpus=2,
        time_limit="01:00:00",
    )
    assert spec == {
        "model": "m",
        "backend": "slurm",
        "vllm_args": ["--max-model-len=4096"],
        "env": {"A": "1"},
        "slurm": {"partition": "gpu", "gpus": 2, "time_limit": "01:00:00"},
    }


def test_flags_overlay_the_current_spec():
    current = {
        "model": "m",
        "backend": "kubernetes",
        "vllm_args": ["--a"],
        "env": {"A": "1"},
        "resources": {"cpu": "1", "memory": "2Gi", "gpu": 0},
    }
    spec = build_spec(current, env=["B=2"], memory="4Gi")
    assert spec["vllm_args"] == ["--a"]
    assert spec["env"] == {"A": "1", "B": "2"}
    assert spec["resources"] == {"cpu": "1", "memory": "4Gi", "gpu": 0}
    assert current["env"] == {"A": "1"}  # not mutated
    assert build_spec(current, clear_args=True)["vllm_args"] == []


def test_env_needs_an_equals_sign():
    with pytest.raises(typer.BadParameter):
        parse_env(["NOEQUALS"])
