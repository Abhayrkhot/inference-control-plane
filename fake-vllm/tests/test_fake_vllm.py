import pytest
from fastapi.testclient import TestClient

from fake_vllm import cli
from fake_vllm.server import create_app


def test_serves_completions_and_vllm_metrics():
    client = TestClient(create_app("chat", "facebook/opt-125m", tokens_per_second=10_000))
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").json()["data"][0]["id"] == "chat"

    response = client.post("/v1/completions", json={"model": "chat", "prompt": "hi", "max_tokens": 3})
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "lorem lorem lorem"
    assert client.post("/v1/completions", json={"model": "other", "prompt": "hi"}).status_code == 404

    text = client.get("/metrics").text
    assert 'vllm:num_requests_waiting{model_name="chat"} 0.0' in text
    assert 'vllm:num_requests_running{model_name="chat"} 0.0' in text
    assert 'vllm:generation_tokens_total{model_name="chat"} 3.0' in text
    assert "vllm:e2e_request_latency_seconds_bucket" in text


def test_health_waits_for_the_startup_delay():
    assert TestClient(create_app("chat", "m", startup_delay=60)).get("/health").status_code == 503


def test_injected_completion_failure():
    client = TestClient(create_app("chat", "m", fail="completions"))
    assert client.get("/health").status_code == 200
    assert client.post("/v1/completions", json={"model": "chat", "prompt": "hi"}).status_code == 500


def test_startup_failure_prints_the_vllm_error(monkeypatch, capsys):
    monkeypatch.setenv("FAKE_VLLM_FAIL", "cuda_oom")
    monkeypatch.setenv("FAKE_VLLM_EXIT_DELAY", "0")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["serve", "m", "--served-model-name", "chat", "--max-model-len", "4096"])
    assert exit_info.value.code == 1
    out = capsys.readouterr()
    assert "CUDA out of memory" in out.err
    assert "--max-model-len" in out.out
