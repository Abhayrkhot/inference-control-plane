from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every setting comes from an ICP_* environment variable or the .env file."""

    model_config = SettingsConfigDict(env_prefix="ICP_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./icp.db"
    prometheus_url: str = "http://localhost:9090"

    # Kubernetes
    kube_context: str = ""  # kubeconfig context to use outside the cluster; empty = current context
    namespace: str = "icp-models"
    # How the control plane reaches a revision's Service. In-cluster DNS by default. When the
    # API runs on your Mac, run `kubectl proxy --port 8001` and use the service proxy instead:
    # http://127.0.0.1:8001/api/v1/namespaces/{namespace}/services/{service}:http/proxy
    service_url_template: str = "http://{service}.{namespace}.svc.cluster.local:8000"
    default_image: str = "icp-fake-vllm:dev"
    model_cache_pvc: str = "icp-model-cache"
    hf_token_secret: str = "hf-token"

    # Rollouts
    poll_interval_s: float = 3.0
    queue_timeout_s: float = 1800.0  # allowed time waiting for resources (unschedulable pod, queued job)
    startup_timeout_s: float = 900.0  # allowed time from "starting" to "ready" (model download + load)
    smoke_timeout_s: float = 60.0

    # Slurm. Defaults point at Northeastern's Explorer cluster.
    slurm_login_host: str = "login.explorer.northeastern.edu"
    slurm_ssh_user: str = ""
    slurm_ssh_key_path: str = "~/.ssh/icp_hpc"
    slurm_known_hosts_path: str = "~/.ssh/known_hosts"
    slurm_root_dir: str = "/scratch/{user}/icp"  # run dirs, the vLLM .sif image, the Hugging Face cache
    slurm_module_load: str = ""  # e.g. "apptainer" if the cluster needs `module load` first
    gateway_image: str = "icp-gateway:dev"
    gateway_ssh_secret: str = "icp-hpc-ssh"

    @property
    def slurm_root(self) -> str:
        return self.slurm_root_dir.format(user=self.slurm_ssh_user)


@lru_cache
def get_settings() -> Settings:
    return Settings()
