from icp_api.backends.base import Backend
from icp_api.backends.kube import KubeOps
from icp_api.backends.kubernetes import KubernetesBackend
from icp_api.backends.slurm import SlurmBackend
from icp_api.backends.ssh import SSHRunner
from icp_api.config import Settings


def build_backends(settings: Settings) -> dict[str, Backend]:
    """Nothing here connects yet: Kubernetes and SSH connections open on first use."""
    kube = KubeOps(settings.namespace, settings.kube_context)
    runner = SSHRunner(
        settings.slurm_login_host,
        settings.slurm_ssh_user,
        settings.slurm_ssh_key_path,
        settings.slurm_known_hosts_path,
    )
    return {
        "kubernetes": KubernetesBackend(settings, kube),
        "slurm": SlurmBackend(settings, kube, runner),
    }
