# Running models on the HPC cluster (Slurm)

The Slurm backend submits each revision as an `sbatch` job that runs vLLM in Apptainer
on a GPU node, then reaches it through an SSH tunnel held by a gateway pod in kind.
Do these steps by hand once before using `--backend slurm`.

Defaults assume Northeastern's Explorer cluster (`login.explorer.northeastern.edu`).
Partition names, GPU types and time limits differ between clusters, so check them.

## 1. Key-based SSH login

The control plane runs unattended, so it needs a key without a passphrase. Use a key just
for this project, so you can revoke it without touching your normal one.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/icp_hpc -C icp-control-plane -N ''
ssh-copy-id -i ~/.ssh/icp_hpc.pub <user>@login.explorer.northeastern.edu
ssh -i ~/.ssh/icp_hpc -o BatchMode=yes <user>@login.explorer.northeastern.edu hostname
```

The last command must print a hostname without asking for anything.

## 2. Look at what you can use

```bash
sinfo -o "%P %a %l %G"                  # partitions, time limits, GPUs
sacctmgr -nP show assoc user=$USER format=partition,qos
module avail apptainer
```

Note the GPU partition name and its time limit, and whether Apptainer needs `module load`
(if so, set `ICP_SLURM_MODULE_LOAD=apptainer`).

## 3. Build the vLLM image in scratch

```bash
sbatch hpc/pull-vllm-sif.sbatch <vllm tag>    # writes /scratch/$USER/icp/vllm.sif
```

## 4. Serve a model by hand

```bash
srun -p <gpu partition> --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
export HF_HUB_CACHE=/scratch/$USER/icp/hf/hub
apptainer exec --nv --bind /scratch/$USER/icp /scratch/$USER/icp/vllm.sif \
  vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

V100 GPUs have no bfloat16, so add `--dtype half` there. Gated models (Llama) need
`huggingface-cli login` on the cluster first.

From your Mac, in another terminal, tunnel through the login node and try it:

```bash
ssh -N -L 8000:<compute node>:8000 <user>@login.explorer.northeastern.edu
curl localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen2.5-1.5B-Instruct", "prompt": "Hello", "max_tokens": 8}'
curl -s localhost:8000/metrics | grep vllm:num_requests
```

When that works, the Slurm backend will too.

## 5. Let the cluster run the gateway

```bash
make hpc-secret HPC_USER=<user>
```

This stores the key, the login node's host key (from your `~/.ssh/known_hosts`) and the
username as the `icp-hpc-ssh` secret in both namespaces, and `make images` builds the
gateway image. Then set `ICP_SLURM_SSH_USER=<user>` in `.env` and:

```bash
icp deploy qwen --backend slurm --model Qwen/Qwen2.5-1.5B-Instruct --time-limit 01:00:00
```

## Being a good cluster citizen

- Keep time limits short and delete deployments you are not using (`icp delete`), which
  cancels the job.
- Blue/green needs two jobs during a rollout. If your account allows one GPU at a time,
  the new job waits behind the old one until the queue timeout.
- Check Research Computing's policy on long-running services before leaving one up.
- Scratch may be purged; the job re-downloads weights if the cache is gone.
