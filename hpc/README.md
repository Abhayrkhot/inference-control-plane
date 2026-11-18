# Running models on the HPC cluster (Slurm)

The Slurm backend submits each revision as an `sbatch` job that runs vLLM in Apptainer
on a GPU node, then reaches it through an SSH tunnel held by a gateway pod in kind.
Do these steps by hand once before using `--backend slurm`.

Defaults assume Northeastern's Explorer cluster (`login.explorer.northeastern.edu`).
Partition names, GPU types and time limits differ between clusters, so check them.

## 1. Key-based SSH login

On Explorer, Research Computing issues your SSH key; you do not make your own. From off
campus the login node only accepts that key: there is no password login, and keys you add
to `~/.ssh/authorized_keys` yourself are not used, so `ssh-copy-id` does not work.

1. In Open OnDemand (https://ood.explorer.northeastern.edu), open **Files → Home Directory**,
   turn on **Show Dotfiles**, open `.ssh`, and download `<user>-<date>` and `<user>-<date>.pub`.
2. Install them and check that login needs no password or passphrase:

```bash
mv ~/Downloads/<user>-<date> ~/Downloads/<user>-<date>.pub ~/.ssh/
chmod 600 ~/.ssh/<user>-<date>
ssh -i ~/.ssh/<user>-<date> -o BatchMode=yes <user>@login.explorer.northeastern.edu hostname
```

The last command must print a hostname without asking for anything. Then set
`ICP_SLURM_SSH_USER` and `ICP_SLURM_SSH_KEY_PATH` in `.env`.

If the Open OnDemand shell fails with `Load key ... Permission denied`, or apps fail with
`Disk quota exceeded`, your issued key file may be owned by root. Ask rchelp@northeastern.edu
to regenerate it.

## 2. Look at what you can use

```bash
sinfo -o "%P %a %l %G"                  # partitions, time limits, GPUs
sacctmgr -nP show assoc user=$USER format=partition,qos
module avail apptainer
```

Note the GPU partition name and its time limit, and whether Apptainer needs `module load`
(if so, set `ICP_SLURM_MODULE_LOAD=apptainer`). `sbatch --test-only -p <partition>
--gres=gpu:1 -t 01:00:00 --wrap true` tells you whether a partition accepts your jobs
without submitting anything.

On Explorer at the time of writing: `gpu` (8 h limit, up to 4 GPUs per user), `gpu-short`
(2 h), and `courses-gpu` (1 day, up to 2 GPUs, course accounts). Apptainer is at
`/usr/bin/apptainer`, so no module is needed. Jobs are charged to your default Slurm
account (`sacctmgr show user $USER`).

## 3. Build the vLLM image in scratch, then check it on a GPU

Copy the two job scripts to the cluster, then submit the build and a GPU check that runs
only if the build succeeds:

```bash
scp hpc/pull-vllm-sif.sbatch hpc/gpu-check.sbatch <user>@login.explorer.northeastern.edu:/scratch/<user>/icp/
```

```bash
cd /scratch/$USER/icp
sbatch pull-vllm-sif.sbatch <vllm tag>                     # writes vllm.sif; prints a job id
sbatch --dependency=afterok:<that job id> gpu-check.sbatch
```

The build unpacks the image on the compute node's local disk; unpacking it on scratch takes
hours. `gpu-check.sbatch` tells you whether the image needs CUDA's compatibility libraries
on this cluster's driver (its output says `mode=plain` or `mode=compat`), then serves a small
model and sends it a completion. If it says `mode=compat`, set `ICP_SLURM_CUDA_COMPAT=true`.

On Explorer (checked with vLLM v0.34.0 on an 80 GB A100, driver 570): `mode=compat` is
needed. Qwen2.5-1.5B-Instruct was ready about 100 s after the job started, with weights
loading in about 13 s. Two Explorer details the job scripts already handle:

- Apptainer mounts your home directory, so packages in `~/.local` (for example a PyTorch
  installed for coursework) would shadow the image's own; the scripts set `PYTHONNOUSERSITE=1`.
- Compute nodes reach the internet through an HTTP proxy, so requests to the node itself
  need `no_proxy=localhost,127.0.0.1`; the scripts set it.

## 4. Serve a model by hand (optional; `gpu-check.sbatch` does this for you)

```bash
srun -p gpu-short --gres=gpu:a100:1 --constraint=a100@80g --mem=32G --time=01:00:00 --pty bash
export HF_HUB_CACHE=/scratch/$USER/icp/hf/hub
export PYTHONNOUSERSITE=1    # don't let packages in ~/.local shadow the image's PyTorch
apptainer exec --nv --bind /scratch/$USER/icp /scratch/$USER/icp/vllm.sif \
  vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

Current vLLM images are built on CUDA 13 for GPUs of compute capability 7.5 and newer, so
on Explorer use A100 or H200; V100 and P100 are not supported. `--constraint=a100@80g`
picks the 80 GB A100s (plain `a100` can also land on a 40 GB card). Gated models (Llama)
need `huggingface-cli login` on the cluster first.

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
make hpc-secret HPC_USER=<user> HPC_KEY=~/.ssh/<user>-<date>
```

This stores the key, the login node's host key (from your `~/.ssh/known_hosts`) and the
username as the `icp-hpc-ssh` secret in both namespaces, and `make images` builds the
gateway image. Then set `ICP_SLURM_SSH_USER=<user>` in `.env` and:

```bash
icp deploy qwen --backend slurm --model Qwen/Qwen2.5-1.5B-Instruct \
  --partition gpu-short --gpu-type a100 --constraint a100@80g --time-limit 01:00:00
```

## Being a good cluster citizen

- Keep time limits short and delete deployments you are not using (`icp delete`), which
  cancels the job.
- Blue/green needs two jobs during a rollout. If your account allows one GPU at a time,
  the new job waits behind the old one until the queue timeout.
- Check Research Computing's policy on long-running services before leaving one up.
- Scratch may be purged; the job re-downloads weights if the cache is gone.
