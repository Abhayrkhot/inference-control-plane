import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { api, type DeploymentSpec } from "../api";

interface Props {
  name?: string; // set: roll out a new revision of this deployment; unset: create one
  initial?: DeploymentSpec;
  onDone: () => void;
}

type FormState = ReturnType<typeof toForm>;

function toForm(spec?: DeploymentSpec) {
  return {
    model: spec?.model ?? "facebook/opt-125m",
    backend: spec?.backend ?? "kubernetes",
    image: spec?.image ?? "",
    args: spec?.vllm_args.join(" ") ?? "",
    env: Object.entries(spec?.env ?? {})
      .map(([k, v]) => `${k}=${v}`)
      .join("\n"),
    replicas: String(spec?.replicas ?? 1),
    cpu: spec?.resources.cpu ?? "500m",
    memory: spec?.resources.memory ?? "1Gi",
    gpu: String(spec?.resources.gpu ?? 0),
    partition: spec?.slurm.partition ?? "gpu",
    gpus: String(spec?.slurm.gpus ?? 1),
    gpuType: spec?.slurm.gpu_type ?? "",
    constraint: spec?.slurm.constraint ?? "",
    slurmMem: spec?.slurm.mem ?? "32G",
    timeLimit: spec?.slurm.time_limit ?? "02:00:00",
  };
}

function fromForm(f: FormState): Partial<DeploymentSpec> {
  const env = Object.fromEntries(
    f.env
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const i = line.indexOf("=");
        return i < 0 ? [line, ""] : [line.slice(0, i), line.slice(i + 1)];
      }),
  );
  return {
    model: f.model.trim(),
    backend: f.backend as DeploymentSpec["backend"],
    image: f.image.trim() || null,
    vllm_args: f.args.split(/\s+/).filter(Boolean),
    env,
    replicas: Number(f.replicas),
    resources: { cpu: f.cpu, memory: f.memory, gpu: Number(f.gpu) },
    slurm: {
      partition: f.partition,
      gpus: Number(f.gpus),
      gpu_type: f.gpuType.trim() || null,
      constraint: f.constraint.trim() || null,
      cpus: 8,
      mem: f.slurmMem,
      time_limit: f.timeLimit,
    },
  };
}

export default function DeployForm({ name, initial, onDone }: Props) {
  const queryClient = useQueryClient();
  const [form, setForm] = useState(() => toForm(initial));
  const [newName, setNewName] = useState("");
  const mutation = useMutation({
    mutationFn: async (): Promise<unknown> =>
      name ? api.rollout(name, fromForm(form)) : api.create(newName, fromForm(form)),
    onSuccess: () => {
      queryClient.invalidateQueries();
      onDone();
    },
  });

  const field = (key: keyof FormState) => ({
    value: form[key],
    onChange: (e: { target: { value: string } }) => setForm({ ...form, [key]: e.target.value }),
  });

  function submit(e: FormEvent) {
    e.preventDefault();
    mutation.mutate();
  }

  const slurm = form.backend === "slurm";
  return (
    <form className="panel form" onSubmit={submit}>
      <h2>{name ? `New revision of ${name}` : "New deployment"}</h2>
      <div className="grid">
        {!name && (
          <label>
            Name
            <input required pattern="[a-z]([-a-z0-9]*[a-z0-9])?" placeholder="chat" value={newName}
              onChange={(e) => setNewName(e.target.value)} />
          </label>
        )}
        <label className="wide">
          Model
          <input required {...field("model")} />
        </label>
        <label>
          Backend
          <select {...field("backend")}>
            <option value="kubernetes">Kubernetes</option>
            <option value="slurm">Slurm (HPC)</option>
          </select>
        </label>
        <label className="wide">
          {slurm ? "Apptainer image (.sif path, optional)" : "Container image (optional)"}
          <input {...field("image")} placeholder={slurm ? "/scratch/<you>/icp/vllm.sif" : "icp-fake-vllm:dev"} />
        </label>
        <label className="wide">
          vLLM arguments (space-separated)
          <input {...field("args")} placeholder="--max-model-len 4096" />
        </label>
        {slurm ? (
          <>
            <label>Partition<input {...field("partition")} /></label>
            <label>GPUs<input type="number" min={1} {...field("gpus")} /></label>
            <label>GPU type<input {...field("gpuType")} placeholder="any" /></label>
            <label>Node constraint<input {...field("constraint")} placeholder="a100@80g" /></label>
            <label>Memory<input {...field("slurmMem")} /></label>
            <label>Time limit<input {...field("timeLimit")} /></label>
          </>
        ) : (
          <>
            <label>Replicas<input type="number" min={1} {...field("replicas")} /></label>
            <label>CPU<input {...field("cpu")} /></label>
            <label>Memory<input {...field("memory")} /></label>
            <label>GPUs<input type="number" min={0} {...field("gpu")} /></label>
          </>
        )}
        <label className="wide">
          Environment (KEY=VALUE per line)
          <textarea rows={2} {...field("env")} placeholder="FAKE_VLLM_FAIL=cuda_oom" />
        </label>
      </div>
      {mutation.error && <p className="error">{(mutation.error as Error).message}</p>}
      <div className="actions">
        <button type="submit" className="primary" disabled={mutation.isPending}>
          {mutation.isPending ? "Starting…" : "Start rollout"}
        </button>
        <button type="button" onClick={onDone}>Cancel</button>
      </div>
    </form>
  );
}
