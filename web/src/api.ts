// Types mirror api/src/icp_api/schemas.py.

export type RevisionStatus = "pending" | "rolling_out" | "active" | "failed" | "superseded";

export interface DeploymentSpec {
  model: string;
  backend: "kubernetes" | "slurm";
  image?: string | null;
  vllm_args: string[];
  env: Record<string, string>;
  replicas: number;
  resources: { cpu: string; memory: string; gpu: number };
  slurm: {
    partition: string;
    gpus: number;
    gpu_type?: string | null;
    cpus: number;
    mem: string;
    time_limit: string;
  };
}

export interface Revision {
  number: number;
  backend: string;
  status: RevisionStatus;
  spec: DeploymentSpec;
  handle: string | null;
  failure_reason: string | null;
  message: string | null;
  source_revision: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  queue_seconds: number | null;
  startup_seconds: number | null;
  gate_seconds: number | null;
  total_seconds: number | null;
}

export interface Deployment {
  name: string;
  created_at: string;
  active_revision: Revision | null;
  latest_revision: Revision | null;
}

export interface RolloutEvent {
  id: number;
  revision: number;
  at: string;
  kind: string;
  message: string;
}

export interface Metrics {
  queries: Record<string, string>;
  current: Record<string, number | null>;
  series: Record<string, [number, number | null][]>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // not JSON; keep the status text
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  list: () => request<Deployment[]>("/deployments"),
  get: (name: string) => request<Deployment>(`/deployments/${name}`),
  revisions: (name: string) => request<Revision[]>(`/deployments/${name}/revisions`),
  events: (name: string) => request<RolloutEvent[]>(`/deployments/${name}/events`),
  metrics: (name: string) => request<Metrics>(`/deployments/${name}/metrics`),
  create: (name: string, spec: Partial<DeploymentSpec>) =>
    request<Deployment>("/deployments", post({ name, spec })),
  rollout: (name: string, spec: Partial<DeploymentSpec>) =>
    request<Revision>(`/deployments/${name}/rollouts`, post(spec)),
  rollback: (name: string, revision?: number) =>
    request<Revision>(`/deployments/${name}/rollback`, post({ revision: revision ?? null })),
  remove: (name: string) => request<void>(`/deployments/${name}`, { method: "DELETE" }),
};

export function fmtSeconds(value: number | null | undefined): string {
  if (value == null) return "–";
  return value < 1 ? `${Math.round(value * 1000)} ms` : `${value.toFixed(1)} s`;
}

export function timingSummary(rev: Revision): string {
  const parts: [string, number | null][] = [
    ["queue", rev.queue_seconds],
    ["startup", rev.startup_seconds],
    ["test", rev.gate_seconds],
    ["total", rev.total_seconds],
  ];
  return parts
    .filter(([, v]) => v != null)
    .map(([label, v]) => `${label} ${fmtSeconds(v)}`)
    .join(" · ");
}

export function clockTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
