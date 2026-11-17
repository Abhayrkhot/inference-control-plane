import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, timingSummary, type Revision } from "../api";
import DeployForm from "../components/DeployForm";
import EventTimeline from "../components/EventTimeline";
import LogViewer from "../components/LogViewer";
import MetricsPanel from "../components/MetricsPanel";
import StatusBadge from "../components/StatusBadge";

function RevisionCard({ title, rev, empty }: { title: string; rev: Revision | null; empty: string }) {
  return (
    <div className="card">
      <div className="card-label">{title}</div>
      {rev ? (
        <>
          <div className="card-title">
            r{rev.number} <StatusBadge status={rev.status} />
          </div>
          <div className="mono">{rev.spec.model}</div>
          <div className="muted">
            {rev.backend}
            {rev.handle ? ` · ${rev.handle}` : ""}
          </div>
          {rev.failure_reason && (
            <div className="error">
              <strong>{rev.failure_reason}</strong>: {rev.message}
            </div>
          )}
          {timingSummary(rev) && <div className="muted">{timingSummary(rev)}</div>}
        </>
      ) : (
        <div className="muted">{empty}</div>
      )}
    </div>
  );
}

export default function DeploymentDetail() {
  const { name = "" } = useParams();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [editing, setEditing] = useState(false);

  const deployment = useQuery({ queryKey: ["deployment", name], queryFn: () => api.get(name), refetchInterval: 3000 });
  const revisions = useQuery({ queryKey: ["revisions", name], queryFn: () => api.revisions(name), refetchInterval: 3000 });
  const rollback = useMutation({
    mutationFn: (revision?: number) => api.rollback(name, revision),
    onSuccess: () => queryClient.invalidateQueries(),
  });
  const remove = useMutation({
    mutationFn: () => api.remove(name),
    onSuccess: () => {
      queryClient.invalidateQueries();
      navigate("/");
    },
  });

  if (deployment.isLoading) return <p className="muted">Loading…</p>;
  if (deployment.error) return <p className="error">{(deployment.error as Error).message}</p>;
  const d = deployment.data!;
  const active = d.active_revision;
  const latest = d.latest_revision;
  const inFlight = latest?.status === "rolling_out" || latest?.status === "pending";
  const actionError = rollback.error ?? remove.error;

  return (
    <>
      <div className="page-head">
        <div>
          <Link to="/" className="muted">
            ← Deployments
          </Link>
          <h1>{d.name}</h1>
        </div>
        <div className="actions">
          <button onClick={() => setEditing((v) => !v)} disabled={inFlight}>
            {editing ? "Close" : "New revision"}
          </button>
          <button onClick={() => rollback.mutate(undefined)} disabled={inFlight || rollback.isPending}>
            Roll back
          </button>
          <button
            className="danger"
            disabled={inFlight || remove.isPending}
            onClick={() => window.confirm(`Delete ${d.name} and tear down what it serves?`) && remove.mutate()}
          >
            Delete
          </button>
        </div>
      </div>
      {actionError && <p className="error">{(actionError as Error).message}</p>}
      {editing && <DeployForm name={d.name} initial={(active ?? latest)?.spec} onDone={() => setEditing(false)} />}

      <section className="cards">
        <RevisionCard title="Serving now" rev={active} empty="Nothing is serving." />
        <RevisionCard title="Latest rollout" rev={latest} empty="No rollouts yet." />
      </section>

      <MetricsPanel name={name} />

      <section className="panel">
        <div className="panel-head">
          <h2>Revisions</h2>
        </div>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Revision</th>
                <th>Status</th>
                <th>Backend</th>
                <th>Model</th>
                <th>vLLM args</th>
                <th>Rollout time</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {(revisions.data ?? []).map((rev) => (
                <tr key={rev.number}>
                  <td className="mono">
                    r{rev.number}
                    {rev.source_revision ? <span className="muted"> (from r{rev.source_revision})</span> : null}
                  </td>
                  <td>
                    <StatusBadge status={rev.status} />
                    {rev.failure_reason && <span className="muted"> {rev.failure_reason}</span>}
                  </td>
                  <td>{rev.backend}</td>
                  <td className="mono">{rev.spec.model}</td>
                  <td className="mono">{rev.spec.vllm_args.join(" ") || "–"}</td>
                  <td className="muted">{timingSummary(rev) || "–"}</td>
                  <td>
                    {rev.status === "superseded" && (
                      <button className="small" disabled={inFlight} onClick={() => rollback.mutate(rev.number)}>
                        Roll back to this
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="split">
        <EventTimeline name={name} live={inFlight} />
        <LogViewer name={name} />
      </div>
    </>
  );
}
