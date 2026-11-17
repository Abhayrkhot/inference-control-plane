import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import DeployForm from "../components/DeployForm";
import StatusBadge from "../components/StatusBadge";

export default function DeploymentList() {
  const [creating, setCreating] = useState(false);
  const { data, error, isLoading } = useQuery({ queryKey: ["deployments"], queryFn: api.list, refetchInterval: 5000 });

  return (
    <>
      <div className="page-head">
        <h1>Deployments</h1>
        <button className="primary" onClick={() => setCreating((v) => !v)}>
          {creating ? "Close" : "New deployment"}
        </button>
      </div>
      {creating && <DeployForm onDone={() => setCreating(false)} />}
      {isLoading && <p className="muted">Loading…</p>}
      {error && <p className="error">{(error as Error).message}</p>}
      {data?.length === 0 && (
        <div className="empty">
          No deployments yet. Create one here, or run <code>icp deploy demo --model facebook/opt-125m</code>.
        </div>
      )}
      {data && data.length > 0 && (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Serving</th>
                <th>Backend</th>
                <th>Model</th>
                <th>Latest rollout</th>
              </tr>
            </thead>
            <tbody>
              {data.map((d) => {
                const shown = d.active_revision ?? d.latest_revision;
                const latest = d.latest_revision;
                return (
                  <tr key={d.name}>
                    <td>
                      <Link to={`/deployments/${d.name}`}>{d.name}</Link>
                    </td>
                    <td>{d.active_revision ? `r${d.active_revision.number}` : <span className="muted">nothing</span>}</td>
                    <td>{shown?.backend ?? "–"}</td>
                    <td className="mono">{shown?.spec.model ?? "–"}</td>
                    <td>
                      {latest ? (
                        <>
                          <StatusBadge status={latest.status} />{" "}
                          <span className="muted">
                            r{latest.number}
                            {latest.failure_reason ? ` · ${latest.failure_reason}` : ""}
                          </span>
                        </>
                      ) : (
                        "–"
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
