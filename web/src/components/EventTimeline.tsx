import { useQuery } from "@tanstack/react-query";
import { api, clockTime } from "../api";

const TONE: Record<string, string> = {
  failed: "bad",
  teardown_failed: "bad",
  retire_failed: "bad",
  rollback: "warn",
  status_error: "warn",
  activated: "good",
  smoke_test_passed: "good",
  ready: "good",
};

export default function EventTimeline({ name, live }: { name: string; live: boolean }) {
  const { data, error } = useQuery({
    queryKey: ["events", name],
    queryFn: () => api.events(name),
    refetchInterval: live ? 1500 : 10000,
  });
  const events = [...(data ?? [])].reverse(); // newest first

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Rollout events</h2>
        {live && <span className="muted">updating</span>}
      </div>
      {error && <p className="error">{(error as Error).message}</p>}
      <ol className="timeline">
        {events.map((e) => (
          <li key={e.id} className={`event event-${TONE[e.kind] ?? "info"}`}>
            <span className="event-time mono">{clockTime(e.at)}</span>
            <span className="event-rev mono">r{e.revision}</span>
            <span className="event-kind">{e.kind.replaceAll("_", " ")}</span>
            <span className="event-message">{e.message}</span>
          </li>
        ))}
        {events.length === 0 && <li className="muted">No events yet.</li>}
      </ol>
    </section>
  );
}
