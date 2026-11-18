import type { RevisionStatus } from "../api";

const LABELS: Record<RevisionStatus, string> = {
  pending: "pending",
  rolling_out: "rolling out",
  active: "serving",
  failed: "failed",
  superseded: "replaced",
  stopped: "stopped",
};

// The dot carries the status color; the label always says it in words.
export default function StatusBadge({ status }: { status: RevisionStatus }) {
  return (
    <span className={`badge badge-${status}`}>
      <span className="badge-dot" aria-hidden="true" />
      {LABELS[status]}
    </span>
  );
}
