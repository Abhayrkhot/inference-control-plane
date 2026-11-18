import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, fmtSeconds } from "../api";

const TILES: { key: string; label: string; fmt: (v: number) => string }[] = [
  { key: "queue_depth", label: "Queue depth", fmt: (v) => v.toFixed(0) },
  { key: "running", label: "Running", fmt: (v) => v.toFixed(0) },
  { key: "tokens_per_second", label: "Tokens / s", fmt: (v) => v.toFixed(1) },
];

const PERCENTILES = [50, 90, 95, 99];
const LATENCY_ROWS: { kind: string; label: string }[] = [
  { kind: "latency", label: "Request" },
  { kind: "ttft", label: "Time to first token" },
];

// One measure per chart: these have different units, so they never share an axis.
const CHARTS: { key: string; title: string; fmt: (v: number) => string }[] = [
  { key: "queue_depth", title: "Queue depth", fmt: (v) => v.toFixed(0) },
  { key: "tokens_per_second", title: "Generated tokens / s", fmt: (v) => v.toFixed(1) },
  { key: "p99_latency_seconds", title: "p99 request latency", fmt: fmtSeconds },
];

// Recharts takes colors as attributes, so read the theme tokens from CSS.
function useCssVars(names: string[]): Record<string, string> {
  const read = () =>
    Object.fromEntries(names.map((n) => [n, getComputedStyle(document.documentElement).getPropertyValue(n).trim()]));
  const [values, setValues] = useState(read);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setValues(read());
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return values;
}

// Show seconds when the data covers only a few minutes, or the ticks repeat the same minute.
const timeLabel =
  (withSeconds: boolean) =>
  (t: number): string =>
    new Date(t * 1000).toLocaleTimeString(
      [],
      withSeconds ? { hour: "2-digit", minute: "2-digit", second: "2-digit" } : { hour: "2-digit", minute: "2-digit" },
    );

export default function MetricsPanel({ name }: { name: string }) {
  const colors = useCssVars(["--series-1", "--grid", "--text-secondary", "--surface-1", "--border"]);
  const { data, error } = useQuery({
    queryKey: ["metrics", name],
    queryFn: () => api.metrics(name),
    refetchInterval: 10000,
    retry: false,
  });

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Metrics</h2>
        <span className="muted">from Prometheus · last 15 min</span>
      </div>
      {error ? (
        <p className="muted">{(error as Error).message}</p>
      ) : (
        <>
          <div className="tiles">
            {TILES.map((tile) => {
              const value = data?.current[tile.key];
              return (
                <div key={tile.key} className="tile">
                  <div className="tile-label">{tile.label}</div>
                  <div className="tile-value">{value == null ? "–" : tile.fmt(value)}</div>
                </div>
              );
            })}
          </div>
          <div className="table-wrap">
            <table className="table percentiles">
              <thead>
                <tr>
                  <th>Latency, last 5 min</th>
                  {PERCENTILES.map((p) => (
                    <th key={p}>p{p}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {LATENCY_ROWS.map((row) => (
                  <tr key={row.kind}>
                    <td>{row.label}</td>
                    {PERCENTILES.map((p) => {
                      const value = data?.current[`p${p}_${row.kind}_seconds`];
                      return (
                        <td key={p} className="num">
                          {value == null ? "–" : fmtSeconds(value)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="charts">
            {CHARTS.map((chart) => {
              const rows = (data?.series[chart.key] ?? []).map(([t, v]) => ({ t, v }));
              const shortSpan = rows.length > 1 && rows[rows.length - 1].t - rows[0].t < 600;
              return (
                <figure key={chart.key} className="chart">
                  <figcaption>{chart.title}</figcaption>
                  {rows.length < 2 ? (
                    <div className="chart-empty muted">No samples yet</div>
                  ) : (
                    <ResponsiveContainer width="100%" height={140}>
                      <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                        <CartesianGrid vertical={false} stroke={colors["--grid"]} />
                        <XAxis dataKey="t" tickFormatter={timeLabel(shortSpan)} minTickGap={48}
                          tick={{ fontSize: 11, fill: colors["--text-secondary"] }} stroke={colors["--grid"]} />
                        <YAxis width={60} tickFormatter={chart.fmt} tickLine={false} axisLine={false}
                          tick={{ fontSize: 11, fill: colors["--text-secondary"] }} />
                        <Tooltip
                          labelFormatter={(t) => timeLabel(true)(Number(t))}
                          formatter={(v) => [chart.fmt(Number(v)), chart.title]}
                          contentStyle={{ background: colors["--surface-1"], border: `1px solid ${colors["--border"]}`, borderRadius: 6, fontSize: 12 }}
                        />
                        <Line type="monotone" dataKey="v" stroke={colors["--series-1"]} strokeWidth={2} dot={false}
                          activeDot={{ r: 4 }} isAnimationActive={false} connectNulls={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </figure>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
