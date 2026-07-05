import { Fragment, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { api, type CsvRow, type LogSummary, type RunRecord } from "../lib/tauri";
import LiveChart from "../components/LiveChart";
import Badge from "../components/Badge";

interface UnifiedRow {
  key: string;
  label: string;
  date: string;
  epochs: number | undefined;
  valLoss: number | undefined;
  valAccuracy: number | undefined;
  logPath: string | undefined;
  status: "completed" | "failed" | "unknown";
  registry?: RunRecord;
}

function fromRegistry(r: RunRecord): UnifiedRow {
  return {
    key: `registry:${r.run_id}`,
    label: r.run_name ? `${r.model_type} (${r.run_name})` : r.model_type,
    date: r.timestamp_start,
    epochs: r.metrics?.epochs_completed,
    valLoss: r.metrics?.final_val_loss,
    valAccuracy: r.metrics?.final_val_accuracy,
    logPath: r.artifacts?.training_log_csv,
    status: r.status,
    registry: r,
  };
}

function fromLog(l: LogSummary): UnifiedRow {
  return {
    key: `log:${l.relative_path}`,
    label: l.file_name.replace(/\.csv$/, ""),
    date: new Date(l.modified_unix_secs * 1000).toISOString(),
    epochs: l.last_epoch,
    valLoss: l.last_val_loss,
    valAccuracy: l.last_val_accuracy,
    logPath: l.relative_path,
    status: "unknown",
  };
}

export default function Runs() {
  const [registryRuns, setRegistryRuns] = useState<RunRecord[]>([]);
  const [logs, setLogs] = useState<LogSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [seriesData, setSeriesData] = useState<Record<string, CsvRow[]>>({});
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    api.listRuns().then(setRegistryRuns);
    api.listAllLogs().then(setLogs);
  }, []);

  const rows = useMemo(() => {
    const claimed = new Set(registryRuns.map((r) => r.artifacts?.training_log_csv).filter(Boolean));
    const logRows = logs.filter((l) => !claimed.has(l.relative_path)).map(fromLog);
    return [...registryRuns.map(fromRegistry), ...logRows].sort((a, b) => (a.date < b.date ? 1 : -1));
  }, [registryRuns, logs]);

  function toggleSelect(row: UnifiedRow) {
    setSelected((prev) => {
      const next = prev.includes(row.key) ? prev.filter((k) => k !== row.key) : [...prev, row.key].slice(-4);
      return next;
    });
  }

  useEffect(() => {
    selected.forEach((key) => {
      if (seriesData[key]) return;
      const row = rows.find((r) => r.key === key);
      if (!row?.logPath) return;
      api.readLogCsv(row.logPath).then((csvRows) => {
        setSeriesData((prev) => ({ ...prev, [key]: csvRows }));
      });
    });
  }, [selected, rows, seriesData]);

  const chartSeries = selected
    .map((key) => {
      const row = rows.find((r) => r.key === key);
      return row ? { label: row.label, rows: seriesData[key] ?? [] } : null;
    })
    .filter((s): s is { label: string; rows: CsvRow[] } => Boolean(s));

  return (
    <div className="mx-auto max-w-6xl px-8 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold text-zinc-100">Runs</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Select up to 4 runs to compare their loss/accuracy curves. Rows from the experiment registry
          (<code className="text-zinc-400">runs/</code>) include full config snapshots; other models show up
          from their raw <code className="text-zinc-400">logs/*.csv</code> files.
        </p>
      </header>

      {chartSeries.length > 0 && (
        <div className="card mb-6 p-4">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Validation loss comparison
          </div>
          <LiveChart series={chartSeries} metric="loss" />
        </div>
      )}

      <div className="card overflow-hidden">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border text-xs uppercase tracking-wider text-zinc-500">
            <tr>
              <th className="w-8 px-3 py-2"></th>
              <th className="px-3 py-2">Model / run</th>
              <th className="px-3 py-2">Date</th>
              <th className="px-3 py-2">Epochs</th>
              <th className="px-3 py-2">Val loss</th>
              <th className="px-3 py-2">Val accuracy</th>
              <th className="px-3 py-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <Fragment key={row.key}>
                <tr className="border-b border-border/60 hover:bg-white/[0.02]">
                  <td className="px-3 py-2">
                    <input
                      type="checkbox"
                      checked={selected.includes(row.key)}
                      onChange={() => toggleSelect(row)}
                      disabled={!row.logPath}
                    />
                  </td>
                  <td className="px-3 py-2">
                    <button
                      onClick={() => setExpanded(expanded === row.key ? null : row.key)}
                      className="flex items-center gap-1 text-zinc-200 hover:text-accent-soft"
                    >
                      {row.registry ? (
                        expanded === row.key ? (
                          <ChevronDown size={13} />
                        ) : (
                          <ChevronRight size={13} />
                        )
                      ) : (
                        <span className="w-[13px]" />
                      )}
                      {row.label}
                    </button>
                  </td>
                  <td className="px-3 py-2 text-zinc-500">{row.date.slice(0, 19).replace("T", " ")}</td>
                  <td className="px-3 py-2 tabular-nums text-zinc-300">{row.epochs ?? "—"}</td>
                  <td className="px-3 py-2 tabular-nums text-zinc-300">
                    {row.valLoss !== undefined ? row.valLoss.toFixed(4) : "—"}
                  </td>
                  <td className="px-3 py-2 tabular-nums text-zinc-300">
                    {row.valAccuracy !== undefined ? `${row.valAccuracy.toFixed(2)}%` : "—"}
                  </td>
                  <td className="px-3 py-2">
                    {row.status === "completed" && <Badge tone="good">completed</Badge>}
                    {row.status === "failed" && <Badge tone="bad">failed</Badge>}
                    {row.status === "unknown" && <Badge tone="neutral">log only</Badge>}
                  </td>
                </tr>
                {expanded === row.key && row.registry && (
                  <tr className="border-b border-border/60 bg-white/[0.015]">
                    <td colSpan={7} className="px-4 py-3">
                      <RunDetail run={row.registry} />
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={7} className="px-3 py-8 text-center text-zinc-600">
                  No runs yet — train a model to see it here.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RunDetail({ run }: { run: RunRecord }) {
  return (
    <div className="grid grid-cols-2 gap-4 text-xs">
      <div>
        <div className="mb-1 font-semibold text-zinc-400">Config snapshot</div>
        <pre className="max-h-56 overflow-auto rounded-md bg-black/30 p-2 text-[11px] text-zinc-400">
          {JSON.stringify(run.config.parsed ?? run.config.raw ?? {}, null, 2)}
        </pre>
      </div>
      <div className="flex flex-col gap-1 text-zinc-400">
        <div className="mb-1 font-semibold text-zinc-400">Environment</div>
        <div>seed: {run.seed}</div>
        {run.environment?.git?.commit && <div>commit: {run.environment.git.commit.slice(0, 10)}</div>}
        {run.environment?.rustc_version && <div>{run.environment.rustc_version}</div>}
        {run.environment?.os && <div>{run.environment.os}</div>}
        {run.artifacts?.checkpoints.map((c) => (
          <div key={c} className="truncate">
            checkpoint: {c}
          </div>
        ))}
      </div>
    </div>
  );
}
