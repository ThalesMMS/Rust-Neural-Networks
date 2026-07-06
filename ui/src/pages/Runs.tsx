import { Fragment, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  ChevronDown,
  ChevronRight,
  FileText,
  GitBranch,
  Search,
  SlidersHorizontal,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Badge from "../components/Badge";
import {
  api,
  type ArtifactPreview,
  type ExperimentSource,
  type ExperimentStatus,
  type ExperimentSummary,
  type GradientPoint,
  type TrainingPoint,
} from "../lib/tauri";

const LINE_COLORS = ["#8166ff", "#34d399", "#f59e0b", "#f472b6", "#38bdf8", "#fb7185"];
const CHART_METRICS = ["loss", "val_accuracy", "learning_rate", "train_time"] as const;
type ChartMetric = (typeof CHART_METRICS)[number];
type SortField = "date" | "label" | "epochs" | "val_loss" | "val_accuracy" | "time" | "status";
type SortDirection = "asc" | "desc";

export default function Runs() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [seriesData, setSeriesData] = useState<Record<string, TrainingPoint[]>>({});
  const [expanded, setExpanded] = useState<string | null>(null);
  const [metric, setMetric] = useState<ChartMetric>("loss");
  const [query, setQuery] = useState("");
  const [modelFilter, setModelFilter] = useState("all");
  const [datasetFilter, setDatasetFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState<ExperimentStatus | "all">("all");
  const [sourceFilter, setSourceFilter] = useState<ExperimentSource | "all">("all");
  const [artifactFilter, setArtifactFilter] = useState("any");
  const [sortField, setSortField] = useState<SortField>("date");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  useEffect(() => {
    api.listExperiments().then(setExperiments);
  }, []);

  useEffect(() => {
    selected.forEach((key) => {
      if (seriesData[key]) return;
      const experiment = experiments.find((item) => item.key === key);
      if (!experiment?.training_log_path) return;
      api.readTrainingSeries(experiment.training_log_path).then((rows) => {
        setSeriesData((prev) => ({ ...prev, [key]: rows }));
      });
    });
  }, [experiments, selected, seriesData]);

  const models = useMemo(
    () => unique(experiments.map((item) => item.model_type).filter(Boolean)).sort(),
    [experiments],
  );
  const datasets = useMemo(
    () => unique(experiments.map((item) => item.dataset_name).filter(Boolean) as string[]).sort(),
    [experiments],
  );

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const rows = experiments.filter((item) => {
      if (modelFilter !== "all" && item.model_type !== modelFilter) return false;
      if (datasetFilter !== "all" && item.dataset_name !== datasetFilter) return false;
      if (statusFilter !== "all" && item.status !== statusFilter) return false;
      if (sourceFilter !== "all" && item.source !== sourceFilter) return false;
      if (artifactFilter === "training" && !item.training_log_path) return false;
      if (artifactFilter === "gradient" && !item.gradient_log_path) return false;
      if (artifactFilter === "warnings" && item.warnings.length === 0) return false;
      if (!needle) return true;
      return searchableText(item).includes(needle);
    });
    return [...rows].sort((a, b) => compareExperiments(a, b, sortField, sortDirection));
  }, [
    artifactFilter,
    datasetFilter,
    experiments,
    modelFilter,
    query,
    sortDirection,
    sortField,
    sourceFilter,
    statusFilter,
  ]);

  const kpis = useMemo(() => buildKpis(experiments), [experiments]);
  const selectedExperiments = selected
    .map((key) => experiments.find((item) => item.key === key))
    .filter((item): item is ExperimentSummary => Boolean(item));

  function toggleSelect(experiment: ExperimentSummary) {
    if (!experiment.training_log_path) return;
    setSelected((prev) =>
      prev.includes(experiment.key)
        ? prev.filter((key) => key !== experiment.key)
        : [...prev, experiment.key].slice(-4),
    );
  }

  function changeSort(field: SortField) {
    if (sortField === field) {
      setSortDirection((current) => (current === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDirection(field === "label" ? "asc" : "desc");
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-8 py-8">
      <header className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold text-zinc-100">Experiments</h1>
          <p className="mt-1 max-w-3xl text-sm text-zinc-500">
            Compare training curves, inspect registry snapshots, and open local artifacts from runs and logs.
          </p>
        </div>
        <Badge tone="accent">{experiments.length} discovered</Badge>
      </header>

      <KpiStrip kpis={kpis} />

      <section className="card mb-5 p-4">
        <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
          <SlidersHorizontal size={14} /> Filters
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-[minmax(180px,1fr)_repeat(5,minmax(120px,160px))]">
          <label className="relative block">
            <Search size={14} className="absolute left-2.5 top-2.5 text-zinc-600" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search model, run, config, command"
              className="w-full rounded-md border border-border bg-surface2 py-2 pl-8 pr-3 text-sm text-zinc-200 outline-none"
            />
          </label>
          <FilterSelect value={modelFilter} onChange={setModelFilter} options={["all", ...models]} />
          <FilterSelect value={datasetFilter} onChange={setDatasetFilter} options={["all", ...datasets]} />
          <FilterSelect
            value={statusFilter}
            onChange={(value) => setStatusFilter(value as ExperimentStatus | "all")}
            options={["all", "completed", "failed", "unknown"]}
          />
          <FilterSelect
            value={sourceFilter}
            onChange={(value) => setSourceFilter(value as ExperimentSource | "all")}
            options={["all", "registry", "log"]}
          />
          <FilterSelect
            value={artifactFilter}
            onChange={setArtifactFilter}
            options={["any", "training", "gradient", "warnings"]}
          />
        </div>
      </section>

      <section className="card mb-5 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
              <BarChart3 size={14} /> Compare
            </div>
            <div className="mt-1 text-xs text-zinc-600">Select up to 4 experiments with metric logs.</div>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {CHART_METRICS.map((name) => (
              <button
                key={name}
                onClick={() => setMetric(name)}
                className={`rounded-md px-2.5 py-1.5 text-xs ${
                  metric === name
                    ? "bg-accent text-white"
                    : "bg-white/5 text-zinc-400 hover:bg-white/10 hover:text-zinc-200"
                }`}
              >
                {metricLabel(name)}
              </button>
            ))}
          </div>
        </div>
        {selectedExperiments.length > 0 ? (
          <TrainingComparisonChart
            metric={metric}
            experiments={selectedExperiments}
            data={seriesData}
          />
        ) : (
          <div className="flex h-56 items-center justify-center rounded-md border border-dashed border-border text-sm text-zinc-600">
            No experiments selected.
          </div>
        )}
      </section>

      <section className="card overflow-hidden">
        <div className="overflow-auto">
          <table className="w-full min-w-[1060px] text-left text-sm">
            <thead className="border-b border-border text-xs uppercase tracking-wider text-zinc-500">
              <tr>
                <th className="w-8 px-3 py-2"></th>
                <th className="w-8 px-3 py-2"></th>
                <SortableHeader label="Model / run" field="label" active={sortField} direction={sortDirection} onClick={changeSort} />
                <SortableHeader label="Date" field="date" active={sortField} direction={sortDirection} onClick={changeSort} />
                <SortableHeader label="Epochs" field="epochs" active={sortField} direction={sortDirection} onClick={changeSort} />
                <SortableHeader label="Val loss" field="val_loss" active={sortField} direction={sortDirection} onClick={changeSort} />
                <SortableHeader label="Best acc" field="val_accuracy" active={sortField} direction={sortDirection} onClick={changeSort} />
                <SortableHeader label="Avg time" field="time" active={sortField} direction={sortDirection} onClick={changeSort} />
                <th className="px-3 py-2">Artifacts</th>
                <SortableHeader label="Status" field="status" active={sortField} direction={sortDirection} onClick={changeSort} />
              </tr>
            </thead>
            <tbody>
              {filtered.map((item) => (
                <Fragment key={item.key}>
                  <tr className="border-b border-border/60 hover:bg-white/[0.02]">
                    <td className="px-3 py-2">
                      <input
                        type="checkbox"
                        checked={selected.includes(item.key)}
                        onChange={() => toggleSelect(item)}
                        disabled={!item.training_log_path}
                        title={item.training_log_path ? "Compare this experiment" : "No metric log"}
                      />
                    </td>
                    <td className="px-3 py-2">
                      <button
                        onClick={() => setExpanded(expanded === item.key ? null : item.key)}
                        className="rounded p-1 text-zinc-500 hover:bg-white/5 hover:text-zinc-200"
                      >
                        {expanded === item.key ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                      </button>
                    </td>
                    <td className="px-3 py-2">
                      <div className="font-medium text-zinc-200">{item.label}</div>
                      <div className="mt-0.5 flex flex-wrap gap-1.5 text-[11px] text-zinc-600">
                        <span>{item.model_type}</span>
                        {item.dataset_name && <span>{item.dataset_name}</span>}
                        {item.config_path && <span className="max-w-[240px] truncate">{item.config_path}</span>}
                      </div>
                    </td>
                    <td className="px-3 py-2 text-zinc-500">{displayDate(item)}</td>
                    <td className="px-3 py-2 tabular-nums text-zinc-300">{item.epochs_completed ?? "-"}</td>
                    <td className="px-3 py-2 tabular-nums text-zinc-300">{fmtNumber(item.best_val_loss)}</td>
                    <td className="px-3 py-2 tabular-nums text-zinc-300">{fmtPercent(item.best_val_accuracy)}</td>
                    <td className="px-3 py-2 tabular-nums text-zinc-300">{fmtSeconds(item.average_epoch_time_seconds)}</td>
                    <td className="px-3 py-2">
                      <ArtifactBadges experiment={item} />
                    </td>
                    <td className="px-3 py-2">
                      <StatusBadge status={item.status} source={item.source} warnings={item.warnings.length} />
                    </td>
                  </tr>
                  {expanded === item.key && (
                    <tr className="border-b border-border/60 bg-white/[0.015]">
                      <td colSpan={10} className="px-4 py-4">
                        <ExperimentDetail experiment={item} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={10} className="px-3 py-10 text-center text-zinc-600">
                    No experiments match the current filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function KpiStrip({ kpis }: { kpis: ReturnType<typeof buildKpis> }) {
  return (
    <div className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <Kpi title="Experiments" value={String(kpis.total)} detail={`${kpis.registry} registry, ${kpis.logs} log only`} />
      <Kpi title="Best accuracy" value={fmtPercent(kpis.bestAccuracy)} detail={kpis.bestLabel ?? "No validation metrics"} />
      <Kpi title="Fastest avg epoch" value={fmtSeconds(kpis.fastestTime)} detail={kpis.fastestLabel ?? "No timing metrics"} />
      <Kpi title="Needs attention" value={String(kpis.issueCount)} detail="failed runs or missing artifacts" />
    </div>
  );
}

function Kpi({ title, value, detail }: { title: string; value: string; detail: string }) {
  return (
    <div className="card p-4">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-600">{title}</div>
      <div className="mt-2 text-2xl font-semibold text-zinc-100">{value}</div>
      <div className="mt-1 truncate text-xs text-zinc-500">{detail}</div>
    </div>
  );
}

function FilterSelect({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: string[];
}) {
  return (
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="rounded-md border border-border bg-surface2 px-2.5 py-2 text-sm text-zinc-200 outline-none"
    >
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  );
}

function SortableHeader({
  label,
  field,
  active,
  direction,
  onClick,
}: {
  label: string;
  field: SortField;
  active: SortField;
  direction: SortDirection;
  onClick: (field: SortField) => void;
}) {
  return (
    <th className="px-3 py-2">
      <button onClick={() => onClick(field)} className="flex items-center gap-1 hover:text-zinc-300">
        {label}
        {active === field && <span className="text-accent-soft">{direction === "asc" ? "up" : "down"}</span>}
      </button>
    </th>
  );
}

function TrainingComparisonChart({
  metric,
  experiments,
  data,
}: {
  metric: ChartMetric;
  experiments: ExperimentSummary[];
  data: Record<string, TrainingPoint[]>;
}) {
  const merged = mergeTrainingRows(metric, experiments, data);
  const keys = Array.from(new Set(merged.flatMap((point) => Object.keys(point).filter((key) => key !== "epoch"))));

  if (merged.length === 0 || keys.length === 0) {
    return (
      <div className="flex h-56 items-center justify-center rounded-md border border-dashed border-border text-sm text-zinc-600">
        Loading metric data.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={merged} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#26263355" vertical={false} />
        <XAxis dataKey="epoch" stroke="#71717a" fontSize={12} tickLine={false} />
        <YAxis stroke="#71717a" fontSize={12} tickLine={false} width={52} />
        <Tooltip
          contentStyle={{ background: "#191924", border: "1px solid #33333f", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "#a1a1aa" }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {keys.map((key, index) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={LINE_COLORS[index % LINE_COLORS.length]}
            strokeWidth={2}
            dot={false}
            strokeDasharray={key.endsWith(" val") ? "4 3" : undefined}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function ExperimentDetail({ experiment }: { experiment: ExperimentSummary }) {
  const [training, setTraining] = useState<TrainingPoint[]>([]);
  const [gradients, setGradients] = useState<GradientPoint[]>([]);
  const [previews, setPreviews] = useState<ArtifactPreview[]>([]);
  const [errors, setErrors] = useState<string[]>([]);

  useEffect(() => {
    setTraining([]);
    setGradients([]);
    setPreviews([]);
    setErrors([]);

    if (experiment.training_log_path) {
      api.readTrainingSeries(experiment.training_log_path).then(setTraining).catch((error) => {
        setErrors((prev) => [...prev, String(error)]);
      });
    }
    if (experiment.gradient_log_path) {
      api.readGradientCsv(experiment.gradient_log_path).then(setGradients).catch((error) => {
        setErrors((prev) => [...prev, String(error)]);
      });
    }

    const previewPaths = unique(
      [
        experiment.training_log_path,
        experiment.gradient_log_path,
        experiment.run_id ? `runs/${experiment.run_id}/run.json` : undefined,
        ...experiment.plots,
      ].filter(Boolean) as string[],
    );
    previewPaths.forEach((path) => {
      api.readArtifactPreview(path, 12000)
        .then((preview) => setPreviews((prev) => [...prev, preview]))
        .catch((error) => setErrors((prev) => [...prev, `${path}: ${String(error)}`]));
    });
  }, [experiment]);

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(360px,0.8fr)]">
      <div className="flex flex-col gap-4">
        <div className="rounded-md border border-border/60 p-3">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Metric curve</div>
          {training.length > 0 ? (
            <TrainingComparisonChart metric="loss" experiments={[experiment]} data={{ [experiment.key]: training }} />
          ) : (
            <EmptyDetail>No metric log loaded.</EmptyDetail>
          )}
        </div>
        <div className="rounded-md border border-border/60 p-3">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Gradient norms</div>
          {gradients.length > 0 ? <GradientChart rows={gradients} /> : <EmptyDetail>No gradient log available.</EmptyDetail>}
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <div className="rounded-md border border-border/60 p-3 text-xs">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Run snapshot</div>
          <dl className="grid grid-cols-[110px_minmax(0,1fr)] gap-x-3 gap-y-1 text-zinc-400">
            <dt className="text-zinc-600">source</dt>
            <dd>{experiment.source}</dd>
            <dt className="text-zinc-600">command</dt>
            <dd className="truncate">{experiment.command ?? "-"}</dd>
            <dt className="text-zinc-600">seed</dt>
            <dd>{experiment.seed ?? "-"}</dd>
            <dt className="text-zinc-600">git</dt>
            <dd className="flex min-w-0 items-center gap-1 truncate">
              <GitBranch size={12} />
              {experiment.environment?.git_commit?.slice(0, 10) ?? "-"}
              {experiment.environment?.git_dirty ? " dirty" : ""}
            </dd>
            <dt className="text-zinc-600">runtime</dt>
            <dd className="truncate">{experiment.environment?.rustc_version ?? "-"}</dd>
          </dl>
          {experiment.checkpoints.length > 0 && (
            <div className="mt-3">
              <div className="mb-1 text-zinc-600">checkpoints</div>
              <div className="flex flex-col gap-1">
                {experiment.checkpoints.map((path) => (
                  <code key={path} className="truncate rounded bg-black/30 px-2 py-1 text-[11px] text-zinc-400">
                    {path}
                  </code>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="rounded-md border border-border/60 p-3">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Config snapshot</div>
          <pre className="max-h-64 overflow-auto rounded-md bg-black/30 p-2 text-[11px] leading-relaxed text-zinc-400">
            {formatConfigSnapshot(experiment)}
          </pre>
        </div>

        {(experiment.warnings.length > 0 || errors.length > 0) && (
          <div className="rounded-md border border-amber-500/20 bg-amber-500/10 p-3 text-xs text-amber-300">
            {[...experiment.warnings, ...errors].map((warning, index) => (
              <div key={`${warning}-${index}`} className="flex gap-2">
                <AlertTriangle size={13} className="mt-0.5 shrink-0" />
                <span>{warning}</span>
              </div>
            ))}
          </div>
        )}

        <div className="rounded-md border border-border/60 p-3">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Artifact previews</div>
          {previews.length > 0 ? (
            <div className="flex flex-col gap-3">
              {previews.map((preview) => (
                <div key={preview.relative_path}>
                  <div className="mb-1 flex items-center justify-between gap-2 text-xs text-zinc-500">
                    <span className="flex min-w-0 items-center gap-1 truncate">
                      <FileText size={12} /> {preview.relative_path}
                    </span>
                    <span className="shrink-0 tabular-nums">
                      {preview.bytes_returned}/{preview.bytes_total} bytes
                    </span>
                  </div>
                  <pre className="max-h-48 overflow-auto rounded-md bg-black/30 p-2 text-[11px] leading-relaxed text-zinc-400">
                    {preview.content}
                    {preview.truncated ? "\n... truncated ..." : ""}
                  </pre>
                </div>
              ))}
            </div>
          ) : (
            <EmptyDetail>No previewable artifacts.</EmptyDetail>
          )}
        </div>
      </div>
    </div>
  );
}

function GradientChart({ rows }: { rows: GradientPoint[] }) {
  const layers = unique(rows.map((row) => row.layer_name));
  const epochs = unique(rows.map((row) => String(row.epoch))).map(Number);
  const data = epochs.map((epoch) => {
    const point: Record<string, number> = { epoch };
    rows
      .filter((row) => row.epoch === epoch)
      .forEach((row) => {
        point[`${row.layer_name} W`] = row.grad_norm_weights;
        point[`${row.layer_name} b`] = row.grad_norm_biases;
      });
    return point;
  });
  const keys = layers.flatMap((layer) => [`${layer} W`, `${layer} b`]);

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#26263355" vertical={false} />
        <XAxis dataKey="epoch" stroke="#71717a" fontSize={12} tickLine={false} />
        <YAxis stroke="#71717a" fontSize={12} tickLine={false} width={52} />
        <Tooltip
          contentStyle={{ background: "#191924", border: "1px solid #33333f", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "#a1a1aa" }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {keys.map((key, index) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={LINE_COLORS[index % LINE_COLORS.length]}
            strokeWidth={2}
            dot={false}
            strokeDasharray={key.endsWith(" b") ? "4 3" : undefined}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function EmptyDetail({ children }: { children: string }) {
  return (
    <div className="flex h-40 items-center justify-center rounded-md border border-dashed border-border text-xs text-zinc-600">
      {children}
    </div>
  );
}

function ArtifactBadges({ experiment }: { experiment: ExperimentSummary }) {
  return (
    <div className="flex flex-wrap gap-1">
      {experiment.training_log_path && <Badge tone="accent">metrics</Badge>}
      {experiment.gradient_log_path && <Badge tone="neutral">gradients</Badge>}
      {experiment.checkpoints.length > 0 && <Badge tone="neutral">{experiment.checkpoints.length} ckpt</Badge>}
      {experiment.warnings.length > 0 && <Badge tone="warn">{experiment.warnings.length} warn</Badge>}
    </div>
  );
}

function StatusBadge({
  status,
  source,
  warnings,
}: {
  status: ExperimentStatus;
  source: ExperimentSource;
  warnings: number;
}) {
  if (status === "completed" && warnings === 0) return <Badge tone="good">{source}</Badge>;
  if (status === "failed") return <Badge tone="bad">failed</Badge>;
  if (warnings > 0) return <Badge tone="warn">check</Badge>;
  return <Badge tone="neutral">{source}</Badge>;
}

function mergeTrainingRows(
  metric: ChartMetric,
  experiments: ExperimentSummary[],
  data: Record<string, TrainingPoint[]>,
) {
  const maxLen = Math.max(0, ...experiments.map((item) => data[item.key]?.length ?? 0));
  const merged: Record<string, number | undefined>[] = [];
  for (let index = 0; index < maxLen; index++) {
    const point: Record<string, number | undefined> = { epoch: index + 1 };
    experiments.forEach((experiment) => {
      const row = data[experiment.key]?.[index];
      if (!row) return;
      if (metric === "loss") {
        point[`${experiment.label} train`] = row.train_loss;
        point[`${experiment.label} val`] = row.val_loss;
      } else if (metric === "val_accuracy") {
        point[experiment.label] = row.val_accuracy;
      } else if (metric === "learning_rate") {
        point[experiment.label] = row.learning_rate;
      } else {
        point[experiment.label] = row.train_time;
      }
    });
    merged.push(point);
  }
  return merged;
}

function buildKpis(items: ExperimentSummary[]) {
  const best = items
    .filter((item) => item.best_val_accuracy !== undefined)
    .sort((a, b) => (b.best_val_accuracy ?? -Infinity) - (a.best_val_accuracy ?? -Infinity))[0];
  const fastest = items
    .filter((item) => item.average_epoch_time_seconds !== undefined)
    .sort((a, b) => (a.average_epoch_time_seconds ?? Infinity) - (b.average_epoch_time_seconds ?? Infinity))[0];
  return {
    total: items.length,
    registry: items.filter((item) => item.source === "registry").length,
    logs: items.filter((item) => item.source === "log").length,
    bestAccuracy: best?.best_val_accuracy,
    bestLabel: best?.label,
    fastestTime: fastest?.average_epoch_time_seconds,
    fastestLabel: fastest?.label,
    issueCount: items.filter((item) => item.status === "failed" || item.warnings.length > 0).length,
  };
}

function compareExperiments(
  a: ExperimentSummary,
  b: ExperimentSummary,
  field: SortField,
  direction: SortDirection,
) {
  const sign = direction === "asc" ? 1 : -1;
  const value = (item: ExperimentSummary) => {
    if (field === "date") return dateValue(item);
    if (field === "label") return item.label.toLowerCase();
    if (field === "epochs") return item.epochs_completed ?? -Infinity;
    if (field === "val_loss") return item.best_val_loss ?? Infinity;
    if (field === "val_accuracy") return item.best_val_accuracy ?? -Infinity;
    if (field === "time") return item.average_epoch_time_seconds ?? Infinity;
    return item.status;
  };
  const av = value(a);
  const bv = value(b);
  if (av < bv) return -1 * sign;
  if (av > bv) return 1 * sign;
  return 0;
}

function searchableText(item: ExperimentSummary) {
  return [
    item.label,
    item.model_type,
    item.dataset_name,
    item.config_path,
    item.training_log_path,
    item.gradient_log_path,
    item.command,
    item.environment?.git_commit,
    ...item.checkpoints,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function formatConfigSnapshot(item: ExperimentSummary) {
  if (item.config_parsed !== undefined) return JSON.stringify(item.config_parsed, null, 2);
  if (item.config_raw) {
    try {
      return JSON.stringify(JSON.parse(item.config_raw), null, 2);
    } catch {
      return item.config_raw;
    }
  }
  return "No registry config snapshot.";
}

function metricLabel(metric: ChartMetric) {
  if (metric === "loss") return "Loss";
  if (metric === "val_accuracy") return "Val accuracy";
  if (metric === "learning_rate") return "Learning rate";
  return "Epoch time";
}

function dateValue(item: ExperimentSummary) {
  if (item.timestamp_start) return Date.parse(item.timestamp_start) || 0;
  return (item.modified_unix_secs ?? 0) * 1000;
}

function displayDate(item: ExperimentSummary) {
  const value = dateValue(item);
  if (!value) return "-";
  return new Date(value).toISOString().slice(0, 19).replace("T", " ");
}

function fmtNumber(value: number | undefined) {
  return value === undefined ? "-" : value.toFixed(4);
}

function fmtPercent(value: number | undefined) {
  return value === undefined ? "-" : `${value.toFixed(2)}%`;
}

function fmtSeconds(value: number | undefined) {
  if (value === undefined) return "-";
  if (value < 60) return `${value.toFixed(2)}s`;
  return `${(value / 60).toFixed(1)}m`;
}

function unique<T>(items: T[]) {
  return Array.from(new Set(items));
}
