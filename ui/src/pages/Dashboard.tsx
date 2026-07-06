import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertTriangle, ArrowRight, BarChart3, CheckCircle2, XCircle } from "lucide-react";
import { api, type DataStatus, type ExperimentSummary, type ModelDescriptor } from "../lib/tauri";
import Badge from "../components/Badge";

export default function Dashboard() {
  const [models, setModels] = useState<ModelDescriptor[]>([]);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [dataStatus, setDataStatus] = useState<DataStatus | null>(null);
  const [trained, setTrained] = useState<Record<string, boolean>>({});
  const navigate = useNavigate();

  useEffect(() => {
    api.listModels().then(setModels);
    api.listExperiments().then(setExperiments);
    api.dataStatus().then(setDataStatus);
  }, []);

  useEffect(() => {
    if (models.length === 0) return;
    const firstCheckpoints = models.map((m) => m.checkpoints[0] ?? "");
    api.checkpointStatus(firstCheckpoints).then((results) => {
      const map: Record<string, boolean> = {};
      models.forEach((m, i) => {
        map[m.id] = m.checkpoints.length > 0 && results[i];
      });
      setTrained(map);
    });
  }, [models]);

  const grouped = useMemo(() => {
    const byCategory = new Map<string, ModelDescriptor[]>();
    for (const m of models) {
      const list = byCategory.get(m.category) ?? [];
      list.push(m);
      byCategory.set(m.category, list);
    }
    return Array.from(byCategory.entries());
  }, [models]);

  const recentExperiments = useMemo(
    () => [...experiments].sort((a, b) => dateValue(b) - dateValue(a)).slice(0, 4),
    [experiments],
  );

  const bestExperiments = useMemo(() => {
    const bestByModel = new Map<string, ExperimentSummary>();
    for (const experiment of experiments) {
      if (experiment.best_val_accuracy === undefined) continue;
      const previous = bestByModel.get(experiment.model_type);
      if (!previous || (experiment.best_val_accuracy ?? -Infinity) > (previous.best_val_accuracy ?? -Infinity)) {
        bestByModel.set(experiment.model_type, experiment);
      }
    }
    return Array.from(bestByModel.values())
      .sort((a, b) => (b.best_val_accuracy ?? -Infinity) - (a.best_val_accuracy ?? -Infinity))
      .slice(0, 4);
  }, [experiments]);

  return (
    <div className="mx-auto max-w-6xl px-8 py-8">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold text-zinc-100">Dashboard</h1>
        <p className="mt-1 text-sm text-zinc-500">
          {models.length} models discovered in this repo. Pick one to configure and train, or head to the
          Playground once something's trained.
        </p>
      </header>

      {dataStatus && (
        <div className="mb-8 flex gap-3">
          <div className="card flex items-center gap-2 px-4 py-2.5 text-sm">
            {dataStatus.mnist_present ? (
              <CheckCircle2 size={16} className="text-emerald-400" />
            ) : (
              <XCircle size={16} className="text-rose-400" />
            )}
            MNIST data {dataStatus.mnist_present ? "present" : "missing"}
          </div>
          <div className="card flex items-center gap-2 px-4 py-2.5 text-sm">
            {dataStatus.cifar10_present ? (
              <CheckCircle2 size={16} className="text-emerald-400" />
            ) : (
              <XCircle size={16} className="text-rose-400" />
            )}
            CIFAR-10 data {dataStatus.cifar10_present ? "present" : "missing"}
          </div>
        </div>
      )}

      <section className="mb-8 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="card p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
              <BarChart3 size={14} /> Recent experiments
            </div>
            <button
              onClick={() => navigate("/runs")}
              className="flex items-center gap-1 rounded-md bg-white/5 px-2 py-1 text-xs text-zinc-300 hover:bg-white/10"
            >
              Open <ArrowRight size={12} />
            </button>
          </div>
          {recentExperiments.length > 0 ? (
            <div className="flex flex-col divide-y divide-border/60">
              {recentExperiments.map((experiment) => (
                <button
                  key={experiment.key}
                  onClick={() => navigate("/runs")}
                  className="flex items-center justify-between gap-3 py-2 text-left hover:text-zinc-100"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm text-zinc-200">{experiment.label}</div>
                    <div className="truncate text-xs text-zinc-600">{displayDate(experiment)}</div>
                  </div>
                  <Badge tone={experiment.warnings.length > 0 ? "warn" : experiment.status === "completed" ? "good" : "neutral"}>
                    {experiment.source}
                  </Badge>
                </button>
              ))}
            </div>
          ) : (
            <div className="py-6 text-sm text-zinc-600">No experiments discovered yet.</div>
          )}
        </div>

        <div className="card p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Best models</div>
            <Badge tone="accent">{experiments.length} total</Badge>
          </div>
          {bestExperiments.length > 0 ? (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {bestExperiments.map((experiment) => (
                <button
                  key={experiment.key}
                  onClick={() => navigate("/runs")}
                  className="rounded-md border border-border/60 p-3 text-left hover:border-accent/50"
                >
                  <div className="truncate text-sm font-medium text-zinc-200">{experiment.model_type}</div>
                  <div className="mt-2 text-xl font-semibold tabular-nums text-zinc-100">
                    {fmtPercent(experiment.best_val_accuracy)}
                  </div>
                  <div className="mt-1 truncate text-xs text-zinc-600">{experiment.label}</div>
                </button>
              ))}
            </div>
          ) : (
            <div className="py-6 text-sm text-zinc-600">No validation metrics found.</div>
          )}
        </div>
      </section>

      <div className="flex flex-col gap-8">
        {grouped.map(([category, items]) => (
          <section key={category}>
            <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">{category}</h2>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {items.map((m) => {
                const dataOk =
                  m.data_requirement === "none" ||
                  (m.data_requirement === "mnist" && dataStatus?.mnist_present) ||
                  (m.data_requirement === "cifar10" && dataStatus?.cifar10_present);
                return (
                  <button
                    key={m.id}
                    onClick={() => navigate(`/train/${m.id}`)}
                    className="card group flex flex-col gap-3 p-4 text-left transition-colors hover:border-accent/50"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="font-medium text-zinc-100">{m.display_name}</div>
                      <ArrowRight
                        size={16}
                        className="mt-0.5 shrink-0 text-zinc-600 transition-transform group-hover:translate-x-0.5 group-hover:text-accent-soft"
                      />
                    </div>
                    <p className="text-xs leading-relaxed text-zinc-500">{m.description}</p>
                    <div className="mt-auto flex flex-wrap gap-1.5 pt-1">
                      {trained[m.id] ? (
                        <Badge tone="good">trained</Badge>
                      ) : m.checkpoints.length > 0 ? (
                        <Badge tone="neutral">not trained yet</Badge>
                      ) : (
                        <Badge tone="neutral">no checkpoint saved</Badge>
                      )}
                      {!dataOk && <Badge tone="bad">data missing</Badge>}
                      {m.caveats.length > 0 && (
                        <Badge tone="warn">
                          <AlertTriangle size={11} /> {m.caveats.length} note{m.caveats.length > 1 ? "s" : ""}
                        </Badge>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function dateValue(experiment: ExperimentSummary) {
  if (experiment.timestamp_start) return Date.parse(experiment.timestamp_start) || 0;
  return (experiment.modified_unix_secs ?? 0) * 1000;
}

function displayDate(experiment: ExperimentSummary) {
  const value = dateValue(experiment);
  if (!value) return "-";
  return new Date(value).toISOString().slice(0, 19).replace("T", " ");
}

function fmtPercent(value: number | undefined) {
  return value === undefined ? "-" : `${value.toFixed(2)}%`;
}
