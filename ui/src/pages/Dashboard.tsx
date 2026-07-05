import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertTriangle, ArrowRight, CheckCircle2, XCircle } from "lucide-react";
import { api, type DataStatus, type ModelDescriptor } from "../lib/tauri";
import Badge from "../components/Badge";

export default function Dashboard() {
  const [models, setModels] = useState<ModelDescriptor[]>([]);
  const [dataStatus, setDataStatus] = useState<DataStatus | null>(null);
  const [trained, setTrained] = useState<Record<string, boolean>>({});
  const navigate = useNavigate();

  useEffect(() => {
    api.listModels().then(setModels);
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
