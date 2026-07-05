import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Play, Square, Save, ChevronDown } from "lucide-react";
import {
  api,
  onTrainingFinished,
  onTrainingLog,
  type ConfigEntry,
  type CsvRow,
  type LogLineEvent,
  type ModelDescriptor,
} from "../lib/tauri";
import { SECTIONS, type ConfigValue } from "../lib/trainingConfigSchema";
import Badge from "../components/Badge";
import LogConsole from "../components/LogConsole";
import LiveChart from "../components/LiveChart";

const POLL_MS = 1200;

export default function Train() {
  const { modelId } = useParams();
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelDescriptor[]>([]);
  const [configEntries, setConfigEntries] = useState<ConfigEntry[]>([]);

  const [configPath, setConfigPath] = useState<string | null>(null);
  const [archPath, setArchPath] = useState<string | null>(null);
  const [configObj, setConfigObj] = useState<ConfigValue | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [step, setStep] = useState(false);
  const [runName, setRunName] = useState("");

  const [runToken, setRunToken] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [lines, setLines] = useState<LogLineEvent[]>([]);
  const [rows, setRows] = useState<CsvRow[]>([]);
  const [finishedMessage, setFinishedMessage] = useState<{ success: boolean; message: string } | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    api.listModels().then(setModels);
    api.listConfigs().then(setConfigEntries);
  }, []);

  const model = useMemo(() => models.find((m) => m.id === modelId) ?? null, [models, modelId]);

  useEffect(() => {
    if (!model) return;
    setConfigPath(model.default_config_path);
    setArchPath(model.default_arch_path);
    setConfigError(null);
    setSaveMessage(null);
    setRows([]);
    setLines([]);
    setFinishedMessage(null);
  }, [model]);

  useEffect(() => {
    if (!configPath) {
      setConfigObj(null);
      return;
    }
    api
      .readConfig(configPath)
      .then((raw) => {
        setConfigObj(JSON.parse(raw));
        setConfigError(null);
      })
      .catch((e) => setConfigError(String(e)));
  }, [configPath]);

  useEffect(() => {
    const unlistenLog = onTrainingLog((payload) => {
      setLines((prev) => (prev.length > 2000 ? [...prev.slice(-1500), payload] : [...prev, payload]));
    });
    const unlistenFinished = onTrainingFinished((payload) => {
      setRunToken((current) => {
        if (current !== payload.run_token) return current;
        setRunning(false);
        setFinishedMessage({ success: payload.success, message: payload.message });
        return current;
      });
    });
    return () => {
      unlistenLog.then((f) => f());
      unlistenFinished.then((f) => f());
    };
  }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  if (!model && models.length > 0 && !modelId) {
    return <ModelPicker models={models} onPick={(id) => navigate(`/train/${id}`)} />;
  }
  if (!model) {
    return <div className="p-8 text-sm text-zinc-500">Loading…</div>;
  }

  const configOptions = configEntries
    .map((c) => c.relative_path)
    .sort((a, b) => {
      const aMatch = a.includes(model.bin_name) ? 0 : 1;
      const bMatch = b.includes(model.bin_name) ? 0 : 1;
      return aMatch - bMatch || a.localeCompare(b);
    });
  const archOptions = configEntries.filter((c) => c.group === "architectures").map((c) => c.relative_path);

  function setField(key: string, value: unknown) {
    setConfigObj((prev) => (prev ? { ...prev, [key]: value } : prev));
  }

  async function handleSave() {
    if (!configPath || !configObj) return;
    try {
      await api.writeConfig(configPath, JSON.stringify(configObj, null, 2));
      setSaveMessage("Saved.");
      setConfigError(null);
    } catch (e) {
      setConfigError(String(e));
    }
  }

  async function handleRun() {
    if (!model) return;
    setLines([]);
    setRows([]);
    setFinishedMessage(null);
    const started = await api.startRun({
      model_id: model.id,
      config_path: configPath,
      arch_path: model.default_arch_path ? archPath : null,
      step,
      run_name: runName || null,
    });
    setRunToken(started.run_token);
    setRunning(true);

    if (pollRef.current) window.clearInterval(pollRef.current);
    let activeLog: string | null = null;
    pollRef.current = window.setInterval(async () => {
      if (!activeLog) {
        activeLog = await api.findActiveLog(started.started_unix_secs);
      }
      if (activeLog) {
        try {
          const csvRows = await api.readLogCsv(activeLog);
          setRows(csvRows);
        } catch {
          // log file may not exist yet
        }
      }
    }, POLL_MS);
  }

  async function handleStop() {
    if (!runToken) return;
    await api.stopRun(runToken);
  }

  return (
    <div className="mx-auto max-w-5xl px-8 py-8">
      <header className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold text-zinc-100">{model.display_name}</h1>
          <p className="mt-1 max-w-2xl text-sm text-zinc-500">{model.description}</p>
          {model.caveats.map((c) => (
            <Badge key={c} tone="warn">
              {c}
            </Badge>
          ))}
        </div>
        <div className="flex gap-2">
          {!running ? (
            <button
              onClick={handleRun}
              className="flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-dim"
            >
              <Play size={15} /> Run
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="flex items-center gap-1.5 rounded-lg bg-rose-600 px-4 py-2 text-sm font-medium text-white hover:bg-rose-700"
            >
              <Square size={15} /> Stop
            </button>
          )}
        </div>
      </header>

      {finishedMessage && (
        <div
          className={`mb-4 rounded-lg px-4 py-2.5 text-sm ${
            finishedMessage.success
              ? "bg-emerald-500/10 text-emerald-400"
              : "bg-rose-500/10 text-rose-400"
          }`}
        >
          {finishedMessage.success ? "Run finished successfully." : "Run failed."} {finishedMessage.message}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="flex flex-col gap-4">
          <div className="card p-4">
            <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
              Live progress
            </div>
            <LiveChart series={[{ label: model.display_name, rows }]} metric="loss" />
          </div>
          <LogConsole lines={lines} />
        </div>

        <div className="flex flex-col gap-4">
          <div className="card p-4">
            <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Config file</div>
            {model.default_config_path === null ? (
              <p className="text-sm text-zinc-500">
                This model has no config file — hyperparameters are fixed in source.
              </p>
            ) : (
              <>
                <SelectField value={configPath ?? ""} onChange={setConfigPath} options={configOptions} />
                {model.default_arch_path && (
                  <div className="mt-2">
                    <div className="mb-1 text-[11px] text-zinc-500">Architecture file (--arch)</div>
                    <SelectField value={archPath ?? ""} onChange={setArchPath} options={archOptions} />
                  </div>
                )}
              </>
            )}
            {model.supports_step && (
              <label className="mt-3 flex items-center gap-2 text-sm text-zinc-300">
                <input type="checkbox" checked={step} onChange={(e) => setStep(e.target.checked)} />
                Step-through debug mode
              </label>
            )}
            {model.supports_registry_flags && (
              <div className="mt-3">
                <div className="mb-1 text-[11px] text-zinc-500">Run name (optional)</div>
                <input
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                  placeholder="e.g. baseline-adamw"
                  className="w-full rounded-lg border border-border bg-surface2 px-3 py-1.5 text-sm text-zinc-200"
                />
              </div>
            )}
          </div>

          {configObj && (
            <div className="card flex max-h-[60vh] flex-col p-4">
              <div className="mb-2 flex items-center justify-between">
                <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Hyperparameters</div>
                <button
                  onClick={handleSave}
                  className="flex items-center gap-1 rounded-md bg-white/5 px-2 py-1 text-xs text-zinc-300 hover:bg-white/10"
                >
                  <Save size={12} /> Save
                </button>
              </div>
              {configError && <p className="mb-2 text-xs text-rose-400">{configError}</p>}
              {saveMessage && <p className="mb-2 text-xs text-emerald-400">{saveMessage}</p>}
              <div className="flex-1 overflow-y-auto pr-1">
                {SECTIONS.filter((s) => !s.onlyForModelIds || s.onlyForModelIds.includes(model.id)).map(
                  (section) => (
                    <FormSection key={section.title} section={section} config={configObj} onChange={setField} />
                  ),
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ModelPicker({ models, onPick }: { models: ModelDescriptor[]; onPick: (id: string) => void }) {
  return (
    <div className="mx-auto max-w-3xl px-8 py-8">
      <h1 className="mb-4 text-2xl font-semibold text-zinc-100">Choose a model to train</h1>
      <div className="grid grid-cols-2 gap-3">
        {models.map((m) => (
          <button
            key={m.id}
            onClick={() => onPick(m.id)}
            className="card p-3 text-left text-sm text-zinc-200 hover:border-accent/50"
          >
            {m.display_name}
          </button>
        ))}
      </div>
    </div>
  );
}

function SelectField({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full appearance-none rounded-lg border border-border bg-surface2 px-3 py-1.5 pr-8 text-sm text-zinc-200"
      >
        {!options.includes(value) && value && <option value={value}>{value}</option>}
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
      <ChevronDown size={14} className="pointer-events-none absolute right-2.5 top-2.5 text-zinc-500" />
    </div>
  );
}

function FormSection({
  section,
  config,
  onChange,
}: {
  section: (typeof SECTIONS)[number];
  config: ConfigValue;
  onChange: (key: string, value: unknown) => void;
}) {
  const visibleFields = section.fields.filter((f) => !f.showIf || f.showIf(config));
  if (visibleFields.length === 0) return null;
  return (
    <div className="mb-4">
      <div className="mb-1.5 text-[13px] font-medium text-zinc-300">{section.title}</div>
      {section.description && <p className="mb-2 text-[11px] text-zinc-600">{section.description}</p>}
      <div className="grid grid-cols-2 gap-2">
        {visibleFields.map((f) => (
          <label key={f.key} className={`flex flex-col gap-1 text-xs ${f.type === "boolean" ? "col-span-2 flex-row items-center" : ""}`}>
            {f.type !== "boolean" && <span className="text-zinc-500">{f.label}</span>}
            {f.type === "select" && (
              <select
                value={String(config[f.key] ?? f.options?.[0]?.value ?? "")}
                onChange={(e) => onChange(f.key, e.target.value)}
                className="rounded-md border border-border bg-surface2 px-2 py-1 text-zinc-200"
              >
                {f.options?.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            )}
            {f.type === "number" && (
              <input
                type="number"
                step={f.step ?? (f.integer ? 1 : "any")}
                min={f.min}
                max={f.max}
                value={config[f.key] === undefined ? "" : String(config[f.key])}
                onChange={(e) =>
                  onChange(f.key, e.target.value === "" ? undefined : Number(e.target.value))
                }
                className="rounded-md border border-border bg-surface2 px-2 py-1 text-zinc-200"
              />
            )}
            {f.type === "boolean" && (
              <>
                <input
                  type="checkbox"
                  checked={Boolean(config[f.key])}
                  onChange={(e) => onChange(f.key, e.target.checked)}
                  className="mr-2"
                />
                <span className="text-zinc-400">{f.label}</span>
              </>
            )}
          </label>
        ))}
      </div>
    </div>
  );
}
