import { useEffect, useMemo, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { json } from "@codemirror/lang-json";
import { Save, Copy } from "lucide-react";
import { api, type ConfigEntry } from "../lib/tauri";

export default function Configs() {
  const [entries, setEntries] = useState<ConfigEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState("");
  const [dirty, setDirty] = useState(false);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const [saveAsPath, setSaveAsPath] = useState<string | null>(null);

  useEffect(() => {
    api.listConfigs().then((list) => {
      setEntries(list);
      if (list.length > 0) setSelected(list[0].relative_path);
    });
  }, []);

  useEffect(() => {
    if (!selected) return;
    api.readConfig(selected).then((raw) => {
      setContent(raw);
      setDirty(false);
      setMessage(null);
    });
  }, [selected]);

  const grouped = useMemo(() => {
    const byGroup = new Map<string, ConfigEntry[]>();
    for (const e of entries) {
      const list = byGroup.get(e.group) ?? [];
      list.push(e);
      byGroup.set(e.group, list);
    }
    return Array.from(byGroup.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [entries]);

  async function handleSave() {
    if (!selected) return;
    try {
      await api.writeConfig(selected, content);
      setDirty(false);
      setMessage({ tone: "ok", text: "Saved." });
    } catch (e) {
      setMessage({ tone: "error", text: String(e) });
    }
  }

  async function handleSaveAs() {
    if (!saveAsPath) return;
    try {
      await api.saveConfigAs(saveAsPath, content);
      setMessage({ tone: "ok", text: `Saved as ${saveAsPath}.` });
      const list = await api.listConfigs();
      setEntries(list);
      setSelected(saveAsPath);
      setSaveAsPath(null);
    } catch (e) {
      setMessage({ tone: "error", text: String(e) });
    }
  }

  return (
    <div className="flex h-full">
      <div className="w-72 shrink-0 overflow-y-auto border-r border-border px-4 py-6">
        <h1 className="mb-4 text-lg font-semibold text-zinc-100">Configs</h1>
        {grouped.map(([group, items]) => (
          <div key={group} className="mb-4">
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">{group}</div>
            <div className="flex flex-col gap-0.5">
              {items.map((e) => (
                <button
                  key={e.relative_path}
                  onClick={() => setSelected(e.relative_path)}
                  className={`truncate rounded-md px-2 py-1 text-left text-xs ${
                    selected === e.relative_path
                      ? "bg-accent/15 text-accent-soft"
                      : "text-zinc-400 hover:bg-white/5 hover:text-zinc-200"
                  }`}
                >
                  {e.file_name}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="flex flex-1 flex-col px-6 py-6">
        {selected ? (
          <>
            <div className="mb-3 flex items-center justify-between">
              <div className="font-mono text-sm text-zinc-300">{selected}</div>
              <div className="flex gap-2">
                <button
                  onClick={() => setSaveAsPath(selected.replace(/\.json$/, "_copy.json"))}
                  className="flex items-center gap-1 rounded-md bg-white/5 px-2.5 py-1.5 text-xs text-zinc-300 hover:bg-white/10"
                >
                  <Copy size={12} /> Save as…
                </button>
                <button
                  onClick={handleSave}
                  disabled={!dirty}
                  className="flex items-center gap-1 rounded-md bg-accent px-2.5 py-1.5 text-xs font-medium text-white disabled:opacity-40"
                >
                  <Save size={12} /> Save
                </button>
              </div>
            </div>

            {saveAsPath !== null && (
              <div className="mb-3 flex items-center gap-2 rounded-lg border border-border bg-surface2 p-2">
                <input
                  value={saveAsPath}
                  onChange={(e) => setSaveAsPath(e.target.value)}
                  className="flex-1 bg-transparent font-mono text-xs text-zinc-200 outline-none"
                />
                <button
                  onClick={handleSaveAs}
                  className="rounded-md bg-accent px-2 py-1 text-xs font-medium text-white"
                >
                  Confirm
                </button>
                <button
                  onClick={() => setSaveAsPath(null)}
                  className="rounded-md bg-white/5 px-2 py-1 text-xs text-zinc-400"
                >
                  Cancel
                </button>
              </div>
            )}

            {message && (
              <div className={`mb-3 text-xs ${message.tone === "ok" ? "text-emerald-400" : "text-rose-400"}`}>
                {message.text}
              </div>
            )}

            <div className="flex-1 overflow-hidden rounded-lg border border-border">
              <CodeMirror
                value={content}
                height="100%"
                theme="dark"
                extensions={[json()]}
                onChange={(value) => {
                  setContent(value);
                  setDirty(true);
                }}
                style={{ height: "100%", fontSize: 13 }}
              />
            </div>
          </>
        ) : (
          <div className="text-sm text-zinc-500">No config files found under config/.</div>
        )}
      </div>
    </div>
  );
}
