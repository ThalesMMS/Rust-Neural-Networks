import { useEffect, useRef } from "react";
import type { LogLineEvent } from "../lib/tauri";

// stderr is colored as caution (amber), not danger (red): cargo's build
// progress and rustc warnings are normal stderr traffic, not failures. Actual
// run failures are signaled by the finished-run banner, not per-line color.
const STREAM_CLASSES: Record<LogLineEvent["stream"], string> = {
  stdout: "text-zinc-300",
  stderr: "text-amber-400/90",
  system: "text-accent-soft",
};

export default function LogConsole({ lines }: { lines: LogLineEvent[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [lines.length]);

  return (
    <div className="card h-72 overflow-y-auto p-3 font-mono text-[12.5px] leading-relaxed">
      {lines.length === 0 ? (
        <div className="text-zinc-600">Console output will appear here once you start a run.</div>
      ) : (
        lines.map((l, i) => (
          <div key={i} className={STREAM_CLASSES[l.stream]}>
            {l.line}
          </div>
        ))
      )}
      <div ref={bottomRef} />
    </div>
  );
}
