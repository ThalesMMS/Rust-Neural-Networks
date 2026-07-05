export default function ProbabilityBars({
  labels,
  probabilities,
}: {
  labels: string[];
  probabilities: number[];
}) {
  const top = probabilities.length ? Math.max(...probabilities) : 0;
  return (
    <div className="flex flex-col gap-1.5">
      {labels.map((label, i) => {
        const p = probabilities[i] ?? 0;
        const isTop = p === top && p > 0;
        return (
          <div key={label} className="flex items-center gap-2 text-xs">
            <div className="w-14 shrink-0 truncate text-zinc-400">{label}</div>
            <div className="h-4 flex-1 overflow-hidden rounded bg-white/5">
              <div
                className={`h-full rounded ${isTop ? "bg-accent" : "bg-accent/40"}`}
                style={{ width: `${Math.max(2, p * 100)}%` }}
              />
            </div>
            <div className={`w-12 shrink-0 text-right tabular-nums ${isTop ? "text-accent-soft" : "text-zinc-500"}`}>
              {(p * 100).toFixed(1)}%
            </div>
          </div>
        );
      })}
    </div>
  );
}
