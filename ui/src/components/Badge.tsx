import type { ReactNode } from "react";

type Tone = "neutral" | "good" | "warn" | "bad" | "accent";

const TONE_CLASSES: Record<Tone, string> = {
  neutral: "bg-white/5 text-zinc-400",
  good: "bg-emerald-500/15 text-emerald-400",
  warn: "bg-amber-500/15 text-amber-400",
  bad: "bg-rose-500/15 text-rose-400",
  accent: "bg-accent/15 text-accent-soft",
};

export default function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`badge ${TONE_CLASSES[tone]}`}>{children}</span>;
}
