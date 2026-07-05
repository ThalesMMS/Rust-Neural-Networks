import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import type { CsvRow } from "../lib/tauri";

const LINE_COLORS = ["#8166ff", "#34d399", "#f59e0b", "#f472b6", "#38bdf8"];

export interface ChartSeries {
  label: string;
  rows: CsvRow[];
}

export default function LiveChart({
  series,
  metric,
}: {
  series: ChartSeries[];
  metric: "loss" | "val_accuracy";
}) {
  const merged: Record<string, number | undefined>[] = [];
  const maxLen = Math.max(0, ...series.map((s) => s.rows.length));
  for (let i = 0; i < maxLen; i++) {
    const point: Record<string, number | undefined> = { epoch: i + 1 };
    series.forEach((s) => {
      const row = s.rows[i];
      if (!row) return;
      if (metric === "loss") {
        point[`${s.label} train`] = row.train_loss;
        point[`${s.label} val`] = row.val_loss;
      } else {
        point[`${s.label} val acc`] = row.val_accuracy;
      }
    });
    merged.push(point);
  }

  const keys = Array.from(
    new Set(merged.flatMap((p) => Object.keys(p).filter((k) => k !== "epoch"))),
  );

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={merged} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#26263355" vertical={false} />
        <XAxis dataKey="epoch" stroke="#71717a" fontSize={12} tickLine={false} />
        <YAxis stroke="#71717a" fontSize={12} tickLine={false} width={44} />
        <Tooltip
          contentStyle={{ background: "#191924", border: "1px solid #33333f", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "#a1a1aa" }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {keys.map((key, i) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={LINE_COLORS[i % LINE_COLORS.length]}
            strokeWidth={2}
            dot={false}
            strokeDasharray={key.endsWith("val") ? "4 3" : undefined}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
