"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { SimulationResponse } from "@/lib/types";

interface ProbabilityChartProps {
  readonly result: SimulationResponse | null;
}

interface Datum {
  readonly basis: string;
  readonly probability: number;
}

export const ProbabilityChart = ({ result }: ProbabilityChartProps): JSX.Element => {
  if (!result) {
    return (
      <div className="flex h-72 items-center justify-center rounded-lg border border-graphite-700/70 bg-void-800 text-xs text-graphite-500">
        Run a circuit to see |&alpha;<sub>i</sub>|<sup>2</sup> distribution.
      </div>
    );
  }

  const data: Datum[] = result.basis_labels.map((basis, i) => ({
    basis: `|${basis}\u27E9`,
    probability: result.probabilities[i] ?? 0,
  }));

  return (
    <div className="rounded-lg border border-graphite-700/70 bg-void-800 p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-[0.3em] text-cyan-neon">
          Probability Distribution
        </h2>
        <span className="text-[10px] uppercase tracking-widest text-graphite-500">
          Born rule&nbsp;&middot;&nbsp;|&alpha;<sub>i</sub>|<sup>2</sup>
        </span>
      </div>
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 12, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="#1a1f29" vertical={false} />
            <XAxis
              dataKey="basis"
              stroke="#7c8597"
              tick={{ fill: "#a3acbd", fontSize: 11, fontFamily: "monospace" }}
              axisLine={{ stroke: "#2a3140" }}
              tickLine={false}
              interval={0}
              angle={data.length > 8 ? -45 : 0}
              textAnchor={data.length > 8 ? "end" : "middle"}
              height={data.length > 8 ? 60 : 30}
            />
            <YAxis
              domain={[0, 1]}
              stroke="#7c8597"
              tick={{ fill: "#a3acbd", fontSize: 11 }}
              axisLine={{ stroke: "#2a3140" }}
              tickLine={false}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <Tooltip
              cursor={{ fill: "rgba(34,226,255,0.06)" }}
              contentStyle={{
                background: "#05060a",
                border: "1px solid #0aa0b8",
                borderRadius: 6,
                color: "#22e2ff",
                fontFamily: "monospace",
                fontSize: 12,
              }}
              labelStyle={{ color: "#22e2ff" }}
              formatter={(value: number) => [value.toFixed(6), "p"]}
            />
            <Bar
              dataKey="probability"
              fill="#22e2ff"
              radius={[3, 3, 0, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
