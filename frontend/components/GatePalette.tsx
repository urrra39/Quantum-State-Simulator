"use client";

import { CONTROLLED_GATES, SINGLE_GATES, type GateName } from "@/lib/types";

interface GatePaletteProps {
  readonly onSelect: (gate: GateName) => void;
  readonly selected: GateName | null;
}

const COLORS: Record<GateName, string> = {
  I: "bg-graphite-700 text-graphite-300 border-graphite-500",
  X: "bg-cyan-deep text-cyan-glow border-cyan-dim",
  Y: "bg-cyan-deep text-cyan-glow border-cyan-dim",
  Z: "bg-cyan-deep text-cyan-glow border-cyan-dim",
  H: "bg-cyan-deep text-cyan-neon border-cyan-glow shadow-neon-soft",
  S: "bg-graphite-900 text-cyan-neon border-cyan-dim",
  T: "bg-graphite-900 text-cyan-neon border-cyan-dim",
  CNOT: "bg-graphite-900 text-amber-warn border-amber-warn",
  CZ: "bg-graphite-900 text-amber-warn border-amber-warn",
  CY: "bg-graphite-900 text-amber-warn border-amber-warn",
  CH: "bg-graphite-900 text-amber-warn border-amber-warn",
  CCX: "bg-graphite-900 text-amber-warn border-amber-warn",
};

const Tile = ({
  name,
  selected,
  onSelect,
}: {
  readonly name: GateName;
  readonly selected: boolean;
  readonly onSelect: (g: GateName) => void;
}): JSX.Element => {
  const handleDragStart = (e: React.DragEvent<HTMLButtonElement>): void => {
    e.dataTransfer.setData("application/x-tensorq-gate", name);
    e.dataTransfer.effectAllowed = "copy";
  };

  return (
    <button
      type="button"
      draggable
      onDragStart={handleDragStart}
      onClick={() => onSelect(name)}
      className={[
        "h-12 w-full rounded-md border text-sm font-bold tracking-widest",
        "transition hover:shadow-neon-soft hover:-translate-y-[1px]",
        COLORS[name],
        selected ? "ring-2 ring-cyan-glow shadow-neon" : "ring-0",
      ].join(" ")}
      aria-pressed={selected}
      aria-label={`Quantum gate ${name}`}
    >
      {name}
    </button>
  );
};

export const GatePalette = ({ onSelect, selected }: GatePaletteProps): JSX.Element => {
  return (
    <aside className="rounded-lg border border-graphite-700/70 bg-void-800 p-4">
      <h2 className="mb-3 text-xs uppercase tracking-[0.3em] text-cyan-neon">
        Gate Palette
      </h2>

      <div className="mb-2 text-[10px] uppercase tracking-widest text-graphite-500">
        Single-qubit
      </div>
      <div className="grid grid-cols-4 gap-2">
        {SINGLE_GATES.map((g) => (
          <Tile key={g} name={g} selected={selected === g} onSelect={onSelect} />
        ))}
      </div>

      <div className="mb-2 mt-5 text-[10px] uppercase tracking-widest text-graphite-500">
        Controlled
      </div>
      <div className="grid grid-cols-3 gap-2">
        {CONTROLLED_GATES.map((g) => (
          <Tile key={g} name={g} selected={selected === g} onSelect={onSelect} />
        ))}
      </div>

      <p className="mt-4 text-[11px] leading-relaxed text-graphite-500">
        Drag a gate onto a qubit wire, or click a tile then click a wire cell.
        For controlled gates, click another row in the same column to toggle a
        control.
      </p>
    </aside>
  );
};
