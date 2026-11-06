import { amplitudeToUnit } from "../lib/session-store";

export function Meter({ history, peak, note, paused }: { history: number[]; peak: number; note: string; paused: boolean }) {
  const level = amplitudeToUnit(peak);
  return (
    <div className="flex items-center gap-3 rounded-lg border border-line bg-panel px-4 py-2">
      <div className="flex h-8 flex-1 items-end gap-px overflow-hidden">
        {history.map((u, i) => (
          <div
            key={i}
            className={`flex-1 rounded-sm ${u > 0.85 ? "bg-red-400" : u > 0.65 ? "bg-amber-400" : "bg-emerald-400/80"}`}
            style={{ height: `${Math.max(4, u * 100)}%`, opacity: paused ? 0.3 : 0.9 }}
          />
        ))}
      </div>
      <div className="w-40 text-right text-xs text-zinc-400">
        <div>{paused ? "Paused" : note || "Listening"}</div>
        <div className="font-mono text-zinc-500">{level <= 0 ? "-inf" : `${(level * 60 - 60).toFixed(0)} dBFS`}</div>
      </div>
    </div>
  );
}
