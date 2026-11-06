import type { SidecarState } from "@shared/types";
import { Button } from "./ui";

export function SidecarBanner({ state }: { state: SidecarState }) {
  if (state.phase === "ready") return null;
  if (state.phase === "starting") {
    return <div className="no-drag border-b border-line bg-panel px-4 py-2 text-sm text-zinc-400">Starting Echo engine…</div>;
  }
  const failed = state.phase === "failed";
  return (
    <div className="no-drag border-b border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="font-semibold">
            {failed ? "Echo engine stopped and could not be restarted." : "Echo engine crashed — restarting…"}
          </div>
          <pre className="selectable mt-1 max-h-40 overflow-auto font-mono text-xs whitespace-pre-wrap text-red-300/80">{state.message}</pre>
        </div>
        {failed && (
          <Button variant="danger" onClick={() => void window.echo.restartSidecar()}>
            Restart engine
          </Button>
        )}
      </div>
    </div>
  );
}
