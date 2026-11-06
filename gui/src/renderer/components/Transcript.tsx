import { useEffect, useRef } from "react";
import type { LineView, NoticeView } from "../lib/session-store";
import { Panel } from "./ui";

const NOTICE_TONE: Record<NoticeView["level"], string> = {
  info: "text-zinc-500",
  warning: "text-amber-300",
  error: "text-red-300",
};

export function Transcript({ lines, notices }: { lines: LineView[]; notices: NoticeView[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [lines, notices]);

  return (
    <Panel title="Transcript" className="min-h-0 flex-1" right={<span className="text-xs text-zinc-500">{lines.length} lines</span>}>
      <div
        ref={scrollRef}
        className="selectable min-h-0 flex-1 space-y-1.5 overflow-y-auto px-4 py-3"
        onScroll={(e) => {
          const el = e.currentTarget;
          stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
        }}
      >
        {notices.slice(-3).map((n) => (
          <div key={`n${n.id}`} className={`text-xs ${NOTICE_TONE[n.level]}`}>
            {n.text}
          </div>
        ))}
        {lines.length === 0 && <div className="pt-8 text-center text-sm text-zinc-600">Waiting for speech…</div>}
        {lines.map((l) => (
          <div key={l.id} className={`flex gap-3 border-l-2 pl-2 ${l.flushed ? "border-accent/50" : "border-transparent"}`}>
            <span className="w-32 shrink-0 pt-0.5 font-mono text-xs text-zinc-600">{l.ts}</span>
            <span className={l.final ? "text-[15px] leading-relaxed text-zinc-100" : "text-[15px] leading-relaxed text-zinc-500"}>
              {l.text}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}
