import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { SessionState } from "../lib/session-store";
import { Panel } from "./ui";

function Md({ children }: { children: string }) {
  return (
    <div className="markdown selectable">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{ a: ({ href, children: c }) => <a href={href} target="_blank" rel="noreferrer">{c}</a> }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

export function Insights({
  insights,
  pending,
  llmOffline,
}: {
  insights: SessionState["insights"];
  pending: SessionState["pendingInsight"];
  llmOffline: boolean;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [insights.length, pending]);

  return (
    <Panel title="Insights" className="min-h-0 flex-1">
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
        {llmOffline && (
          <p className="text-sm text-zinc-500">
            LLM unavailable: configure API keys in `.env` to enable insights and chat. The transcript still works.
          </p>
        )}
        {!llmOffline && insights.length === 0 && !pending && (
          <p className="text-sm text-zinc-500">Insights appear after enough of the meeting has been transcribed.</p>
        )}
        {insights.map((p) => (
          <article key={p.n} className="rounded-md border border-line bg-panel-2 p-3">
            <div className="mb-1 flex items-center gap-2 text-xs text-zinc-500">
              <span className="font-mono">#{p.n}</span>
              <span className="font-mono">{p.stamp}</span>
              {p.kind === "wiki" ? (
                <span className="rounded bg-accent/15 px-1.5 py-0.5 text-accent" title="From your Lumina knowledge vault">
                  From {p.vault || "knowledge vault"}
                </span>
              ) : (
                <span className="rounded bg-black/30 px-1.5 py-0.5 text-zinc-400">
                  {p.kind === "comment" ? "Comment" : "Research"}
                </span>
              )}
            </div>
            <Md>{p.body_md}</Md>
            {p.web_md && (
              <details className="mt-2 text-sm">
                <summary className="cursor-pointer text-xs text-zinc-400">{p.kind === "wiki" ? "Vault pages" : "Web search"}</summary>
                <Md>{p.web_md}</Md>
              </details>
            )}
          </article>
        ))}
        {pending && (
          <div className="animate-pulse text-sm text-zinc-500">
            Generating pass {pending.n} @ {pending.stamp}…
          </div>
        )}
        <div ref={endRef} />
      </div>
    </Panel>
  );
}
