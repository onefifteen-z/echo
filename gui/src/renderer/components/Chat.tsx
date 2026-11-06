import { useEffect, useRef, useState } from "react";
import type { ChatItem } from "../lib/session-store";
import { NotebookEditor } from "./NotebookEditor";

interface Props {
  items: ChatItem[];
  disabled: boolean;
  onAsk: (q: string) => void;
  notebook: { docKey: string; initial: string; onSave: (text: string) => Promise<unknown> };
}

export function ChatNotebookPanel({ items, disabled, onAsk, notebook }: Props) {
  const [tab, setTab] = useState<"chat" | "notebook">("chat");
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [items, tab]);

  return (
    <section className="flex h-80 min-h-0 shrink-0 flex-col rounded-lg border border-line bg-panel">
      <header className="flex items-center gap-1 border-b border-line px-2">
        {(
          [
            ["chat", "Ask about this meeting"],
            ["notebook", "My notebook"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`-mb-px border-b-2 px-2 py-2 text-xs font-semibold tracking-wide uppercase ${
              tab === id ? "border-accent text-zinc-200" : "border-transparent text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {label}
          </button>
        ))}
      </header>
      <div className={tab === "chat" ? "flex min-h-0 flex-1 flex-col" : "hidden"}>
        <div className="selectable min-h-0 flex-1 space-y-2 overflow-y-auto px-4 py-3 text-sm">
          {items.map((c) => (
            <div key={c.id} className="space-y-1">
              <div className="text-cyan-300">You: {c.question}</div>
              <div className={c.error ? "text-red-300" : "text-zinc-200"}>
                {c.answer === null ? <span className="animate-pulse text-zinc-500">Thinking…</span> : c.answer}
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </div>
        <form
          className="border-t border-line p-2"
          onSubmit={(e) => {
            e.preventDefault();
            const q = draft.trim();
            if (!q) return;
            setDraft("");
            onAsk(q);
          }}
        >
          <input
            className="w-full rounded-md border border-line bg-panel-2 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-accent disabled:opacity-50"
            placeholder={disabled ? "LLM unavailable — chat disabled" : "Ask a question…"}
            disabled={disabled}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
        </form>
      </div>
      <div className={tab === "notebook" ? "flex min-h-0 flex-1 flex-col" : "hidden"}>
        <NotebookEditor
          docKey={notebook.docKey}
          initial={notebook.initial}
          onSave={notebook.onSave}
          placeholder="Your own notes during the meeting (Markdown). Included in the summary."
        />
      </div>
    </section>
  );
}
