import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type SaveState = "idle" | "pending" | "saving" | "saved" | "error";

const SAVE_LABEL: Record<SaveState, string> = {
  idle: "",
  pending: "Editing…",
  saving: "Saving…",
  saved: "Saved",
  error: "Save failed",
};

interface Props {
  /** Text loaded from the engine / disk. Changing `docKey` resets the editor to this value. */
  initial: string;
  docKey: string;
  onSave: (text: string) => Promise<unknown>;
  placeholder?: string;
  readOnly?: boolean;
  debounceMs?: number;
}

export function NotebookEditor({ initial, docKey, onSave, placeholder, readOnly, debounceMs = 800 }: Props) {
  const [text, setText] = useState(initial);
  const [mode, setMode] = useState<"edit" | "preview">("edit");
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pending = useRef<string | null>(null);
  // Bound per document in the docKey effect, so a flush on switch saves into the old document.
  const saveRef = useRef(onSave);

  const flush = () => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    const value = pending.current;
    if (value === null) return;
    pending.current = null;
    setSaveState("saving");
    saveRef.current(value).then(
      () => setSaveState((s) => (s === "saving" ? "saved" : s)),
      () => setSaveState("error"),
    );
  };

  useEffect(() => {
    saveRef.current = onSave;
    setText(initial);
    setSaveState("idle");
    pending.current = null;
    return flush;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docKey]);

  function change(value: string) {
    setText(value);
    pending.current = value;
    setSaveState("pending");
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(flush, debounceMs);
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-1 border-b border-line px-3 py-1.5 text-xs">
        {(["edit", "preview"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded px-2 py-0.5 ${mode === m ? "bg-panel-2 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"}`}
          >
            {m === "edit" ? "Edit" : "Preview"}
          </button>
        ))}
        <span className={`ml-auto ${saveState === "error" ? "text-red-300" : "text-zinc-500"}`}>{SAVE_LABEL[saveState]}</span>
      </div>
      {mode === "edit" ? (
        <textarea
          className="selectable min-h-0 flex-1 resize-none bg-transparent px-4 py-3 font-mono text-sm leading-relaxed text-zinc-100 outline-none placeholder:text-zinc-600"
          value={text}
          readOnly={readOnly}
          placeholder={placeholder ?? "Your own notes (Markdown). Included when the summary is generated."}
          onChange={(e) => change(e.target.value)}
          onBlur={flush}
        />
      ) : (
        <div className="markdown selectable min-h-0 flex-1 overflow-y-auto px-4 py-3">
          {text.trim() ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
          ) : (
            <p className="text-sm text-zinc-600">Nothing written yet.</p>
          )}
        </div>
      )}
    </div>
  );
}
