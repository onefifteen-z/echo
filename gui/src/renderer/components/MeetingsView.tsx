import { useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { MeetingDetail, MeetingSummary } from "@shared/types";
import { NotebookEditor } from "./NotebookEditor";
import { Button, Chip } from "./ui";

type Tab = "summary" | "notebook" | "transcript" | "insights" | "details";

const TABS: { id: Tab; label: string }[] = [
  { id: "summary", label: "Summary" },
  { id: "notebook", label: "Notebook" },
  { id: "transcript", label: "Transcript" },
  { id: "insights", label: "Insights" },
  { id: "details", label: "Details" },
];

interface Props {
  /** Bumped by the parent when meetings change elsewhere (meeting ended, summary written). */
  refreshKey: number;
  selectId?: string | null;
  onOpenLive: () => void;
  toast: (text: string, tone?: "info" | "error", path?: string) => void;
}

function errorText(e: unknown): string {
  return (e instanceof Error ? e.message : String(e)).replace(/^Error invoking remote method '[^']+': (Error: )?/, "");
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function formatDuration(s: number | null | undefined): string {
  if (s === null || s === undefined) return "";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h} h ${m} min` : m > 0 ? `${m} min` : `${s} s`;
}

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

export function MeetingsView({ refreshKey, selectId, onOpenLive, toast }: Props) {
  const [meetings, setMeetings] = useState<MeetingSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string | null>(selectId ?? null);
  const [detail, setDetail] = useState<MeetingDetail | null>(null);
  const [tab, setTab] = useState<Tab>("summary");
  const [regenerating, setRegenerating] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const [contextDraft, setContextDraft] = useState("");

  const loadList = useCallback(async () => {
    try {
      const r = await window.echo.invoke("meetings.list", {});
      setMeetings(r.meetings);
      setSelected((cur) => cur ?? r.meetings[0]?.id ?? null);
    } catch (e) {
      toast(errorText(e), "error");
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const loadDetail = useCallback(
    async (id: string) => {
      try {
        const d = await window.echo.invoke("meetings.get", { id });
        setDetail(d);
        setTitleDraft(d.title);
        setContextDraft(d.context);
      } catch (e) {
        setDetail(null);
        toast(errorText(e), "error");
      }
    },
    [toast],
  );

  useEffect(() => void loadList(), [loadList, refreshKey]);
  useEffect(() => {
    if (selectId) setSelected(selectId);
  }, [selectId]);
  useEffect(() => {
    if (selected) void loadDetail(selected);
    else setDetail(null);
  }, [selected, loadDetail, refreshKey]);

  useEffect(
    () =>
      window.echo.onEvent((ev) => {
        if (ev.type === "summary.written" && ev.id && ev.id === selected) void loadDetail(ev.id);
      }),
    [selected, loadDetail],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return meetings;
    return meetings.filter((m) => `${m.title}\n${m.context}\n${m.summary_preview}`.toLowerCase().includes(q));
  }, [meetings, query]);

  async function update(patch: { title?: string; context?: string; notebook?: string }) {
    if (!detail) return;
    const id = detail.id;
    const d = await window.echo.invoke("meetings.update", { id, ...patch });
    if (patch.notebook === undefined) setDetail(d);
    setMeetings((ms) =>
      ms.map((m) =>
        m.id === id
          ? { ...m, title: d.title, context: d.context, has_notebook: patch.notebook !== undefined ? !!patch.notebook.trim() : m.has_notebook }
          : m,
      ),
    );
  }

  async function saveTitle() {
    if (!detail || titleDraft.trim() === detail.title) return;
    try {
      await update({ title: titleDraft });
    } catch (e) {
      toast(errorText(e), "error");
    }
  }

  async function regenerate() {
    if (!detail) return;
    setRegenerating(true);
    setTab("summary");
    setDetail({ ...detail, summary_md: "_Generating meeting summary…_" });
    try {
      const r = await window.echo.invoke("meetings.regenerate_summary", { id: detail.id });
      setDetail((d) => (d && d.id === r.id ? { ...d, summary_md: r.summary_md } : d));
      void loadList();
      toast("Summary regenerated.");
    } catch (e) {
      toast(errorText(e), "error");
      void loadDetail(detail.id);
    } finally {
      setRegenerating(false);
    }
  }

  async function remove() {
    if (!detail) return;
    if (!window.confirm(`Delete "${detail.title}"? The folder is moved to the Trash.`)) return;
    try {
      await window.echo.trashMeeting(detail.id);
      toast("Meeting moved to Trash.");
      setSelected(null);
      setDetail(null);
      await loadList();
    } catch (e) {
      toast(errorText(e), "error");
    }
  }

  const live = !!detail?.live;

  return (
    <div className="grid min-h-0 flex-1 grid-cols-[320px_minmax(0,1fr)] gap-3 p-3">
      <aside className="flex min-h-0 flex-col rounded-lg border border-line bg-panel">
        <div className="border-b border-line p-2">
          <input
            className="w-full rounded-md border border-line bg-panel-2 px-3 py-1.5 text-sm text-zinc-100 outline-none focus:border-accent"
            placeholder="Search meetings…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <ul className="min-h-0 flex-1 overflow-y-auto">
          {loading && <li className="p-4 text-sm text-zinc-500">Loading…</li>}
          {!loading && filtered.length === 0 && (
            <li className="p-4 text-sm text-zinc-500">{meetings.length ? "No matches." : "No saved meetings yet."}</li>
          )}
          {filtered.map((m) => (
            <li key={m.id}>
              <button
                onClick={() => setSelected(m.id)}
                className={`w-full border-b border-line/60 px-3 py-2.5 text-left transition ${
                  m.id === selected ? "bg-accent-soft" : "hover:bg-panel-2"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="truncate text-sm font-medium text-zinc-100">{m.title}</span>
                  {m.live && <Chip tone="ok">Live</Chip>}
                </div>
                <div className="mt-0.5 text-xs text-zinc-500">
                  {formatDate(m.started_at)}
                  {m.duration_s ? ` · ${formatDuration(m.duration_s)}` : ""}
                  {m.line_count !== null ? ` · ${m.line_count} lines` : ""}
                  {m.has_notebook ? " · notes" : ""}
                </div>
                {m.summary_preview && <div className="mt-1 line-clamp-2 text-xs text-zinc-400">{m.summary_preview}</div>}
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <section className="flex min-h-0 flex-col rounded-lg border border-line bg-panel">
        {!detail ? (
          <div className="pt-24 text-center text-sm text-zinc-500">
            {meetings.length ? "Select a meeting." : "Meetings you end or save appear here."}
          </div>
        ) : (
          <>
            <header className="flex items-start gap-3 border-b border-line px-4 py-3">
              <div className="min-w-0 flex-1">
                <input
                  className="-ml-1 w-full rounded border border-transparent bg-transparent px-1 text-lg font-semibold text-zinc-100 outline-none hover:border-line focus:border-accent"
                  value={titleDraft}
                  onChange={(e) => setTitleDraft(e.target.value)}
                  onBlur={() => void saveTitle()}
                  onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
                />
                <div className="mt-0.5 text-xs text-zinc-500">
                  {formatDate(detail.started_at)}
                  {detail.duration_s ? ` · ${formatDuration(detail.duration_s)}` : ""} · {detail.transcript.length} lines
                  {detail.legacy ? " · saved by an older version" : ""}
                </div>
              </div>
              {live ? (
                <Button variant="primary" onClick={onOpenLive}>
                  Open live view
                </Button>
              ) : (
                <>
                  <Button onClick={() => void window.echo.openPath(detail.path)}>Open folder</Button>
                  <Button variant="danger" disabled={regenerating} onClick={() => void remove()}>
                    Delete
                  </Button>
                </>
              )}
            </header>
            <nav className="flex gap-1 border-b border-line px-3">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`-mb-px border-b-2 px-3 py-2 text-sm ${
                    tab === t.id ? "border-accent text-zinc-100" : "border-transparent text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </nav>

            {tab === "summary" && (
              <div className="min-h-0 flex-1 overflow-y-auto p-4">
                <div className="mb-3 flex items-center gap-3">
                  <Button variant="primary" disabled={regenerating || live} onClick={() => void regenerate()}>
                    {regenerating ? "Regenerating…" : "Regenerate summary"}
                  </Button>
                  <span className="text-xs text-zinc-500">
                    {live
                      ? "Available after the meeting ends."
                      : "Uses the transcript, live insights, context and your notebook."}
                  </span>
                </div>
                {detail.summary_md.trim() ? (
                  <Md>{detail.summary_md}</Md>
                ) : (
                  <p className="text-sm text-zinc-500">No summary yet.</p>
                )}
              </div>
            )}
            {tab === "notebook" && (
              <NotebookEditor
                docKey={detail.id}
                initial={detail.notebook_md}
                onSave={(text) => update({ notebook: text })}
              />
            )}
            {tab === "transcript" && (
              <div className="selectable min-h-0 flex-1 space-y-1.5 overflow-y-auto px-4 py-3">
                {detail.transcript.length === 0 && <p className="text-sm text-zinc-500">No transcript captured.</p>}
                {detail.transcript.map((line, i) => {
                  const m = /^\[([^\]]+)\]\s*(.*)$/.exec(line);
                  return (
                    <div key={i} className="flex gap-3">
                      <span className="w-32 shrink-0 pt-0.5 font-mono text-xs text-zinc-600">{m ? m[1] : ""}</span>
                      <span className="text-[15px] leading-relaxed text-zinc-100">{m ? m[2] : line}</span>
                    </div>
                  );
                })}
              </div>
            )}
            {tab === "insights" && (
              <div className="min-h-0 flex-1 overflow-y-auto p-4">
                {detail.insights_md.trim() ? <Md>{detail.insights_md}</Md> : <p className="text-sm text-zinc-500">No insights.</p>}
              </div>
            )}
            {tab === "details" && (
              <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
                <div>
                  <label className="mb-1 block text-xs font-medium text-zinc-400">Context (agenda, participants, background)</label>
                  <textarea
                    className="h-32 w-full resize-none rounded-md border border-line bg-panel-2 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-accent"
                    value={contextDraft}
                    onChange={(e) => setContextDraft(e.target.value)}
                  />
                  <div className="mt-2 flex justify-end">
                    <Button
                      disabled={contextDraft.trim() === detail.context}
                      onClick={() =>
                        void update({ context: contextDraft }).then(
                          () => toast("Context saved."),
                          (e) => toast(errorText(e), "error"),
                        )
                      }
                    >
                      Save context
                    </Button>
                  </div>
                </div>
                <dl className="grid grid-cols-[140px_1fr] gap-y-1.5 text-sm">
                  {[
                    ["Started", formatDate(detail.started_at)],
                    ["Ended", detail.ended_at ? formatDate(detail.ended_at) : "—"],
                    ["Duration", formatDuration(detail.duration_s) || "—"],
                    ["Mode", detail.mode || "—"],
                    ["Language", detail.language || "—"],
                    ["Speech engine", detail.asr_source || "—"],
                    ["Folder", detail.path],
                  ].map(([k, v]) => (
                    <div key={k} className="contents">
                      <dt className="text-zinc-500">{k}</dt>
                      <dd className="selectable truncate text-zinc-200" title={v}>
                        {v}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
