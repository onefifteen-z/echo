import { useEffect, useRef, useState } from "react";
import { EditorContent, useEditor, type Editor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Image from "@tiptap/extension-image";
import Placeholder from "@tiptap/extension-placeholder";
import { TableKit } from "@tiptap/extension-table";
import { TaskItem, TaskList } from "@tiptap/extension-list";
import { Markdown } from "@tiptap/markdown";
import type { EditorView } from "@tiptap/pm/view";

type SaveState = "idle" | "pending" | "saving" | "saved" | "error";

const SAVE_LABEL: Record<SaveState, string> = {
  idle: "",
  pending: "Editing…",
  saving: "Saving…",
  saved: "Saved",
  error: "Save failed",
};

const NO_MEETING = "Images can be added once the meeting is saved.";

interface Props {
  /** Markdown loaded from the engine / disk. Changing `docKey` resets the editor to this value. */
  initial: string;
  docKey: string;
  /** Meeting folder name; image files are stored in its `notes-assets/`. Null disables images. */
  meetingId: string | null;
  onSave: (text: string) => Promise<unknown>;
  onError?: (message: string) => void;
  placeholder?: string;
  readOnly?: boolean;
  debounceMs?: number;
}

function assetUrl(meetingId: string | null, src: string | null): string | null {
  if (!src || !meetingId || !src.startsWith("notes-assets/")) return src;
  return `echo-asset://meeting/${encodeURIComponent(meetingId)}/${src.split("/").map(encodeURIComponent).join("/")}`;
}

function relativeSrc(src: string | null): string | null {
  const m = src?.match(/^echo-asset:\/\/meeting\/[^/]+\/(notes-assets\/[^/]+)$/);
  return m ? decodeURIComponent(m[1]) : src;
}

/** Markdown keeps the relative `notes-assets/...` path; only the rendered `<img>` points at `echo-asset://`. */
function noteImage(meetingId: string | null) {
  return Image.extend({
    addAttributes() {
      return {
        ...this.parent?.(),
        src: {
          default: null,
          parseHTML: (el: HTMLElement) => relativeSrc(el.getAttribute("src")),
          renderHTML: (attrs: { src?: string | null }) => ({ src: assetUrl(meetingId, attrs.src ?? null) }),
        },
      };
    },
  });
}

function imageFiles(list: FileList | null | undefined): File[] {
  return [...(list ?? [])].filter((f) => f.type.startsWith("image/"));
}

function cleanError(e: unknown): string {
  return (e instanceof Error ? e.message : String(e)).replace(/^Error invoking remote method '[^']+': (Error: )?/, "");
}

export function NotebookEditor({ initial, docKey, meetingId, onSave, onError, placeholder, readOnly, debounceMs = 800 }: Props) {
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [notice, setNotice] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pending = useRef<string | null>(null);
  // Bound per document in the docKey effect, so a flush on switch saves into the old document.
  const saveRef = useRef(onSave);
  const fileInput = useRef<HTMLInputElement | null>(null);

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

  const change = (value: string) => {
    pending.current = value;
    setSaveState("pending");
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(flush, debounceMs);
  };

  const report = (message: string) => {
    setNotice(message);
    onError?.(message);
  };

  /** Save files to the meeting folder and insert them at `pos` (or the cursor). */
  const insertImages = async (editor: Editor, files: File[], pos?: number) => {
    if (!meetingId) return report(NO_MEETING);
    setNotice(null);
    for (const file of files) {
      try {
        const bytes = new Uint8Array(await file.arrayBuffer());
        const { src } = await window.echo.saveNoteImage(meetingId, bytes, file.type);
        const node = { type: "image", attrs: { src, alt: file.name.replace(/\.[^.]+$/, "") || null } };
        if (pos === undefined) editor.chain().focus().insertContent(node).run();
        else editor.chain().focus().insertContentAt(pos, node).run();
      } catch (e) {
        report(cleanError(e));
      }
    }
  };

  const editorRef = useRef<Editor | null>(null);
  const editor = useEditor(
    {
      extensions: [
        StarterKit.configure({ link: { openOnClick: false } }),
        noteImage(meetingId),
        TaskList,
        TaskItem.configure({ nested: true }),
        TableKit,
        Placeholder.configure({ placeholder: placeholder ?? "Your own notes. Paste or drop images here. Included when the summary is generated." }),
        Markdown,
      ],
      content: initial,
      contentType: "markdown",
      editable: !readOnly,
      shouldRerenderOnTransaction: true,
      editorProps: {
        attributes: { class: "markdown selectable notebook-doc" },
        handlePaste: (_view: EditorView, event: ClipboardEvent) => {
          const files = imageFiles(event.clipboardData?.files);
          if (!files.length || !editorRef.current) return false;
          event.preventDefault();
          void insertImages(editorRef.current, files);
          return true;
        },
        handleDrop: (view: EditorView, event: DragEvent, _slice: unknown, moved: boolean) => {
          const files = imageFiles(event.dataTransfer?.files);
          if (moved || !files.length || !editorRef.current) return false;
          event.preventDefault();
          const pos = view.posAtCoords({ left: event.clientX, top: event.clientY })?.pos;
          void insertImages(editorRef.current, files, pos);
          return true;
        },
      },
      onUpdate: ({ editor: ed }) => change(ed.getMarkdown()),
      onBlur: () => flush(),
    },
    [meetingId],
  );
  editorRef.current = editor;

  useEffect(() => {
    saveRef.current = onSave;
    if (editor && !editor.isDestroyed && editor.getMarkdown() !== initial) {
      editor.commands.setContent(initial, { contentType: "markdown", emitUpdate: false });
    }
    setSaveState("idle");
    setNotice(null);
    pending.current = null;
    return flush;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docKey, editor]);

  useEffect(() => {
    editor?.setEditable(!readOnly);
  }, [editor, readOnly]);

  const tool = (label: string, title: string, active: boolean, run: () => void) => (
    <button
      key={title}
      title={title}
      disabled={readOnly || !editor}
      onMouseDown={(e) => e.preventDefault()}
      onClick={run}
      className={`rounded px-1.5 py-0.5 ${active ? "bg-panel-2 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"} disabled:opacity-40`}
    >
      {label}
    </button>
  );
  const chain = () => editor!.chain().focus();

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-0.5 border-b border-line px-3 py-1.5 text-xs">
        {tool("B", "Bold", !!editor?.isActive("bold"), () => chain().toggleBold().run())}
        {tool("I", "Italic", !!editor?.isActive("italic"), () => chain().toggleItalic().run())}
        {tool("H", "Heading", !!editor?.isActive("heading"), () => chain().toggleHeading({ level: 2 }).run())}
        {tool("•", "Bullet list", !!editor?.isActive("bulletList"), () => chain().toggleBulletList().run())}
        {tool("☐", "Checklist", !!editor?.isActive("taskList"), () => chain().toggleTaskList().run())}
        {tool("Image", "Insert image", false, () => (meetingId ? fileInput.current?.click() : report(NO_MEETING)))}
        <input
          ref={fileInput}
          type="file"
          accept="image/png,image/jpeg,image/gif,image/webp"
          multiple
          className="hidden"
          onChange={(e) => {
            const files = imageFiles(e.target.files);
            e.target.value = "";
            if (editor && files.length) void insertImages(editor, files);
          }}
        />
        <span className={`ml-auto truncate pl-2 ${notice || saveState === "error" ? "text-red-300" : "text-zinc-500"}`}>
          {notice ?? SAVE_LABEL[saveState]}
        </span>
      </div>
      <EditorContent
        editor={editor}
        className="notebook min-h-0 flex-1 overflow-y-auto px-4 py-3"
        onDragOver={(e) => {
          if (!readOnly && e.dataTransfer.types.includes("Files")) e.preventDefault();
        }}
        onDrop={(e) => {
          // ProseMirror skips handleDrop when the point maps to no position (padding, empty space below the text).
          const files = imageFiles(e.dataTransfer.files);
          if (e.defaultPrevented || readOnly || !editor || !files.length) return;
          e.preventDefault();
          void insertImages(editor, files);
        }}
      />
    </div>
  );
}
