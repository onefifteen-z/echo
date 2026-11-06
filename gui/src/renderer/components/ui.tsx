import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "ghost" | "danger";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-accent text-black hover:brightness-110 disabled:opacity-40",
  ghost: "bg-panel-2 text-zinc-200 hover:bg-line disabled:opacity-40",
  danger: "bg-red-500/15 text-red-300 hover:bg-red-500/25 disabled:opacity-40",
};

export function Button({
  variant = "ghost",
  className = "",
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      {...rest}
      className={`no-drag rounded-md px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed ${VARIANTS[variant]} ${className}`}
    />
  );
}

type Tone = "neutral" | "ok" | "warn" | "error" | "accent";

const TONES: Record<Tone, string> = {
  neutral: "bg-panel-2 text-zinc-300 border-line",
  ok: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
  warn: "bg-amber-500/10 text-amber-300 border-amber-500/40",
  error: "bg-red-500/10 text-red-300 border-red-500/40",
  accent: "bg-accent-soft text-accent border-accent/40",
};

export function Chip({ tone = "neutral", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs ${TONES[tone]}`}>
      {children}
    </span>
  );
}

export function Panel({ title, right, children, className = "" }: { title: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`flex min-h-0 flex-col rounded-lg border border-line bg-panel ${className}`}>
      <header className="flex items-center justify-between border-b border-line px-4 py-2">
        <h2 className="text-xs font-semibold tracking-wide text-zinc-400 uppercase">{title}</h2>
        {right}
      </header>
      {children}
    </section>
  );
}

export const LANGUAGE_LABELS = { japanese: "Japanese", english: "English", chinese: "Chinese" } as const;

export const APPLE_LOCALE = { japanese: "ja_JP", english: "en_US", chinese: "zh_CN" } as const;

/** Bare status from `apple.status` (`"engine:status"`), e.g. `installed`, `supported`, `unsupported`. */
export function appleLocaleState(status: Record<string, string>, language: keyof typeof APPLE_LOCALE): string {
  return (status[APPLE_LOCALE[language]] ?? "").split(":")[1] ?? "unsupported";
}
