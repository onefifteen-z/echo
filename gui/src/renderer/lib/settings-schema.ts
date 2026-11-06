/** Curated settings shown as forms; anything else lives in the Advanced key/value editor. */

export type FieldKind = "text" | "secret" | "number" | "select" | "bool" | "file";

export interface FieldDef {
  key: string;
  label: string;
  kind: FieldKind;
  help?: string;
  /** Shown as placeholder: the engine's built-in default. */
  placeholder?: string;
  options?: { value: string; label: string }[];
  min?: number;
}

export interface GroupDef {
  id: string;
  title: string;
  description?: string;
  fields: FieldDef[];
}

const bool = (key: string, label: string, help: string, placeholder: string): FieldDef => ({
  key,
  label,
  kind: "bool",
  help,
  placeholder,
});

export const SETTINGS_GROUPS: GroupDef[] = [
  {
    id: "workspace",
    title: "Workspace",
    fields: [
      {
        key: "ECHO_DICTIONARY_PATH",
        label: "Glossary file",
        kind: "file",
        help: "YAML list of names and terms used to correct ASR and guide the LLM.",
        placeholder: "dictionary.yml in the project",
      },
    ],
  },
  {
    id: "llm",
    title: "Language model",
    description:
      "Used for insights, chat, and summaries. Echo picks Anthropic first, then an OpenAI-compatible endpoint, then OpenAI.",
    fields: [
      { key: "ANTHROPIC_API_KEY", label: "Anthropic API key", kind: "secret" },
      { key: "ECHO_ANTHROPIC_MODEL", label: "Anthropic model", kind: "text", placeholder: "Echo default" },
      {
        key: "BASE_URL",
        label: "OpenAI-compatible base URL",
        kind: "text",
        help: "LiteLLM, vLLM, an AI gateway, … Requires API key and model.",
        placeholder: "https://…/v1",
      },
      { key: "API_KEY", label: "OpenAI-compatible API key", kind: "secret" },
      { key: "MODEL", label: "OpenAI-compatible model", kind: "text", placeholder: "e.g. gpt-4.1-mini" },
      { key: "OPENAI_API_KEY", label: "OpenAI API key", kind: "secret" },
      { key: "ECHO_OPENAI_MODEL", label: "OpenAI model", kind: "text", placeholder: "Echo default" },
    ],
  },
  {
    id: "speech",
    title: "Speech recognition",
    fields: [
      {
        key: "ECHO_ASR_BACKEND",
        label: "Default speech engine",
        kind: "select",
        help: "Preselected in the New meeting dialog.",
        options: [
          { value: "auto", label: "Auto (Apple → API → local)" },
          { value: "apple", label: "Apple on-device" },
          { value: "api", label: "Whisper API" },
          { value: "local", label: "Local Kotoba-Whisper" },
        ],
      },
      {
        key: "ECHO_WHISPER_API_BASE_URL",
        label: "Whisper API base URL",
        kind: "text",
        help: "Separate endpoint for /audio/transcriptions. Uses the OpenAI-compatible API key.",
        placeholder: "same as base URL",
      },
      { key: "ECHO_WHISPER_API_KEY", label: "Whisper API key", kind: "secret", help: "Only if it differs from the API key above." },
      { key: "ECHO_WHISPER_API_MODEL", label: "Whisper API model", kind: "text", placeholder: "whisper-1" },
      { key: "ECHO_WHISPER_API_TIMEOUT", label: "Whisper API timeout (s)", kind: "number", placeholder: "20", min: 1 },
      {
        key: "ECHO_WHISPER_API_CHUNK_SECONDS",
        label: "Whisper API window (s)",
        kind: "number",
        placeholder: "8",
        min: 2,
        help: "Shorter windows show text sooner but cost more requests.",
      },
      {
        key: "ECHO_WHISPER_DEVICE",
        label: "Local model device",
        kind: "select",
        options: [
          { value: "cpu", label: "CPU" },
          { value: "mps", label: "Apple GPU (mps)" },
          { value: "cuda", label: "CUDA" },
        ],
        placeholder: "Auto",
      },
    ],
  },
  {
    id: "search",
    title: "Web search",
    description: "Perplexity is used when a base URL and key are set; otherwise ddgs (DuckDuckGo, Bing, …).",
    fields: [
      { key: "ECHO_PERPLEXITY_BASE_URL", label: "Perplexity base URL", kind: "text", placeholder: "https://…/v1" },
      {
        key: "ECHO_PERPLEXITY_API_KEY",
        label: "Perplexity API key",
        kind: "secret",
        help: "Falls back to AI_GATEWAY_API_KEY, then the OpenAI-compatible API key.",
      },
      { key: "ECHO_PERPLEXITY_MODEL", label: "Perplexity model", kind: "text", placeholder: "sonar" },
      { key: "ECHO_PERPLEXITY_TIMEOUT", label: "Perplexity timeout (s)", kind: "number", placeholder: "30", min: 1 },
      {
        key: "ECHO_DDGS_BACKEND",
        label: "ddgs backend",
        kind: "text",
        placeholder: "auto",
        help: "auto, bing, duckduckgo, yahoo, …",
      },
    ],
  },
  {
    id: "insights",
    title: "Insights & transcript",
    fields: [
      { key: "ECHO_INSIGHT_LIVE_MIN_INTERVAL", label: "Min seconds between insights", kind: "number", placeholder: "45", min: 0 },
      { key: "ECHO_INSIGHT_PAUSE_SECONDS", label: "Pause before insight (s)", kind: "number", placeholder: "3", min: 0 },
      { key: "ECHO_INSIGHT_LLM_TIMEOUT", label: "Insight LLM timeout (s)", kind: "number", placeholder: "120", min: 1 },
      { key: "ECHO_INSIGHT_DDG_TIMEOUT", label: "Web search timeout (s)", kind: "number", placeholder: "45", min: 1 },
      { key: "ECHO_LIVE_SEGMENT_PAUSE_SECONDS", label: "Silence that ends a line (s)", kind: "number", placeholder: "2.0", min: 0 },
      bool("ECHO_ENABLE_TRANSCRIPT_REWRITE", "LLM transcript cleanup", "Rewrite finalized lines with the LLM.", "On"),
    ],
  },
  {
    id: "network",
    title: "Network & debugging",
    fields: [
      bool("ECHO_OFFLINE", "Offline mode", "Never call the LLM or web search.", "Off"),
      bool("ECHO_ASSUME_ONLINE", "Skip reachability check", "Assume the LLM endpoint is reachable.", "Off"),
      bool("ECHO_HF_LOCAL_ONLY", "Local models only", "Do not download Hugging Face models.", "Off"),
      bool("ECHO_DEBUG", "Debug logging", "Write debug events to the engine log.", "Off"),
    ],
  },
];

export const CURATED_KEYS: string[] = SETTINGS_GROUPS.flatMap((g) => g.fields.map((f) => f.key));

const KEY_RE = /^[A-Z][A-Z0-9_]*$/;
const RESERVED = new Set(["PATH", "HOME", "USER", "SHELL", "ECHO_SERVER_TOKEN", "ECHO_APP_SETTINGS_KEYS", "PYTHONUNBUFFERED"]);

/** Mirrors main-process validation so the form can flag problems before saving. */
export function keyError(key: string): string | null {
  if (!KEY_RE.test(key)) return "Use uppercase letters, digits and _ (must start with a letter).";
  if (RESERVED.has(key) || /^(PYTHON|DYLD_|ELECTRON_)/.test(key)) return "Reserved; cannot be set here.";
  return null;
}

export function isSecretKey(key: string): boolean {
  return /(KEY|TOKEN|SECRET|PASSWORD)/.test(key);
}

export function valueError(field: FieldDef, value: string): string | null {
  const v = value.trim();
  if (!v) return null;
  if (field.kind === "number") {
    const n = Number(v);
    if (!Number.isFinite(n)) return "Enter a number.";
    if (field.min !== undefined && n < field.min) return `Must be at least ${field.min}.`;
  }
  if (field.key.endsWith("BASE_URL") && !/^https?:\/\/\S+$/.test(v)) return "Enter a full URL (https://…).";
  if (field.key === "ECHO_PERPLEXITY_BASE_URL" && !v.startsWith("https://")) return "Perplexity requires https://.";
  return null;
}
