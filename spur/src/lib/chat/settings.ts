import { chatPyOrigin } from "./remote";

function url(path: string): string {
  return `${chatPyOrigin()}${path}`;
}

export const SETTINGS_KEYS = [
  "llm_server",
  "api_key",
  "model",
  "pre_llm",
  "embedding_llm",
  "pre_server",
  "embedding_server",
  "vision_llm",
  "vision_server",
  "agent_llm",
  "agent_server",
  "coder_llm",
  "coder_server",
  "casual_llm",
  "casual_server",
  "general_llm",
  "general_server",
  "structured_llm",
  "structured_server",
  "nsfw_llm",
  "nsfw_server",
  "polisher_llm",
  "polisher_server",
  "entity_llm",
  "entity_server",
  "tavily_key",
] as const;

export type SettingsKey = (typeof SETTINGS_KEYS)[number];
export type SettingsValues = Record<SettingsKey, string>;

export const ROUTE_ROWS: {
  id: string;
  label: string;
  llm: SettingsKey;
  server: SettingsKey;
}[] = [
  { id: "vision", label: "Vision", llm: "vision_llm", server: "vision_server" },
  { id: "agent", label: "Agent", llm: "agent_llm", server: "agent_server" },
  { id: "coder", label: "Coder", llm: "coder_llm", server: "coder_server" },
  { id: "casual", label: "Casual", llm: "casual_llm", server: "casual_server" },
  { id: "general", label: "General", llm: "general_llm", server: "general_server" },
  { id: "structured", label: "Structured", llm: "structured_llm", server: "structured_server" },
  { id: "nsfw", label: "NSFW", llm: "nsfw_llm", server: "nsfw_server" },
  { id: "polisher", label: "Polisher", llm: "polisher_llm", server: "polisher_server" },
  { id: "entity", label: "Entity", llm: "entity_llm", server: "entity_server" },
];

export type SettingsPayload = {
  ok: boolean;
  error?: string;
  path?: string;
  values: SettingsValues;
  effective: SettingsValues;
  message?: string;
  busy?: boolean;
};

export type PingResult = {
  ok: boolean;
  error?: string | null;
  models: string[];
  url?: string;
};

function emptyValues(): SettingsValues {
  return Object.fromEntries(SETTINGS_KEYS.map((k) => [k, ""])) as SettingsValues;
}

function asValues(raw: unknown): SettingsValues {
  const base = emptyValues();
  if (!raw || typeof raw !== "object") return base;
  const rec = raw as Record<string, unknown>;
  for (const key of SETTINGS_KEYS) {
    const v = rec[key];
    if (typeof v === "string") base[key] = v;
    else if (v != null) base[key] = String(v);
  }
  return base;
}

export async function fetchSettings(): Promise<SettingsPayload> {
  const res = await fetch(url("/api/settings"));
  const json = (await res.json()) as Partial<SettingsPayload>;
  return {
    ok: Boolean(json.ok),
    error: json.error,
    path: json.path,
    values: asValues(json.values),
    effective: asValues(json.effective),
    busy: Boolean(json.busy),
  };
}

export async function saveSettings(values: SettingsValues): Promise<SettingsPayload> {
  const res = await fetch(url("/api/settings"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }),
  });
  const json = (await res.json()) as Partial<SettingsPayload>;
  return {
    ok: Boolean(json.ok),
    error: json.error,
    path: json.path,
    values: asValues(json.values ?? values),
    effective: asValues(json.effective),
    message: json.message,
  };
}

export async function pingSettings(host: string, apiKey: string): Promise<PingResult> {
  const res = await fetch(url("/api/settings/ping"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ host, api_key: apiKey }),
  });
  const json = (await res.json()) as PingResult;
  return {
    ok: Boolean(json.ok),
    error: json.error,
    models: Array.isArray(json.models) ? json.models.map(String) : [],
    url: json.url,
  };
}
