import { chatPyOrigin } from "./remote";
import { uniqueHosts } from "./settings-hosts";

export {
  canonicalModelId,
  dedupeModelRows,
  hostForRole,
  normalizeHost,
  preferModelId,
  uniqueHosts,
} from "./settings-hosts";

function url(path: string): string {
  return `${chatPyOrigin()}${path}`;
}

const BASE_SETTINGS_KEYS = [
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
  "rerank_llm",
  "rerank_server",
  "tavily_key",
  "sd_server",
  "sd_model",
] as const;

/** Per-model sampling knobs (temperature / top_p / reasoning effort). */
export const TUNING_KEYS = [
  "model_temp",
  "model_topp",
  "model_reasoning_effort",
  "pre_temp",
  "pre_topp",
  "pre_reasoning_effort",
  "vision_temp",
  "vision_topp",
  "vision_reasoning_effort",
  "agent_temp",
  "agent_topp",
  "agent_reasoning_effort",
  "coder_temp",
  "coder_topp",
  "coder_reasoning_effort",
  "casual_temp",
  "casual_topp",
  "casual_reasoning_effort",
  "general_temp",
  "general_topp",
  "general_reasoning_effort",
  "structured_temp",
  "structured_topp",
  "structured_reasoning_effort",
  "nsfw_temp",
  "nsfw_topp",
  "nsfw_reasoning_effort",
  "polisher_temp",
  "polisher_topp",
  "polisher_reasoning_effort",
  "entity_temp",
  "entity_topp",
  "entity_reasoning_effort",
] as const;

export const SETTINGS_KEYS = [...BASE_SETTINGS_KEYS, ...TUNING_KEYS] as const;

export type SettingsKey = (typeof SETTINGS_KEYS)[number];
export type SettingsValues = Record<SettingsKey, string>;

export const ROUTE_GROUPS: {
  title: string;
  rows: {
    id: string;
    label: string;
    llm: SettingsKey;
    server: SettingsKey;
  }[];
}[] = [
  {
    title: "Assistant Related Routes:",
    rows: [
      { id: "vision", label: "Vision", llm: "vision_llm", server: "vision_server" },
      { id: "agent", label: "Agent", llm: "agent_llm", server: "agent_server" },
      { id: "casual", label: "Casual", llm: "casual_llm", server: "casual_server" },
      { id: "general", label: "General", llm: "general_llm", server: "general_server" },
      { id: "coder", label: "Coder", llm: "coder_llm", server: "coder_server" },
      { id: "structured", label: "Structured", llm: "structured_llm", server: "structured_server" },
    ],
  },
  {
    title: "RAG:",
    rows: [
      { id: "rerank", label: "Rerank", llm: "rerank_llm", server: "rerank_server" },
    ],
  },
  {
    title: "Optional Story Related Routes:",
    rows: [
      { id: "nsfw", label: "NSFW", llm: "nsfw_llm", server: "nsfw_server" },
      { id: "polisher", label: "Polisher", llm: "polisher_llm", server: "polisher_server" },
      { id: "entity", label: "Entity", llm: "entity_llm", server: "entity_server" },
    ],
  },
];

export const ROUTE_ROWS = ROUTE_GROUPS.flatMap((g) => g.rows);

/** Sampling-slider keys for one model row. */
export type TuningKeys = {
  temp: SettingsKey;
  topp: SettingsKey;
  effort: SettingsKey;
};

export const GENERATOR_TUNING: TuningKeys = {
  temp: "model_temp",
  topp: "model_topp",
  effort: "model_reasoning_effort",
};

export const PRE_TUNING: TuningKeys = {
  temp: "pre_temp",
  topp: "pre_topp",
  effort: "pre_reasoning_effort",
};

/** Sampling sliders per Route Models row; null = not a chat-sampling model. */
export const ROUTE_TUNING: Record<string, TuningKeys | null> = {
  vision: {
    temp: "vision_temp",
    topp: "vision_topp",
    effort: "vision_reasoning_effort",
  },
  agent: {
    temp: "agent_temp",
    topp: "agent_topp",
    effort: "agent_reasoning_effort",
  },
  casual: {
    temp: "casual_temp",
    topp: "casual_topp",
    effort: "casual_reasoning_effort",
  },
  general: {
    temp: "general_temp",
    topp: "general_topp",
    effort: "general_reasoning_effort",
  },
  coder: {
    temp: "coder_temp",
    topp: "coder_topp",
    effort: "coder_reasoning_effort",
  },
  structured: {
    temp: "structured_temp",
    topp: "structured_topp",
    effort: "structured_reasoning_effort",
  },
  // A cross-encoder scores passages over /v1/rerank — no sampling knobs.
  rerank: null,
  nsfw: {
    temp: "nsfw_temp",
    topp: "nsfw_topp",
    effort: "nsfw_reasoning_effort",
  },
  polisher: {
    temp: "polisher_temp",
    topp: "polisher_topp",
    effort: "polisher_reasoning_effort",
  },
  entity: {
    temp: "entity_temp",
    topp: "entity_topp",
    effort: "entity_reasoning_effort",
  },
};

const ROLE_SERVER_KEYS = SETTINGS_KEYS.filter(
  (key): key is SettingsKey => key.endsWith("_server") && key !== "sd_server",
);

/** Unique model-server URLs from a settings form (skips Stable Diffusion). */
export function uniqueRoleHosts(values: SettingsValues): string[] {
  return uniqueHosts(ROLE_SERVER_KEYS.map((key) => values[key]));
}

export type SettingsPayload = {
  ok: boolean;
  error?: string;
  path?: string;
  values: SettingsValues;
  effective: SettingsValues;
  message?: string;
  busy?: boolean;
};

export type ModelInfo = { id: string; loaded?: boolean | null };

export type PingResult = {
  ok: boolean;
  error?: string | null;
  models: string[];
  details?: ModelInfo[];
  loaded?: string[];
  knows_loaded?: boolean;
  source?: string;
  url?: string;
  current?: string;
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
  const details = Array.isArray(json.details)
    ? json.details
        .filter((row) => row && typeof row.id === "string")
        .map((row) => ({ id: row.id, loaded: row.loaded ?? null }))
    : [];
  return {
    ok: Boolean(json.ok),
    error: json.error,
    models: Array.isArray(json.models) ? json.models.map(String) : [],
    details,
    loaded: Array.isArray(json.loaded) ? json.loaded.map(String) : [],
    knows_loaded: Boolean(json.knows_loaded),
    source: json.source,
    url: json.url,
  };
}

export async function pingSd(host: string): Promise<PingResult> {
  const res = await fetch(url("/api/settings/ping"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ host, kind: "sd" }),
  });
  const json = (await res.json()) as PingResult;
  return {
    ok: Boolean(json.ok),
    error: json.error,
    models: Array.isArray(json.models) ? json.models.map(String) : [],
    current: typeof json.current === "string" ? json.current : "",
    url: json.url,
    source: json.source,
  };
}
