export type LlmTarget = {
  baseUrl: string;
  apiKey: string;
  model: string;
  webSearch: boolean;
};

function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

export function resolveLlm(
  env: NodeJS.ProcessEnv = process.env,
): LlmTarget | null {
  const base = (env.LLM_BASE_URL || env.OPENAI_BASE_URL || "").trim();
  if (base) {
    return {
      baseUrl: trimSlash(base),
      apiKey: (env.LLM_API_KEY || env.OPENAI_API_KEY || "lm-studio").trim() || "lm-studio",
      model: (env.LLM_MODEL || env.OPENAI_MODEL || "local-model").trim() || "local-model",
      webSearch: /api\.x\.ai/i.test(base),
    };
  }
  const xai = (env.XAI_API_KEY || "").trim();
  if (!xai) return null;
  return {
    baseUrl: "https://api.x.ai/v1",
    apiKey: xai,
    model: (env.LLM_MODEL || "").trim() || "grok-4.5",
    webSearch: true,
  };
}
