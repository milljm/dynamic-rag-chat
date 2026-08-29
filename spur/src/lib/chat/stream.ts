export type ChatRequest = {
  mode: "assistant" | "story";
  branchId: string;
  messages: { role: "user" | "assistant"; content: string }[];
  rag: { source: string; text: string }[];
  images: string[];
  includes?: string[];
  useAgent?: boolean;
  noContext?: boolean;
  rare?: string[];
  oocDiagnostics?: string;
};

export type ChatEvent =
  | { type: "status"; message: string; model?: string; route?: string; context?: number; recalled?: string[] }
  | { type: "token"; content: string }
  | { type: "reasoning"; content: string }
  | { type: "usage"; promptTokens: number; completionTokens: number; model: string; tokenSavings?: number }
  | { type: "error"; error: string }
  | { type: "documents"; documents: { name: string; chars: number }[] }
  | { type: "done" };

export async function streamSse(
  path: string,
  body: unknown,
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    let message = `Chat failed (${res.status})`;
    try {
      const parsed = (await res.json()) as { error?: string };
      if (parsed.error) message = parsed.error;
    } catch {
      /* keep status text */
    }
    throw new Error(message);
  }

  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith(":")) continue;
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;
      try {
        const event = JSON.parse(payload) as ChatEvent;
        try {
          onEvent(event);
        } catch {
          /* a render throw must not abort the LLM stream */
        }
      } catch {
        /* skip malformed chunk */
      }
    }
  }
}

export async function streamChat(
  input: ChatRequest,
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  return streamSse("/api/chat", input, onEvent, signal);
}
