import { createFileRoute } from "@tanstack/react-router";
import { z } from "zod";
import { resolveLlm, type LlmTarget } from "@/lib/chat/llm";
import { systemFor } from "@/lib/chat/prompts";

const Body = z.object({
  mode: z.enum(["assistant", "story"]),
  branchId: z.string().min(1).max(80),
  messages: z
    .array(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string().max(24_000),
      }),
    )
    .max(40),
  rag: z
    .array(
      z.object({
        source: z.string().max(200),
        text: z.string().max(8_000),
      }),
    )
    .max(12),
  images: z.array(z.string().max(2_000_000)).max(3),
  includes: z.array(z.string().url().max(2_000)).max(4).optional(),
  useAgent: z.boolean().optional(),
  noContext: z.boolean().optional(),
  rare: z.array(z.string().max(40)).max(6).optional(),
  oocDiagnostics: z.string().max(8_000).optional(),
});

type XaiDelta = {
  content?: string | null;
  reasoning_content?: string | null;
};

type XaiChunk = {
  model?: string;
  choices?: { delta?: XaiDelta; finish_reason?: string | null }[];
  usage?: { prompt_tokens?: number; completion_tokens?: number };
};

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      GET: async () => {
        const llm = resolveLlm();
        return Response.json({
          available: Boolean(llm),
          provider: llm?.baseUrl ?? null,
          model: llm?.model ?? null,
        });
      },
      POST: async ({ request }) => {
        const llm = resolveLlm();
        if (!llm) {
          return Response.json(
            {
              error:
                "No model server configured. Set LLM_BASE_URL to your LM Studio (or other OpenAI-compatible) endpoint, e.g. http://127.0.0.1:1234/v1",
            },
            { status: 503 },
          );
        }

        let json: unknown;
        try {
          json = await request.json();
        } catch {
          return Response.json({ error: "Invalid JSON" }, { status: 400 });
        }

        const parsed = Body.safeParse(json);
        if (!parsed.success) {
          return Response.json(
            { error: "Invalid chat request." },
            { status: 400 },
          );
        }

        const {
          mode,
          branchId,
          messages,
          rag,
          images,
          includes,
          useAgent,
          noContext,
          rare,
          oocDiagnostics,
        } = parsed.data;
        if (!messages.length) {
          return Response.json({ error: "No messages." }, { status: 400 });
        }

        const lastUser =
          [...messages].reverse().find((m) => m.role === "user")?.content ?? "";

        const encoder = new TextEncoder();
        const sendLine = (controller: ReadableStreamDefaultController, obj: unknown) => {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
        };

        const stream = new ReadableStream({
          async start(controller) {
            try {
              let agentBlock = "";
              if (useAgent && mode === "assistant") {
                if (llm.webSearch) {
                  sendLine(controller, {
                    type: "status",
                    message: "Agent tool web search…",
                  });
                  const searched = await runAgentSearch(llm, lastUser);
                  if (searched.error) {
                    agentBlock =
                      "=== AGENT_TOOL_RESULT ===\nERROR: Tool execution failed.\nINSTRUCTION: Inform the user that the web search failed. Do NOT fabricate or guess.\n";
                  } else {
                    agentBlock = `=== AGENT_TOOL_RESULT ===\n${searched.text}\n`;
                  }
                } else {
                  sendLine(controller, {
                    type: "status",
                    message: "Local model — answering without live web search",
                  });
                }
              }

              const fetched = noContext ? [] : await fetchIncludes(includes ?? []);
              const ragAll = noContext
                ? agentBlock
                  ? [{ source: "agent", text: agentBlock }]
                  : []
                : [
                    ...rag,
                    ...fetched,
                    ...(agentBlock
                      ? [{ source: "agent", text: agentBlock }]
                      : []),
                  ];
              const ragBlock = ragAll.length
                ? `Retrieved context:\n${ragAll
                    .map((c, i) => `[${i + 1}] (${c.source})\n${c.text}`)
                    .join("\n\n")}`
                : "";

              const history = messages.slice(-16);
              const last = history[history.length - 1];
              const prior = history.slice(0, -1);

              const xaiMessages: unknown[] = [
                {
                  role: "system",
                  content: systemFor(mode, ragBlock, {
                    agent: Boolean(useAgent),
                    noContext: Boolean(noContext),
                    rare,
                    oocDiagnostics,
                  }),
                },
                ...prior.map((m) => ({ role: m.role, content: m.content })),
              ];

              if (last.role === "user" && images.length) {
                xaiMessages.push({
                  role: "user",
                  content: [
                    {
                      type: "text",
                      text: last.content || "(see attached image)",
                    },
                    ...images.map((url) => ({
                      type: "image_url",
                      image_url: { url },
                    })),
                  ],
                });
              } else {
                xaiMessages.push({ role: last.role, content: last.content });
              }

              sendLine(controller, {
                type: "status",
                message: "Processing Prompt…",
                model: llm.model,
                route: mode === "story" ? "story" : "general",
                context: 0,
              });

              const completionsUrl = `${llm.baseUrl}/chat/completions`;
              const headers: Record<string, string> = {
                "Content-Type": "application/json",
                Authorization: `Bearer ${llm.apiKey}`,
              };
              if (llm.webSearch) {
                headers["x-grok-conv-id"] = `spur-${branchId}`.slice(0, 64);
              }

              let upstream: Response;
              try {
                upstream = await fetch(completionsUrl, {
                  method: "POST",
                  headers,
                  body: JSON.stringify({
                    model: llm.model,
                    stream: true,
                    max_tokens: 1600,
                    temperature: mode === "story" ? 0.9 : 0.4,
                    messages: xaiMessages,
                  }),
                });
              } catch {
                sendLine(controller, {
                  type: "error",
                  error:
                    "Could not reach the model server. If you are using LM Studio, start the local server and load a model.",
                });
                controller.close();
                return;
              }

              if (!upstream.ok || !upstream.body) {
                const errText = await upstream.text().catch(() => "");
                sendLine(controller, {
                  type: "error",
                  error:
                    upstream.status === 429
                      ? "The model is busy. Try again in a moment."
                      : errText.slice(0, 240) || "Upstream model error.",
                });
                controller.close();
                return;
              }

              const decoder = new TextDecoder();
              let carry = "";
              let announced = false;
              const reader = upstream.body.getReader();
              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                carry += decoder.decode(value, { stream: true });
                const lines = carry.split("\n");
                carry = lines.pop() ?? "";
                for (const line of lines) {
                  const trimmed = line.trim();
                  if (!trimmed.startsWith("data:")) continue;
                  const payload = trimmed.slice(5).trim();
                  if (!payload) continue;
                  if (payload === "[DONE]") {
                    sendLine(controller, { type: "done" });
                    continue;
                  }
                  let chunk: XaiChunk;
                  try {
                    chunk = JSON.parse(payload) as XaiChunk;
                  } catch {
                    continue;
                  }
                  const delta = chunk.choices?.[0]?.delta;
                  if (delta && !announced) {
                    announced = true;
                    sendLine(controller, {
                      type: "status",
                      message: "Streaming…",
                      model: llm.model,
                      route: mode === "story" ? "story" : "general",
                      context: 0,
                    });
                  }
                  if (delta?.reasoning_content) {
                    sendLine(controller, {
                      type: "reasoning",
                      content: delta.reasoning_content,
                    });
                  }
                  if (delta?.content) {
                    sendLine(controller, {
                      type: "token",
                      content: delta.content,
                    });
                  }
                  if (chunk.usage) {
                    sendLine(controller, {
                      type: "usage",
                      model: chunk.model ?? llm.model,
                      promptTokens: chunk.usage.prompt_tokens ?? 0,
                      completionTokens: chunk.usage.completion_tokens ?? 0,
                    });
                  }
                }
              }
              sendLine(controller, { type: "done" });
            } catch (err) {
              sendLine(controller, {
                type: "error",
                error: err instanceof Error ? err.message : "Stream failed",
              });
            } finally {
              controller.close();
            }
          },
        });

        return new Response(stream, {
          headers: {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            Connection: "keep-alive",
          },
        });
      },
    },
  },
});

async function runAgentSearch(
  llm: LlmTarget,
  query: string,
): Promise<{ text: string; error?: boolean }> {
  const today = new Date().toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
  try {
    const res = await fetch(`${llm.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${llm.apiKey}`,
      },
      body: JSON.stringify({
        model: llm.model,
        stream: false,
        max_tokens: 900,
        temperature: 0.3,
        tools: [{ type: "web_search" }],
        messages: [
          {
            role: "system",
            content: `You are a helpful research assistant. Today's date is ${today}. Use web search to find accurate, up-to-date information. Return concise notes with source URLs.`,
          },
          { role: "user", content: query },
        ],
      }),
    });
    if (!res.ok) return { text: "", error: true };
    const body = (await res.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    const text = body.choices?.[0]?.message?.content?.trim() ?? "";
    if (!text) return { text: "", error: true };
    return { text };
  } catch {
    return { text: "", error: true };
  }
}

async function fetchIncludes(
  urls: string[],
): Promise<{ source: string; text: string }[]> {
  const out: { source: string; text: string }[] = [];
  for (const url of urls.slice(0, 4)) {
    try {
      const parsed = new URL(url);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        continue;
      }
      const res = await fetch(url, {
        signal: AbortSignal.timeout(8000),
        headers: { Accept: "text/plain, text/html, application/json" },
        redirect: "follow",
      });
      if (!res.ok) continue;
      const ct = res.headers.get("content-type") ?? "";
      if (!/text\/|json|xml|javascript|markdown/i.test(ct) && ct !== "") {
        continue;
      }
      const text = (await res.text()).replace(/\s+/g, " ").trim().slice(0, 4000);
      if (text) out.push({ source: parsed.hostname, text });
    } catch {
      /* skip unreachable includes */
    }
  }
  return out;
}
