import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import {
  lastUserInputs,
  lastUserMessage,
  modeOf,
  stringifyBranch,
  turnCount,
} from "./branch-mode";
import { parseComposerInput, parseIncludes, SLASH_HELP } from "./commands";
import { retrieve } from "./rag";
import { chatPyOrigin, usesChatPy } from "./remote";
import { ragFromPending, useChatStore } from "./store";
import { streamChat, streamSse } from "./stream";
import { feedThink } from "./think";
import type { Message, RagChunk, StreamMetrics, TurnFlags } from "./types";

function estimateTokens(text: string): number {
  return Math.max(1, Math.round(text.length / 4));
}

export function useSend() {
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const inflight = useRef(false);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const generate = useCallback(
    async (opts: {
      text: string;
      regenerate?: boolean;
      agent?: boolean;
      noContext?: boolean;
      rare?: string[];
      ooc?: boolean;
      includeBranch?: string;
    }) => {
      const store = useChatStore.getState();
      const originId = store.currentId;
      const branch = store.branches[originId];
      if (!branch) return;
      if (inflight.current) return;

      if (usesChatPy()) {
        await generateViaChatPy(opts, inflight, abortRef, setStreaming);
        return;
      }

      if (opts.regenerate) {
        store.popLastAssistant();
      }

      const mode = modeOf(branch);
      const agent = Boolean(opts.agent) && mode === "assistant";
      const noContext = Boolean(opts.noContext);
      const ooc = Boolean(opts.ooc);
      const includes = parseIncludes(opts.text);
      if (includes.paths.length) {
        toast.message(
          "Filesystem includes aren't available here — attach the file instead.",
        );
      }

      const pending = opts.regenerate
        ? lastUserMessage(useChatStore.getState().branches[originId]?.messages ?? [])
            ?.attachments ?? []
        : store.pendingAttachments;
      const ragExtra: RagChunk[] = noContext ? [] : ragFromPending(pending);
      if (opts.includeBranch && !noContext) {
        const other =
          store.branches[opts.includeBranch] ??
          Object.values(store.branches).find(
            (b) => b.name === opts.includeBranch,
          );
        if (!other) {
          toast.error(`Unknown branch '${opts.includeBranch}'.`);
          return;
        }
        ragExtra.push({
          id: `include-${other.id}`,
          source: `include:${other.name}`,
          text: stringifyBranch(other),
        });
      }

      const flags: TurnFlags | undefined =
        agent || noContext || ooc || opts.includeBranch
          ? {
              agent,
              noContext,
              ooc,
              includeBranch: opts.includeBranch,
            }
          : undefined;

      if (!opts.regenerate) {
        const trimmed = opts.text.trim();
        if (!trimmed && pending.length === 0) return;
        store.appendMessage(
          {
            id: crypto.randomUUID(),
            role: "user",
            content: trimmed || "(attachment)",
            attachments: pending.length ? pending : undefined,
            flags,
            createdAt: Date.now(),
          },
          ragExtra,
          originId,
        );
        store.clearPending();
      }

      inflight.current = true;
      const assistantId = crypto.randomUUID();
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        status: agent ? "Agent tool web search…" : "Processing Prompt…",
        createdAt: Date.now(),
      };
      store.appendMessage(assistantMsg, [], originId);

      const latest = useChatStore.getState().branches[originId];
      const history = (latest?.messages ?? [])
        .filter((m) => m.id !== assistantId)
        .slice(-16)
        .map((m) => ({ role: m.role, content: m.content }));

      const collection = noContext ? [] : (latest?.rag ?? []);
      const query = opts.text.trim();
      const hits = noContext ? [] : retrieve(collection, query, 4);
      const images = pending
        .filter((a) => a.kind === "image" && a.dataUrl)
        .map((a) => a.dataUrl!)
        .slice(0, 3);

      const oocDiagnostics = store.pendingOoc;
      store.setPendingOoc("");

      const started = performance.now();
      let first = true;
      let ttft = 0;
      let content = "";
      let reasoning = "";
      let promptTokens = 0;
      let completionTokens = 0;
      let tokenSavings = 0;
      let model = "grok-4.5";
      let route = "";
      let context = 0;
      const think = { inThink: false, neverThink: false };

      const ac = new AbortController();
      abortRef.current = ac;
      setStreaming(true);

      const patch = (partial: Partial<Message>) => {
        useChatStore.getState().replaceMessage(assistantId, partial, originId);
      };

      try {
        await streamChat(
          {
            mode,
            branchId: originId,
            messages: history,
            rag: hits.map((h) => ({ source: h.source, text: h.text })),
            images,
            includes: includes.urls,
            useAgent: agent,
            noContext,
            rare: opts.rare,
            oocDiagnostics: oocDiagnostics || undefined,
          },
          (event) => {
            if (event.type === "status") {
              if (event.model) model = event.model;
              if (event.route) route = event.route;
              if (event.context) context = event.context;
              patch({
                status: event.message,
                streamingModel: event.model || model || undefined,
                streamingRoute: event.route || route || undefined,
                streamingContext: event.context || context || undefined,
              });
            } else if (event.type === "token") {
              if (first) {
                ttft = (performance.now() - started) / 1000;
                first = false;
              }
              const split = feedThink(event.content, think);
              content += split.content;
              reasoning += split.reasoning;
              patch({
                content,
                reasoning: reasoning || undefined,
                status: content ? undefined : "Streaming…",
                streamingModel: content ? undefined : model || undefined,
                streamingRoute: content ? undefined : route || undefined,
                streamingContext: content ? undefined : context || undefined,
              });
            } else if (event.type === "reasoning") {
              if (first) {
                ttft = (performance.now() - started) / 1000;
                first = false;
              }
              reasoning += event.content;
              patch({
                reasoning,
                status: content ? undefined : "Streaming…",
                streamingModel: content ? undefined : model || undefined,
                streamingRoute: content ? undefined : route || undefined,
                streamingContext: content ? undefined : context || undefined,
              });
            } else if (event.type === "usage") {
              promptTokens = event.promptTokens;
              completionTokens = event.completionTokens;
              if (event.tokenSavings != null) tokenSavings = event.tokenSavings;
              model = event.model;
            } else if (event.type === "error") {
              content = content || event.error;
              patch({ content, status: undefined });
            }
          },
          ac.signal,
        );
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          if (!content) patch({ content: "Generation stopped." });
        } else {
          const message =
            err instanceof Error ? err.message : "Something went wrong.";
          patch({ content: content || message, status: undefined });
        }
      } finally {
        const generationTime = (performance.now() - started) / 1000;
        const tokenCount = completionTokens || estimateTokens(content);
        const metrics: StreamMetrics = {
          model,
          tokenCount,
          generationTime,
          promptTokens: promptTokens || context,
          tokenSavings,
          ttft,
        };
        useChatStore.getState().replaceMessage(
          assistantId,
          {
            content,
            reasoning: reasoning || undefined,
            metrics,
            status: undefined,
          },
          originId,
        );
        if (ooc || /^\s*(?:OOC:|SYSTEM:|OOC>)/i.test(content)) {
          useChatStore.getState().setPendingOoc(content);
        }
        setStreaming(false);
        abortRef.current = null;
        inflight.current = false;
      }
    },
    [],
  );

  const send = useCallback(
    async (text: string) => {
      const parsed = parseComposerInput(text);
      const store = useChatStore.getState();
      const branch = store.branches[store.currentId];
      if (!branch) return;

      if (parsed.kind === "command") {
        handleLocalCommand(parsed.command, parsed.args, generate);
        return;
      }

      if (parsed.kind === "include") {
        if (!parsed.branch) {
          toast.message("Usage: \\include BRANCH your question");
          return;
        }
        if (!parsed.text && store.pendingAttachments.length === 0) {
          toast.message("Usage: \\include BRANCH your question");
          return;
        }
        await generate({
          text: parsed.text,
          includeBranch: parsed.branch,
          rare: parsed.rare,
          ooc: parsed.ooc,
        });
        return;
      }

      if (parsed.kind === "inline") {
        if (!parsed.text && store.pendingAttachments.length === 0) {
          toast.message(
            parsed.agent
              ? "Usage: \\agent your question"
              : "Usage: \\no-context your question",
          );
          return;
        }
        if (parsed.agent && modeOf(branch) !== "assistant") {
          toast.message("Agent is only available in assistant mode.");
          return;
        }
        await generate({
          text: parsed.text,
          agent: parsed.agent,
          noContext: parsed.noContext,
          rare: parsed.rare,
          ooc: parsed.ooc,
        });
        return;
      }

      await generate({
        text: parsed.text,
        agent: store.forceAgent,
        noContext: false,
        rare: parsed.rare,
        ooc: parsed.ooc,
      });
    },
    [generate],
  );

  const regenerate = useCallback(async () => {
    if (inflight.current) return;
    const store = useChatStore.getState();
    const branch = store.branches[store.currentId];
    if (!branch) return;
    const last = lastUserMessage(branch.messages);
    if (!last || (!last.content && !last.attachments?.length)) {
      toast.message("Nothing to regenerate.");
      return;
    }
    const flags = last.flags;
    await generate({
      text: last.content,
      regenerate: true,
      agent: Boolean(flags?.agent || store.forceAgent),
      noContext: Boolean(flags?.noContext),
      includeBranch: flags?.includeBranch,
      ooc: Boolean(flags?.ooc),
    });
  }, [generate]);

  return { send, stop, regenerate, streaming };
}

function handleLocalCommand(
  command: string,
  args: string,
  generate: (opts: GenerateOpts) => Promise<void>,
) {
  const store = useChatStore.getState();
  const branch = store.branches[store.currentId];

  switch (command) {
    case "help":
      toast.message(SLASH_HELP.map((row) => `${row.cmd} — ${row.hint}`).join("\n"));
      return;
    case "turn":
      toast.message(`Turn ${turnCount(branch?.messages ?? [])}`);
      return;
    case "history": {
      const n = Math.max(1, Number.parseInt(args, 10) || 5);
      const lines = lastUserInputs(branch?.messages ?? [], n);
      toast.message(
        lines.length
          ? lines.map((t, i) => `${i + 1}. ${t}`).join("\n")
          : "No user turns yet.",
      );
      return;
    }
    case "branch": {
      if (!args.trim()) {
        toast.message(
          `Branches: ${Object.keys(store.branches).sort().join(", ")}`,
        );
        return;
      }
      const result = store.createBranch(args.trim());
      if (result.ok) toast.success(`Branched to ${result.id}`);
      else toast.error(result.error);
      return;
    }
    case "reset": {
      const result = store.resetBranch();
      if (result.ok) toast.success(`Reset '${store.currentId}'.`);
      else toast.error(result.error);
      return;
    }
    case "delete-last": {
      const result = store.deleteLastTurn();
      if (result.ok) toast.success("Deleted last turn.");
      else toast.message(result.error);
      return;
    }
    case "rewind": {
      const n = Number.parseInt(args, 10);
      const result = store.rewindTo(n);
      if (result.ok) toast.success(`Rewound to turn ${n}.`);
      else toast.error(result.error);
      return;
    }
    case "dbranch": {
      const result = store.deleteBranch(args.trim());
      if (result.ok) toast.success(`Deleted '${args.trim()}'.`);
      else toast.error(result.error);
      return;
    }
    case "regenerate": {
      const last = lastUserMessage(branch?.messages ?? []);
      if (!last || (!last.content && !last.attachments?.length)) {
        toast.message("Nothing to regenerate.");
        return;
      }
      const flags = last.flags;
      void generate({
        text: last.content,
        regenerate: true,
        agent: Boolean(flags?.agent || store.forceAgent),
        noContext: Boolean(flags?.noContext),
        includeBranch: flags?.includeBranch,
        ooc: Boolean(flags?.ooc),
      });
      return;
    }
    default:
      return;
  }
}

type GenerateOpts = {
  text: string;
  regenerate?: boolean;
  agent?: boolean;
  noContext?: boolean;
  rare?: string[];
  ooc?: boolean;
  includeBranch?: string;
};

async function generateViaChatPy(
  opts: GenerateOpts,
  inflight: { current: boolean },
  abortRef: { current: AbortController | null },
  setStreaming: (v: boolean) => void,
) {
  const store = useChatStore.getState();
  const originId = store.currentId;
  const branch = store.branches[originId];
  if (!branch) return;

  if (opts.regenerate) store.popLastAssistant();

  const mode = modeOf(branch);
  const agent = Boolean(opts.agent) && mode === "assistant";
  const noContext = Boolean(opts.noContext);
  const pending = opts.regenerate
    ? lastUserMessage(useChatStore.getState().branches[originId]?.messages ?? [])
        ?.attachments ?? []
    : store.pendingAttachments;
  const flags: TurnFlags | undefined =
    agent || noContext || opts.ooc || opts.includeBranch
      ? {
          agent,
          noContext,
          ooc: opts.ooc,
          includeBranch: opts.includeBranch,
        }
      : undefined;

  if (!opts.regenerate) {
    const trimmed = opts.text.trim();
    if (!trimmed && pending.length === 0) return;
    store.appendMessage(
      {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmed || "(attachment)",
        attachments: pending.length ? pending : undefined,
        flags,
        createdAt: Date.now(),
      },
      [],
      originId,
    );
    store.clearPending();
  }

  inflight.current = true;
  const assistantId = crypto.randomUUID();
  store.appendMessage(
    {
      id: assistantId,
      role: "assistant",
      content: "",
      status: "Working — RAG / agent / prompt…",
      createdAt: Date.now(),
    },
    [],
    originId,
  );

  const images = pending
    .filter((a) => a.kind === "image" && a.dataUrl)
    .map((a) => a.dataUrl!)
    .slice(0, 3);
  const files = pending
    .filter((a) => a.kind === "text" && a.text)
    .map((a) => ({ name: a.name, text: a.text! }));

  const started = performance.now();
  let first = true;
  let ttft = 0;
  let content = "";
  let reasoning = "";
  let promptTokens = 0;
  let completionTokens = 0;
  let tokenSavings = 0;
  let model = "";
  let route = "";
  let context = 0;
  let recalling = false;
  let recalled: string[] = [];

  const ac = new AbortController();
  abortRef.current = ac;
  setStreaming(true);
  const patch = (partial: Partial<Message>) => {
    useChatStore.getState().replaceMessage(assistantId, partial, originId);
  };

  try {
    await streamSse(
      `${chatPyOrigin()}/api/chat`,
      {
        text: opts.text,
        regenerate: Boolean(opts.regenerate),
        useAgent: agent,
        noContext,
        rare: opts.rare,
        includeBranch: opts.includeBranch,
        images,
        files,
        attachments: pending.map((a) => ({
          id: a.id,
          name: a.name,
          mime: a.mime,
          kind: a.kind,
          size: a.size,
          text: a.kind === "text" ? a.text : undefined,
          // dataUrl stays in `images` for vision; pickling it truncates history.
          dataUrl: a.kind === "image" ? a.dataUrl : undefined,
        })),
      },
      (event) => {
        if (event.type === "documents") {
          window.dispatchEvent(new Event("spur-documents"));
        } else if (event.type === "status") {
          if (event.model) model = event.model;
          if (event.route) route = event.route;
          if (event.context) context = event.context;
          recalling = /Recalling Documents?/i.test(event.message || "");
          if (event.recalled?.length) recalled = event.recalled;
          patch({
            status: event.message,
            streamingModel: recalling ? undefined : event.model || model || undefined,
            streamingRoute: recalling ? undefined : event.route || route || undefined,
            streamingContext: recalling ? undefined : event.context || context || undefined,
            recalled: recalled.length ? recalled : undefined,
          });
        } else if (event.type === "token") {
          if (first) {
            ttft = (performance.now() - started) / 1000;
            first = false;
          }
          content += event.content;
          patch({
            content,
            reasoning: reasoning || undefined,
            ...(recalling
              ? {}
              : {
                  status: "Streaming…",
                  streamingModel: model || undefined,
                  streamingRoute: route || undefined,
                  streamingContext: context || undefined,
                }),
          });
        } else if (event.type === "reasoning") {
          if (first) {
            ttft = (performance.now() - started) / 1000;
            first = false;
          }
          reasoning += event.content;
          patch({
            reasoning,
            ...(recalling
              ? {}
              : {
                  status: "Streaming…",
                  streamingModel: model || undefined,
                  streamingRoute: route || undefined,
                  streamingContext: context || undefined,
                }),
          });
        } else if (event.type === "usage") {
          promptTokens = event.promptTokens;
          completionTokens = event.completionTokens;
          if (event.tokenSavings != null) tokenSavings = event.tokenSavings;
          model = event.model;
        } else if (event.type === "error") {
          content = content || event.error;
          patch({ content, status: undefined });
        }
      },
      ac.signal,
    );
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      if (!content) patch({ content: "Generation stopped." });
    } else {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      patch({ content: content || message, status: undefined });
    }
  } finally {
    const generationTime = (performance.now() - started) / 1000;
    const tokenCount = completionTokens || estimateTokens(content);
    useChatStore.getState().replaceMessage(
      assistantId,
      {
        content,
        reasoning: reasoning || undefined,
        metrics: {
          model,
          tokenCount,
          generationTime,
          promptTokens: promptTokens || context,
          tokenSavings,
          ttft,
        },
        recalled: recalled.length ? recalled : undefined,
        status: undefined,
      },
      originId,
    );
    setStreaming(false);
    abortRef.current = null;
    inflight.current = false;
  }
}
