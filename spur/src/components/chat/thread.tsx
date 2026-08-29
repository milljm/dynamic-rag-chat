import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  ArrowDown,
  Bot,
  BookOpen,
  GitBranch,
  Globe,
  Lock,
  PanelLeft,
} from "lucide-react";
import { isLockedBranch, modeOf, turnCount } from "@/lib/chat/branch-mode";
import { useChatStore } from "@/lib/chat/store";
import type { Message } from "@/lib/chat/types";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ThemeToggle } from "./theme-toggle";
import { Markdown } from "./markdown";

const NEAR_BOTTOM = 96;

export function Thread({
  streaming,
  onRevealSidebar,
}: {
  streaming: boolean;
  onRevealSidebar?: () => void;
}) {
  const currentId = useChatStore((s) => s.currentId);
  const branch = useChatStore((s) => s.branches[s.currentId]);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const [pinned, setPinned] = useState(true);

  function pinToBottom() {
    pinnedRef.current = true;
    setPinned(true);
    const el = scrollerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }

  function releasePin() {
    pinnedRef.current = false;
    setPinned(false);
  }

  useEffect(() => {
    pinToBottom();
  }, [currentId]);

  useEffect(() => {
    if (!pinnedRef.current) return;
    const el = scrollerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [branch?.messages, streaming]);

  if (!branch) return null;

  const mode = modeOf(branch);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex items-center gap-3 border-b border-border px-4 py-3 md:px-8">
        {onRevealSidebar && (
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="hidden md:inline-flex"
            aria-label="Show sidebar"
            onClick={onRevealSidebar}
          >
            <PanelLeft />
          </Button>
        )}
        <GitBranch className="size-4 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h1 className="truncate text-sm font-medium">{branch.name}</h1>
            {isLockedBranch(branch.id) && (
              <Lock className="size-3 text-muted-foreground" />
            )}
          </div>
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
            {mode === "assistant" ? (
              <Bot className="size-3" />
            ) : (
              <BookOpen className="size-3" />
            )}
            <span className="capitalize">{mode}</span>
            <span className="font-mono tabular-nums">
              · {turnCount(branch.messages)} turns
            </span>
          </p>
        </div>
        <ThemeToggle />
      </header>

      <div className="relative min-h-0 flex-1">
        <div
          ref={scrollerRef}
          className="h-full overflow-y-auto [overflow-anchor:none]"
          onScroll={(e) => {
            const el = e.currentTarget;
            const near =
              el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM;
            if (near !== pinnedRef.current) {
              pinnedRef.current = near;
              setPinned(near);
            }
            }}
        >
          <div className="mx-auto flex max-w-3xl flex-col gap-5 px-4 py-6 md:px-8">
            {branch.messages.length === 0 ? (
              <EmptyState
                name={branch.name}
                mode={mode}
                locked={isLockedBranch(branch.id)}
              />
            ) : (
              branch.messages.map((msg, i) => (
                <MessageBubble
                  key={msg.id}
                  message={msg}
                  turn={
                    msg.role === "user"
                      ? branch.messages
                          .slice(0, i + 1)
                          .filter((m) => m.role === "user").length
                      : undefined
                  }
                  pending={
                    streaming &&
                    msg.role === "assistant" &&
                    i === branch.messages.length - 1
                  }
                  onInspect={releasePin}
                />
              ))
            )}
          </div>
        </div>
        {!pinned && (
          <button
            type="button"
            className="absolute bottom-4 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1.5 rounded-full bg-popover px-3 py-1.5 text-xs text-popover-foreground shadow-[var(--shadow-border)]"
            onClick={pinToBottom}
          >
            <ArrowDown className="size-3" />
            {streaming ? "Resume live" : "Jump to latest"}
          </button>
        )}
      </div>
    </div>
  );
}

function EmptyState({
  name,
  mode,
  locked,
}: {
  name: string;
  mode: "assistant" | "story";
  locked: boolean;
}) {
  const lock = locked
    ? `, a protected branch locked to ${mode} mode.`
    : ` in ${mode} mode — toggle freely, or fork to keep this path.`;
  return (
    <div className="flex flex-col items-start gap-3 py-16">
      <p className="font-display text-3xl italic tracking-tight text-foreground">
        {mode === "story" ? "Pick up the thread." : "Ask with context."}
      </p>
      <p className="max-w-md text-sm leading-relaxed text-muted-foreground">
        You are on <span className="text-foreground">{name}</span>
        {lock} Paperclip a file on this turn; after that it lives under
        Documents — mention it by name to bring it back.{" "}
        {mode === "story" ? (
          <>
            Switch to the <span className="text-foreground">assistant</span>{" "}
            branch for research, tools, or live search.
          </>
        ) : (
          <>
            Use <span className="font-mono text-foreground">\agent</span> for
            live search, or switch to the{" "}
            <span className="text-foreground">story</span> branch to write.
          </>
        )}
      </p>
    </div>
  );
}

function MessageBubble({
  message,
  pending,
  turn,
  onInspect,
}: {
  message: Message;
  pending: boolean;
  turn?: number;
  onInspect?: () => void;
}) {
  const isUser = message.role === "user";
  return (
    <article
      className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}
    >
      <div
        className={cn(
          "relative max-w-[min(100%,40rem)] px-4 py-3",
          isUser
            ? "ml-auto rounded-lg rounded-br-xs bg-user-bubble"
            : "rounded-lg rounded-bl-xs bg-assistant-bubble",
        )}
      >
        {isUser && turn != null && turn > 0 ? (
          <span
            className="pointer-events-none absolute -right-1 -top-2 z-10 rounded-md px-1.5 py-px font-mono text-[10px] tabular-nums tracking-tight text-turn"
            style={{
              background: "var(--spur-turn-bg)",
              boxShadow: "var(--spur-turn-shadow)",
            }}
            title={`Turn ${turn}`}
          >
            {turn}
          </span>
        ) : null}
        {message.attachments && message.attachments.length > 0 && (
          <ul className="mb-2 space-y-1">
            {message.attachments.map((att) => (
              <li key={att.id} className="text-xs text-muted-foreground">
                {att.name}
                {att.kind === "image" && att.dataUrl && (
                  <img
                    src={att.dataUrl}
                    alt={att.name}
                    className="mt-2 max-h-48 rounded-sm outline outline-1 -outline-offset-1 outline-foreground/10"
                  />
                )}
              </li>
            ))}
          </ul>
        )}
        {(message.flags?.agent ||
          message.flags?.noContext ||
          message.flags?.includeBranch ||
          message.flags?.ooc) && (
          <p className="mb-2 flex items-center gap-1.5 text-xs text-muted-foreground">
            {message.flags.agent && (
              <>
                <Globe className="size-3" />
                Agent
              </>
            )}
            {message.flags.noContext && <span>No context</span>}
            {message.flags.includeBranch && (
              <span>Include {message.flags.includeBranch}</span>
            )}
            {message.flags.ooc && <span>OOC</span>}
          </p>
        )}
        {message.reasoning && (
          <details
            className="mb-2 text-xs text-muted-foreground"
            onToggle={(e) => {
              if (!(e.currentTarget as HTMLDetailsElement).open) return;
              onInspect?.();
              e.currentTarget.scrollIntoView({ block: "nearest" });
            }}
          >
            <summary className="cursor-pointer select-none">Reasoning</summary>
            <div className="mt-2 max-h-64 overflow-y-auto whitespace-pre-wrap leading-relaxed">
              {message.reasoning}
            </div>
          </details>
        )}
        {message.content ? (
          <Markdown text={message.content} />
        ) : null}
        {pending ? (
          <StatusLine
            status={message.status}
            model={message.streamingModel}
            route={message.streamingRoute}
            context={message.streamingContext}
            recalled={message.recalled}
          />
        ) : null}
        {message.metrics && !pending && (
          <p className="mt-3 flex flex-wrap items-center gap-x-1.5 font-mono text-[10px] tabular-nums text-muted-foreground">
            <FootStat tip="Time to first token">
              TTFT {message.metrics.ttft.toFixed(2)}s
            </FootStat>
            <span aria-hidden>·</span>
            <FootStat tip="Generation time after first token">
              Gen{" "}
              {(message.metrics.generationTime - message.metrics.ttft).toFixed(2)}
              s
            </FootStat>
            <span aria-hidden>·</span>
            <FootStat tip="Completion tokens">
              {message.metrics.tokenCount} tok
            </FootStat>
            <span aria-hidden>·</span>
            <FootStat tip="Tokens per second">
              {tps(message.metrics).toFixed(1)} T/s
            </FootStat>
            <span aria-hidden>·</span>
            <FootStat tip="De-duplicated tokens">
              DUP {fmtK(message.metrics.tokenSavings)}
            </FootStat>
            <span aria-hidden>·</span>
            <FootStat tip="Packed context size">
              CTX {fmtK(message.metrics.promptTokens)}
            </FootStat>
            <span aria-hidden>·</span>
            <FootStat tip="Model">{message.metrics.model}</FootStat>
            {message.recalled?.length ? (
              <FootStat tip={`Recalled ${message.recalled.join(", ")}`}>
                <span className="ml-0.5 text-muted-foreground/40">📄</span>
              </FootStat>
            ) : null}
          </p>
        )}
      </div>
    </article>
  );
}


function FootStat({
  tip,
  children,
}: {
  tip: string;
  children: ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="cursor-help">{children}</span>
      </TooltipTrigger>
      <TooltipContent side="top">{tip}</TooltipContent>
    </Tooltip>
  );
}

function fmtK(n?: number): string {
  if (!n || n <= 0) return "0";
  if (n < 1000) return String(Math.round(n));
  const k = n / 1000;
  const s = k >= 10 ? k.toFixed(0) : k.toFixed(1).replace(/\.0$/, "");
  return `${s}K`;
}

function fmtContext(n?: number): string {
  if (!n || n <= 0) return "";
  if (n < 1000) return `[${n}]`;
  const k = n / 1000;
  return `[${k >= 10 ? k.toFixed(0) : k.toFixed(1)}k]`;
}

function RecallMark({ names }: { names?: string[] }) {
  if (!names?.length) return null;
  return (
    <span
      className="ml-1.5 text-[10px] text-muted-foreground/40"
      title={`Recalled ${names.join(", ")}`}
    >
      📄
    </span>
  );
}

function StatusLine({
  status,
  model,
  route,
  context,
  recalled,
}: {
  status?: string;
  model?: string;
  route?: string;
  context?: number;
  recalled?: string[];
}) {
  const label = status || "Processing Prompt…";
  const recall = label.match(/^(Recalling Documents?…?)\s*(\[.*\])?\s*$/i);
  if (recall) {
    return (
      <p className="text-sm text-muted-foreground">
        <span className="shimmer-text">{recall[1]}</span>
        {recall[2] ? (
          <span className="ml-1.5 font-mono text-[10px] font-normal tracking-tight text-muted-foreground/40">
            {recall[2]}
          </span>
        ) : null}
      </p>
    );
  }
  const showModel =
    Boolean(model) && /^(Streaming|Processing Prompt)/i.test(label);
  return (
    <p className="text-sm text-muted-foreground">
      <span className="shimmer-text">{label}</span>
      {showModel ? (
        <span className="ml-1.5 font-mono text-[10px] font-normal tracking-tight text-muted-foreground/40">
          [{model}]
          {route ? ` [${route}]` : ""}
          {context ? ` ${fmtContext(context)}` : ""}
          <RecallMark names={recalled} />
        </span>
      ) : (
        <RecallMark names={recalled} />
      )}
    </p>
  );
}

function tps(m: Message["metrics"]): number {
  if (!m) return 0;
  const gen = m.generationTime - m.ttft;
  if (gen <= 0) return 0;
  return m.tokenCount / gen;
}