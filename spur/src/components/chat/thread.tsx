import {
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { Virtuoso, type VirtuosoHandle, type Components } from "react-virtuoso";
import { useShallow } from "zustand/react/shallow";
import {
  ArrowDown,
  Bot,
  BookOpen,
  Check,
  ChevronRight,
  GitBranch,
  Globe,
  Lock,
  Palette,
  PanelLeft,
  Pencil,
  Trash2,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { isLockedBranch, modeOf } from "@/lib/chat/branch-mode";
import { deleteDocument, usesChatPy } from "@/lib/chat/remote";
import { useChatStore } from "@/lib/chat/store";
import type { Message } from "@/lib/chat/types";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ThemeToggle } from "./theme-toggle";
import { SettingsButton } from "./settings-panel";
import { Markdown } from "./markdown";
import { ChatImage } from "./chat-image";

const NEAR_BOTTOM = 96;

// Item wrapper restores the old thread container's centering and its
// gap-5/py-6 rhythm inside the virtualized list.
const VIRTUOSO_COMPONENTS: Components<string> = {
  Header: () => <div className="h-6" aria-hidden="true" />,
  Footer: () => <div className="h-1" aria-hidden="true" />,
  Item: ({ children, ...props }) => (
    <div {...props} className="mx-auto w-full max-w-3xl px-4 pb-5 md:px-8">
      {children}
    </div>
  ),
};

export function Thread({
  streaming,
  onRevealSidebar,
  onEditUser,
}: {
  streaming: boolean;
  onRevealSidebar?: () => void;
  onEditUser?: (messageId: string, text: string) => void;
}) {
  const currentId = useChatStore((s) => s.currentId);
  // Primitive / shallow-compared subscriptions only: per-token content
  // patches must not re-render the thread shell. Only the streaming
  // bubble (via its own message subscription below) updates per frame.
  const hasBranch = useChatStore((s) => Boolean(s.branches[s.currentId]));
  const ids = useChatStore(
    useShallow((s) => s.branches[s.currentId]?.messages.map((m) => m.id) ?? []),
  );
  const turns = useChatStore(
    useShallow((s) => {
      const messages = s.branches[s.currentId]?.messages;
      if (!messages) return [];
      const out: number[] = [];
      let n = 0;
      for (const m of messages) {
        if (m.role === "user") n += 1;
        out.push(m.role === "user" ? n : 0);
      }
      return out;
    }),
  );
  const name = useChatStore((s) => s.branches[s.currentId]?.name ?? "");
  const mode = useChatStore((s) => {
    const branch = s.branches[s.currentId];
    return branch ? modeOf(branch) : "assistant";
  });
  const locked = useChatStore((s) => {
    const branch = s.branches[s.currentId];
    return branch ? isLockedBranch(branch.id) : false;
  });

  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const pinnedRef = useRef(true);
  const [pinned, setPinned] = useState(true);

  // Opening a reasoning frame must not fight the live follow scroll.
  const releasePin = useCallback(() => {
    pinnedRef.current = false;
    setPinned(false);
  }, []);

  const handleAtBottom = useCallback((atBottom: boolean) => {
    pinnedRef.current = atBottom;
    setPinned(atBottom);
  }, []);

  // A branch switch swaps the whole list; the Virtuoso remount below
  // (key={currentId}) lands on the newest item by itself.
  useEffect(() => {
    pinnedRef.current = true;
    setPinned(true);
  }, [currentId]);

  function jumpToLatest() {
    pinnedRef.current = true;
    setPinned(true);
    const last = ids.length - 1;
    if (last >= 0) {
      virtuosoRef.current?.scrollToIndex({
        index: last,
        align: "end",
        behavior: "smooth",
      });
    }
  }

  const turnTotal = turns.reduce((n, t) => n + (t > 0 ? 1 : 0), 0);

  const renderItem = useCallback(
    (index: number, id: string) => (
      <MessageBubble
        messageId={id}
        turn={turns[index] || undefined}
        isLast={index === ids.length - 1}
        streaming={streaming}
        onEditUser={onEditUser}
        onInspect={releasePin}
      />
    ),
    [ids, turns, streaming, onEditUser, releasePin],
  );

  if (!hasBranch) return null;

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
            <h1 className="truncate text-sm font-medium">{name}</h1>
            {locked && <Lock className="size-3 text-muted-foreground" />}
          </div>
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
            {mode === "assistant" ? (
              <Bot className="size-3" />
            ) : (
              <BookOpen className="size-3" />
            )}
            <span className="capitalize">{mode}</span>
            <span className="font-mono tabular-nums">· {turnTotal} turns</span>
          </p>
        </div>
        <SettingsButton streaming={streaming} />
        <ThemeToggle />
      </header>

      <div className="relative min-h-0 flex-1">
        {ids.length === 0 ? (
          <div className="h-full overflow-y-auto">
            <div className="mx-auto max-w-3xl px-4 py-6 md:px-8">
              <EmptyState name={name} mode={mode} locked={locked} />
            </div>
          </div>
        ) : (
          <Virtuoso
            key={currentId}
            ref={virtuosoRef}
            className="h-full"
            data={ids}
            computeItemKey={(_, id) => id}
            initialTopMostItemIndex={ids.length - 1}
            followOutput={pinned ? "smooth" : false}
            atBottomStateChange={handleAtBottom}
            atBottomThreshold={NEAR_BOTTOM}
            increaseViewportBy={{ top: 640, bottom: 640 }}
            itemContent={renderItem}
            components={VIRTUOSO_COMPONENTS}
          />
        )}
        {!pinned && (
          <button
            type="button"
            className="absolute bottom-4 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1.5 rounded-full bg-popover px-3 py-1.5 text-xs text-popover-foreground shadow-[var(--shadow-border)]"
            onClick={jumpToLatest}
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

const MessageBubble = memo(function MessageBubble({
  messageId,
  turn,
  isLast,
  streaming,
  onEditUser,
  onInspect,
}: {
  messageId: string;
  turn?: number;
  isLast: boolean;
  streaming: boolean;
  onEditUser?: (messageId: string, text: string) => void;
  onInspect?: () => void;
}) {
  // Subscribe to this message only. applyReplaceMessage keeps untouched
  // message objects referentially stable, so a streamed token re-renders
  // exactly one bubble instead of the whole thread.
  const bubbleMessage = useChatStore((s) => {
    const branch = s.branches[s.currentId];
    return branch?.messages.find((m) => m.id === messageId);
  });
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  if (!bubbleMessage) return null;
  // Definite type for the hoisted edit helpers below.
  const message = bubbleMessage;

  const isUser = message.role === "user";
  const pending = streaming && isLast && message.role === "assistant";
  const ragNames = message.ragIds?.length ? message.ragIds : message.recalled;

  function startEdit() {
    setDraft(message.content);
    setEditing(true);
  }

  function cancelEdit() {
    setEditing(false);
    setDraft(message.content);
  }

  function saveEdit() {
    const next = draft.trim();
    if (!next && !message.attachments?.length) return;
    setEditing(false);
    onEditUser?.(message.id, next || message.content);
  }

  const editLineCount = Math.max(
    4,
    draft.split("\n").reduce(
      (n, line) => n + Math.max(1, Math.ceil(line.length / 52)),
      0,
    ),
  );

  return (
    <article
      className={cn(
        "group flex w-full",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      <div
        className={cn(
          "relative max-w-[min(100%,40rem)] px-4 py-3",
          isUser
            ? "ml-auto rounded-lg rounded-br-xs bg-user-bubble"
            : "rounded-lg rounded-bl-xs bg-assistant-bubble",
          // Shrink-to-fit + textarea width:100% collapses to ~20ch.
          editing && "w-full min-w-[min(100%,20rem)]",
        )}
      >
        {isUser && turn != null && turn > 0 ? (
          <CornerChip side="right" title={`Turn ${turn}`}>
            {turn}
          </CornerChip>
        ) : null}
        {!isUser ? <TokenChip message={message} pending={pending} /> : null}
        {message.attachments && message.attachments.length > 0 && (
          <ul className="mb-2 space-y-1">
            {message.attachments.map((att) => (
              <li key={att.id} className="text-xs text-muted-foreground">
                {att.name}
                {att.kind === "image" && att.dataUrl && (
                  <ChatImage
                    src={att.dataUrl}
                    alt={att.name}
                    name={att.name}
                    prompt={att.prompt}
                    negative={att.negative}
                  />
                )}
              </li>
            ))}
          </ul>
        )}
        {(message.flags?.agent ||
          message.flags?.image ||
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
            {message.flags.image && (
              <>
                <Palette className="size-3" />
                Image
              </>
            )}
            {message.flags.noContext && <span>No context</span>}
            {message.flags.includeBranch && (
              <span>Include {message.flags.includeBranch}</span>
            )}
            {message.flags.ooc && <span>OOC</span>}
          </p>
        )}
        <ReasoningFrame
          text={message.reasoning}
          pending={pending}
          onInspect={onInspect}
        />
        {editing ? (
          <div className="w-full space-y-2">
            <Textarea
              value={draft}
              rows={Math.min(16, editLineCount)}
              aria-label="Edit message"
              className="min-h-24 w-full min-w-0 resize-y bg-background/60"
              autoFocus
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  e.preventDefault();
                  e.stopPropagation();
                  cancelEdit();
                }
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  e.stopPropagation();
                  saveEdit();
                }
              }}
            />
            <div className="flex justify-end gap-1">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                aria-label="Cancel edit"
                onClick={cancelEdit}
              >
                <X className="size-3.5" />
                Cancel
              </Button>
              <Button
                type="button"
                size="sm"
                aria-label="Save and re-run"
                disabled={!draft.trim() && !message.attachments?.length}
                onClick={saveEdit}
              >
                <Check className="size-3.5" />
                Re-run
              </Button>
            </div>
          </div>
        ) : message.content ? (
          <Markdown text={message.content} />
        ) : null}
        {isUser && onEditUser && !streaming && !editing ? (
          <div className="mt-2 flex justify-end">
            <button
              type="button"
              className="flex size-7 items-center justify-center rounded-sm text-muted-foreground/50 transition-colors hover:bg-accent hover:text-foreground"
              aria-label="Edit message"
              title="Edit and re-run from here"
              onClick={startEdit}
            >
              <Pencil className="size-3.5" />
            </button>
          </div>
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
        {!isUser && !pending && ragNames?.length ? (
          <RagIdList names={ragNames} />
        ) : null}
      </div>
    </article>
  );
});

function RagIdList({ names }: { names: string[] }) {
  const canDelete = usesChatPy();
  return (
    <ul className="mt-2 flex flex-wrap gap-1">
      {names.map((name) => (
        <li
          key={name}
          className="flex max-w-full items-center gap-1 rounded-sm bg-secondary px-1.5 py-0.5 text-[10px] text-muted-foreground"
          title={`RAG: ${name}`}
        >
          <span className="min-w-0 truncate font-mono">{name}</span>
          {canDelete ? (
            <button
              type="button"
              className="text-muted-foreground hover:text-destructive"
              aria-label={`Delete ${name} from documents`}
              onClick={async () => {
                if (!window.confirm(`Remove ${name} from gold? Chat history stays.`)) {
                  return;
                }
                const result = await deleteDocument(name);
                if (!result.ok) {
                  toast.error(result.error || `Could not delete ${name}`);
                  return;
                }
                toast.success(`Deleted ${name}`);
                window.dispatchEvent(new Event("spur-documents"));
              }}
            >
              <Trash2 className="size-3" />
            </button>
          ) : null}
        </li>
      ))}
    </ul>
  );
}


function CornerChip({
  side,
  title,
  leaving,
  children,
}: {
  side: "left" | "right";
  title?: string;
  leaving?: boolean;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "pointer-events-none absolute -top-2 z-10 rounded-md px-1.5 py-px font-mono text-[10px] tabular-nums tracking-tight text-turn transition-[opacity,transform] duration-500",
        side === "right" ? "-right-1" : "-left-1",
        leaving && "tok-chip-leave",
      )}
      style={{
        background: "var(--spur-turn-bg)",
        boxShadow: "var(--spur-turn-shadow)",
      }}
      title={title}
    >
      {children}
    </span>
  );
}

function liveTokenCount(message: Message, pending: boolean): number {
  if (!pending && message.metrics?.tokenCount) return message.metrics.tokenCount;
  const text = `${message.content || ""}${message.reasoning || ""}`;
  if (!text) return 0;
  return Math.max(1, Math.round(text.length / 4));
}

function useTokenChip(liveTokens: number, generating: boolean) {
  const [display, setDisplay] = useState(0);
  const [phase, setPhase] = useState<"hidden" | "live" | "hold" | "leave">("hidden");
  const last = useRef(0);
  const timers = useRef<{ hold: number | null; leave: number | null }>({
    hold: null,
    leave: null,
  });

  function clearTimers() {
    if (timers.current.hold != null) window.clearTimeout(timers.current.hold);
    if (timers.current.leave != null) window.clearTimeout(timers.current.leave);
    timers.current = { hold: null, leave: null };
  }

  useEffect(() => {
    if (generating && liveTokens > 0) {
      last.current = liveTokens;
      setDisplay(liveTokens);
      setPhase("live");
      clearTimers();
    }
  }, [generating, liveTokens]);

  useEffect(() => {
    if (generating) return;
    if (last.current <= 0) return;
    setDisplay(last.current);
    setPhase("hold");
    clearTimers();
    timers.current.hold = window.setTimeout(() => {
      setPhase("leave");
      timers.current.leave = window.setTimeout(() => {
        setPhase("hidden");
        setDisplay(0);
        last.current = 0;
      }, 500);
    }, 2000);
    return clearTimers;
  }, [generating]);

  return { display, phase };
}

function TokenChip({ message, pending }: { message: Message; pending: boolean }) {
  const n = liveTokenCount(message, pending);
  const { display, phase } = useTokenChip(n, pending);
  if (phase === "hidden" || display <= 0) return null;
  return (
    <CornerChip
      side="left"
      leaving={phase === "leave"}
      title={`${display.toLocaleString("en-US")} tokens`}
    >
      {display.toLocaleString("en-US")}
    </CornerChip>
  );
}

function ReasoningFrame({
  text,
  pending,
  onInspect,
}: {
  text?: string;
  pending: boolean;
  onInspect?: () => void;
}) {
  if (!text) return null;
  return (
    <details
      className="group mb-3 overflow-hidden rounded-md border border-border/70 bg-background/50"
      onToggle={(e) => {
        if (!(e.currentTarget as HTMLDetailsElement).open) return;
        onInspect?.();
        e.currentTarget.scrollIntoView({ block: "nearest" });
      }}
    >
      <summary className="flex cursor-pointer list-none items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-medium tracking-wide text-muted-foreground select-none [&::-webkit-details-marker]:hidden">
        <ChevronRight className="size-3 shrink-0 transition-transform group-open:rotate-90" />
        Reasoning
        {pending ? (
          <span className="ml-auto text-[10px] font-normal tracking-normal text-muted-foreground/70">
            live
          </span>
        ) : null}
      </summary>
      <div className="max-h-64 overflow-y-auto border-t border-border/60 px-2.5 py-2 text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground">
        {text}
      </div>
    </details>
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

function splitPromptPct(label: string): { text: string; pct: string } {
  const match = label.match(/^(Processing Prompt…?)\s+([\d.]+%)\s*$/i);
  if (!match) return { text: label, pct: "" };
  return { text: match[1], pct: match[2] };
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
  const { text, pct } = splitPromptPct(label);
  const showModel =
    Boolean(model) && /^(Streaming|Processing Prompt|Reasoning)/i.test(text);
  return (
    <p className="text-sm text-muted-foreground">
      <span className="shimmer-text">{text}</span>
      {showModel ? (
        <span className="ml-1.5 font-mono text-[10px] font-normal tracking-tight text-muted-foreground/40">
          {pct ? `${pct} ` : ""}
          [{model}]
          {route ? ` [${route}]` : ""}
          {context ? ` ${fmtContext(context)}` : ""}
          <RecallMark names={recalled} />
        </span>
      ) : (
        <>
          {pct ? (
            <span className="ml-1.5 font-mono text-[10px] font-normal tracking-tight text-muted-foreground/40">
              {pct}
            </span>
          ) : null}
          <RecallMark names={recalled} />
        </>
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