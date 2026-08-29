import { useRef, useState } from "react";
import { ArrowUp, Paperclip, RefreshCw, Square, X } from "lucide-react";
import { toast } from "sonner";
import { lastUserMessage, modeOf, turnCount } from "@/lib/chat/branch-mode";
import { useChatStore } from "@/lib/chat/store";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

export function Composer({
  streaming,
  onSend,
  onStop,
  onRegenerate,
}: {
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onRegenerate: () => void;
}) {
  const [value, setValue] = useState("");
  const [dragDepth, setDragDepth] = useState(0);
  const pending = useChatStore((s) => s.pendingAttachments);
  const addFiles = useChatStore((s) => s.addFiles);
  const removePending = useChatStore((s) => s.removePending);
  const forceAgent = useChatStore((s) => s.forceAgent);
  const setForceAgent = useChatStore((s) => s.setForceAgent);
  const currentId = useChatStore((s) => s.currentId);
  const branch = useChatStore((s) => s.branches[s.currentId]);
  const ref = useRef<HTMLTextAreaElement>(null);

  const mode = branch ? modeOf(branch) : "story";
  const turns = turnCount(branch?.messages ?? []);
  const lastUser = lastUserMessage(branch?.messages ?? []);
  const canRegen =
    !streaming &&
    Boolean(lastUser && (lastUser.content || lastUser.attachments?.length));
  const canSend =
    !streaming && (value.trim().length > 0 || pending.length > 0);
  const agentOn = mode === "assistant" && forceAgent;
  const dragging = dragDepth > 0;

  function submit() {
    if (!canSend) return;
    const text = value;
    setValue("");
    onSend(text);
    ref.current?.focus();
  }

  async function ingest(fileList: File[]) {
    if (!fileList.length) return;
    const { added, skipped, unreadable } = await addFiles(fileList);
    if (skipped) {
      toast.message(
        `Skipped ${skipped} unsupported file${skipped === 1 ? "" : "s"}.`,
      );
    }
    if (unreadable) {
      toast.error(
        `Couldn't read ${unreadable} file${unreadable === 1 ? "" : "s"}.`,
      );
    }
    if (added) {
      toast.success(`Attached ${added} file${added === 1 ? "" : "s"}.`);
    }
  }

  return (
    <div className="border-t border-border bg-background paper px-4 py-3 md:px-8">
      <div className="mx-auto max-w-3xl">
        <div
          className={cn(
            "relative rounded-lg bg-card p-2 shadow-[var(--shadow-border)]",
            dragging && "shadow-[var(--shadow-border-hover)]",
          )}
          onDragEnter={(e) => {
            if (![...e.dataTransfer.types].includes("Files")) return;
            e.preventDefault();
            setDragDepth((n) => n + 1);
          }}
          onDragLeave={() => setDragDepth((n) => Math.max(0, n - 1))}
          onDragOver={(e) => {
            if (![...e.dataTransfer.types].includes("Files")) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
          }}
          onDrop={(e) => {
            e.preventDefault();
            setDragDepth(0);
            void ingest([...e.dataTransfer.files]);
          }}
        >
          {dragging && (
            <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center rounded-lg bg-background/80">
              <p className="text-sm text-foreground">Drop files to attach</p>
            </div>
          )}
          {pending.length > 0 && (
            <ul className="mb-1 flex flex-wrap gap-1 px-2 pt-1">
              {pending.map((att) => (
                <li
                  key={att.id}
                  className="flex items-center gap-1.5 rounded-sm bg-secondary px-2 py-1 text-xs"
                >
                  {att.kind === "image" && att.dataUrl ? (
                    <img
                      src={att.dataUrl}
                      alt=""
                      className="size-5 rounded-[2px] object-cover"
                    />
                  ) : (
                    <Paperclip className="size-3 text-muted-foreground" />
                  )}
                  <span className="max-w-36 truncate">{att.name}</span>
                  <button
                    type="button"
                    className="text-muted-foreground hover:text-foreground"
                    aria-label={`Remove ${att.name}`}
                    onClick={() => removePending(att.id)}
                  >
                    <X className="size-3" />
                  </button>
                </li>
              ))}
            </ul>
          )}
          <Textarea
            ref={ref}
            value={value}
            rows={2}
            placeholder="Message · drop files here · commands start with \\"
            aria-label="Message"
            className="min-h-16 border-0 bg-transparent shadow-none focus-visible:ring-0"
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            onPaste={(e) => {
              const files = [...e.clipboardData.files];
              if (!files.length) return;
              void ingest(files);
              if (!e.clipboardData.getData("text")) e.preventDefault();
            }}
          />
          <div className="flex items-center justify-between gap-2 px-1 pb-1">
            <div className="flex items-center gap-1">
              <label className="relative inline-flex size-10 items-center justify-center rounded-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground">
                <Paperclip className="size-4" />
                <span className="sr-only">Attach files</span>
                <input
                  type="file"
                  multiple
                  className="sr-only"
                  accept=".png,.jpg,.jpeg,.gif,.webp,.txt,.md,.py,.json,.csv,.html,.js,.ts,.css"
                  onChange={(e) => {
                    const files = [...(e.target.files ?? [])];
                    e.target.value = "";
                    void ingest(files);
                  }}
                />
              </label>
              <button
                type="button"
                aria-pressed={agentOn}
                disabled={mode !== "assistant" || streaming}
                title={
                  mode === "assistant"
                    ? "Force live web search this turn"
                    : "Agent is locked to assistant mode"
                }
                onClick={() => setForceAgent(!forceAgent)}
                className={cn(
                  "h-8 rounded-sm px-2.5 text-xs font-medium transition-colors",
                  agentOn
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-foreground",
                  mode !== "assistant" && "opacity-40",
                )}
              >
                Agent
              </button>
            </div>
            <div className="flex items-center gap-1">
              {streaming ? (
                <Button
                  type="button"
                  variant="secondary"
                  size="icon"
                  aria-label="Stop generation"
                  onClick={onStop}
                >
                  <Square className="size-4" />
                </Button>
              ) : (
                <>
                  <Button
                    type="button"
                    variant="secondary"
                    size="icon"
                    aria-label="Regenerate last reply"
                    disabled={!canRegen}
                    onClick={onRegenerate}
                  >
                    <RefreshCw className="size-4" />
                  </Button>
                  <Button
                    type="button"
                    size="icon"
                    aria-label="Send"
                    disabled={!canSend}
                    onClick={submit}
                  >
                    <ArrowUp className="size-4" />
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
        <p className="mt-2 text-center font-mono text-xs tabular-nums text-muted-foreground">
          turn {turns} · {currentId} · {mode}
          {agentOn ? " · agent" : ""}
        </p>
      </div>
    </div>
  );
}
