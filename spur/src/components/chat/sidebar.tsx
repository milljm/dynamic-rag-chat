import { useEffect, useState, type ReactNode } from "react";
import {
  BookOpen,
  Bot,
  FileText,
  GitBranch,
  Lock,
  PanelLeftClose,
  Plus,
  Download,
  RotateCcw,
  Trash2,
  Undo2,
  X,
} from "lucide-react";
import { toast } from "sonner";
import {
  ARTIFACT_TTL,
  artifactsFromMessages,
  downloadTextFile,
  type LivedArtifact,
} from "@/lib/chat/artifacts";
import { isLockedBranch, modeOf, turnCount } from "@/lib/chat/branch-mode";
import { SLASH_HELP } from "@/lib/chat/commands";
import { listDocuments, deleteDocument, usesChatPy } from "@/lib/chat/remote";
import type { GoldDocument } from "@/lib/chat/remote";
import { useChatStore } from "@/lib/chat/store";
import type { Branch } from "@/lib/chat/types";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { ModeToggle } from "./mode-toggle";

const SECTION_KEY = "spur-sec-";

function readSectionOpen(id: string, fallback: boolean): boolean {
  if (typeof window === "undefined") return fallback;
  try {
    const v = window.localStorage.getItem(SECTION_KEY + id);
    if (v === "1") return true;
    if (v === "0") return false;
  } catch {
    /* ignore */
  }
  return fallback;
}

function writeSectionOpen(id: string, open: boolean) {
  try {
    window.localStorage.setItem(SECTION_KEY + id, open ? "1" : "0");
  } catch {
    /* ignore */
  }
}

function SidebarSection({
  id,
  title,
  defaultOpen,
  badge,
  children,
  className,
  bodyClassName,
}: {
  id: string;
  title: string;
  defaultOpen: boolean;
  badge?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  const [open, setOpen] = useState(() => readSectionOpen(id, defaultOpen));
  return (
    <details
      open={open}
      className={cn("border-b border-border last:border-b-0", className)}
      onToggle={(e) => {
        const next = (e.currentTarget as HTMLDetailsElement).open;
        if (next === open) return;
        setOpen(next);
        writeSectionOpen(id, next);
      }}
    >
      <summary className="cursor-pointer px-4 py-3 text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
        {title}
        {badge != null ? (
          <span className="ml-2 font-mono text-xs font-normal normal-case tabular-nums tracking-normal">
            {badge}
          </span>
        ) : null}
      </summary>
      <div className={cn("px-4 pb-3", bodyClassName)}>{children}</div>
    </details>
  );
}

export function Sidebar({
  className,
  onNavigate,
  onCollapse,
  streaming = false,
}: {
  className?: string;
  onNavigate?: () => void;
  onCollapse?: () => void;
  streaming?: boolean;
}) {
  const currentId = useChatStore((s) => s.currentId);
  const branches = useChatStore((s) => s.branches);
  const switchBranch = useChatStore((s) => s.switchBranch);
  const setMode = useChatStore((s) => s.setMode);
  const createBranch = useChatStore((s) => s.createBranch);
  const deleteBranch = useChatStore((s) => s.deleteBranch);
  const current = branches[currentId];
  const files = artifactsFromMessages(current?.messages ?? []);

  const list = Object.values(branches).sort((a, b) => {
    if (a.id === currentId) return -1;
    if (b.id === currentId) return 1;
    if (isLockedBranch(a.id) !== isLockedBranch(b.id)) {
      return isLockedBranch(a.id) ? 1 : -1;
    }
    return b.updatedAt - a.updatedAt;
  });

  return (
    <aside
      className={cn(
        "flex h-full min-h-0 w-full flex-col bg-card paper text-card-foreground",
        className,
      )}
    >
      <div className="flex items-start justify-between gap-2 px-5 pb-4 pt-6">
        <div>
          <p className="font-display text-2xl italic leading-none tracking-tight">
            Spur
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {usesChatPy() ? "Front-end for chat.py" : "Branched RAG chat"}
          </p>
        </div>
        {onCollapse && (
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="mt-0.5 hidden md:inline-flex"
            aria-label="Collapse sidebar"
            onClick={onCollapse}
          >
            <PanelLeftClose />
          </Button>
        )}
      </div>

      <Separator />

      <ScrollArea className="min-h-0 flex-1">
        <SidebarSection id="mode" title="Mode" defaultOpen>
          {current && (
            <ModeToggle
              branchId={current.id}
              mode={current.mode}
              onChange={(mode) => {
                const ok = setMode(mode);
                if (!ok) toast.message("Mode is locked on this branch.");
              }}
            />
          )}
        </SidebarSection>

        <SidebarSection
          id="branches"
          title="Branches"
          defaultOpen
          badge={list.length}
          bodyClassName="px-2"
        >
          <ul className="space-y-1">
            {list.map((branch) => (
              <BranchRow
                key={branch.id}
                branch={branch}
                active={branch.id === currentId}
                onSwitch={() => {
                  const ok = switchBranch(branch.id);
                  if (!ok) {
                    toast.error(`Could not switch to ${branch.name}.`);
                    return;
                  }
                  onNavigate?.();
                }}
                onDelete={() => {
                  const result = deleteBranch(branch.id);
                  if (!result.ok) {
                    toast.error(result.error);
                    return;
                  }
                  toast.success(`Deleted '${branch.name}'.`);
                }}
              />
            ))}
          </ul>
          <div className="px-2">
            <CreateBranchForm
              onCreate={(raw) => {
                const result = createBranch(raw);
                if (!result.ok) {
                  toast.error(result.error);
                  return false;
                }
                toast.success(`Branched to ${result.id}`);
                onNavigate?.();
                return true;
              }}
            />
          </div>
        </SidebarSection>

        <HistoryTools />
        <SlashHelp />
        <GoldDocuments
          currentId={currentId}
          messageN={current?.messages.length ?? 0}
          streaming={streaming}
        />

        <SidebarSection
          id="files"
          title="Downloadable Files"
          defaultOpen={false}
          badge={files.length || undefined}
        >
          <GeneratedFiles files={files} />
        </SidebarSection>
      </ScrollArea>
    </aside>
  );
}

function BranchRow({
  branch,
  active,
  onSwitch,
  onDelete,
}: {
  branch: Branch;
  active: boolean;
  onSwitch: () => void;
  onDelete: () => void;
}) {
  const turns = turnCount(branch.messages);
  const mode = modeOf(branch);
  const preview = lastAssistantPreview(branch);
  const locked = isLockedBranch(branch.id);
  const canDelete = !active && !locked;

  return (
    <li
      className={cn(
        "flex items-stretch gap-0.5 rounded-md transition-[background-color,box-shadow] duration-150",
        active ? "bg-accent shadow-[var(--shadow-border)]" : "hover:bg-accent/70",
      )}
    >
      <button
        type="button"
        disabled={active}
        onClick={onSwitch}
        aria-current={active ? "page" : undefined}
        className="flex min-w-0 flex-1 items-start gap-2 px-3 py-2.5 text-left"
      >
        <span
          className={cn(
            "mt-1.5 size-1.5 shrink-0 rounded-full",
            active ? "bg-primary" : "bg-muted-foreground/40",
          )}
        />
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5">
            <span className="truncate text-sm font-medium">{branch.name}</span>
            {locked && (
              <Lock className="size-3 shrink-0 text-muted-foreground" />
            )}
          </span>
          <span className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
            {mode === "assistant" ? (
              <Bot className="size-3" />
            ) : (
              <BookOpen className="size-3" />
            )}
            <span className="capitalize">{mode}</span>
            <span className="font-mono tabular-nums">· {turns} turns</span>
          </span>
          {preview && (
            <span className="mt-1 block truncate text-xs text-muted-foreground/80">
              {preview}
            </span>
          )}
        </span>
      </button>
      {canDelete && (
        <button
          type="button"
          aria-label={`Delete ${branch.name}`}
          onClick={onDelete}
          className="relative mt-1 mr-1 flex size-8 shrink-0 items-center justify-center rounded-sm text-muted-foreground transition-colors hover:bg-destructive/15 hover:text-destructive"
        >
          <X className="size-3.5" />
        </button>
      )}
    </li>
  );
}

function lastAssistantPreview(branch: Branch): string {
  const last = [...branch.messages]
    .reverse()
    .find((m) => m.role === "assistant" && m.content);
  if (!last) return "";
  const flat = last.content.replace(/\s+/g, " ").trim();
  return flat.length > 42 ? `${flat.slice(0, 42)}…` : flat;
}

function CreateBranchForm({
  onCreate,
}: {
  onCreate: (raw: string) => boolean;
}) {
  const [name, setName] = useState("");

  return (
    <form
      className="space-y-2 border-t border-border py-3"
      onSubmit={(e) => {
        e.preventDefault();
        if (!name.trim()) return;
        if (onCreate(name)) setName("");
      }}
    >
      <label className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        <GitBranch className="size-3.5" />
        New branch
      </label>
      <div className="flex gap-2">
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="alt-ending or testing@5"
          aria-label="Branch name"
        />
        <Button type="submit" size="icon" aria-label="Create branch">
          <Plus />
        </Button>
      </div>
      <p className="text-[10px] leading-relaxed text-muted-foreground/70">
        Append @N to fork from the first N turns.
      </p>
    </form>
  );
}

function HistoryTools() {
  const [n, setN] = useState("");
  const currentId = useChatStore((s) => s.currentId);
  const turns = useChatStore(
    (s) => turnCount(s.branches[s.currentId]?.messages ?? []),
  );
  const deleteLastTurn = useChatStore((s) => s.deleteLastTurn);
  const resetBranch = useChatStore((s) => s.resetBranch);
  const rewindTo = useChatStore((s) => s.rewindTo);

  return (
    <SidebarSection id="history" title="History tools" defaultOpen={false}>
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            className="h-10"
            disabled={turns === 0}
            onClick={() => {
              const result = deleteLastTurn();
              if (result.ok) toast.success("Deleted last turn.");
              else toast.message(result.error);
            }}
          >
            <Undo2 className="size-3.5" />
            Delete last
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            className="h-10"
            onClick={() => {
              const result = resetBranch();
              if (result.ok) toast.success(`Reset '${currentId}'.`);
              else toast.error(result.error);
            }}
          >
            <Trash2 className="size-3.5" />
            Reset
          </Button>
        </div>
        <div className="flex gap-2">
          <Input
            inputMode="numeric"
            value={n}
            onChange={(e) => setN(e.target.value)}
            placeholder={turns ? `Rewind 1–${turns}` : "No turns"}
            aria-label="Rewind to turn"
            disabled={turns === 0}
          />
          <Button
            type="button"
            variant="secondary"
            size="icon"
            aria-label="Rewind"
            disabled={turns === 0}
            onClick={() => {
              const parsed = Number.parseInt(n, 10);
              const result = rewindTo(parsed);
              if (result.ok) {
                toast.success(`Rewound to turn ${parsed}.`);
                setN("");
              } else toast.error(result.error);
            }}
          >
            <RotateCcw />
          </Button>
        </div>
      </div>
    </SidebarSection>
  );
}

function SlashHelp() {
  return (
    <SidebarSection id="slash" title="Slash commands" defaultOpen={false}>
      <ul className="space-y-1.5">
        {SLASH_HELP.map((row) => (
          <li key={row.cmd} className="space-y-0.5 text-xs">
            <code className="font-mono text-foreground/80">{row.cmd}</code>
            <p className="text-[10px] leading-relaxed text-muted-foreground/70">{row.hint}</p>
          </li>
        ))}
      </ul>
    </SidebarSection>
  );
}

function ttlBadgeStyle(remaining: number): React.CSSProperties {
  if (remaining <= 1) {
    return {
      color: "var(--spur-ttl-warn)",
      background: "var(--spur-ttl-warn-bg)",
      boxShadow: "var(--spur-ttl-warn-shadow)",
    };
  }
  return {
    color: "var(--spur-turn)",
    background: "var(--spur-turn-bg)",
    boxShadow: "var(--spur-turn-shadow)",
  };
}

function GeneratedFiles({
  files,
}: {
  files: LivedArtifact[];
}) {
  if (files.length === 0) {
    return (
      <p className="text-[10px] leading-relaxed text-muted-foreground/70">
        Files the AI creates appear here for {ARTIFACT_TTL} turns after they
        show up in history.
      </p>
    );
  }
  return (
    <div className="space-y-2">
      <ul className="space-y-1">
        {files.map((art) => (
          <li
            key={art.file}
            className="flex items-center gap-2 rounded-sm bg-secondary px-2 py-1.5 text-xs"
          >
            <span
              className="shrink-0 rounded-md px-1.5 py-px font-mono text-[10px] tabular-nums tracking-tight"
              style={ttlBadgeStyle(art.remaining)}
              title={`${art.remaining} turn${art.remaining === 1 ? "" : "s"} left`}
            >
              {art.remaining}
            </span>
            <span className="min-w-0 flex-1 truncate font-mono">{art.file}</span>
            <button
              type="button"
              className="relative size-8 text-muted-foreground after:absolute after:left-1/2 after:top-1/2 after:size-10 after:-translate-x-1/2 after:-translate-y-1/2 hover:text-foreground"
              aria-label={`Download ${art.file}`}
              onClick={() => downloadTextFile(art.file, art.text)}
            >
              <Download className="size-3.5" />
            </button>
          </li>
        ))}
      </ul>
      <p className="text-[10px] leading-relaxed text-muted-foreground/70">
        Only files from the last {ARTIFACT_TTL} turns. The number is turns left
        before it drops off.
      </p>
    </div>
  );
}

function fmtChars(n: number): string {
  if (n < 1000) return `${n}`;
  const k = n / 1000;
  return k >= 10 ? `${k.toFixed(0)}k` : `${k.toFixed(1)}k`;
}

function GoldDocuments({
  currentId,
  messageN,
  streaming,
}: {
  currentId: string;
  messageN: number;
  streaming: boolean;
}) {
  const [docs, setDocs] = useState<GoldDocument[]>([]);
  const refresh = () => {
    if (!usesChatPy()) {
      setDocs([]);
      return;
    }
    listDocuments()
      .then(setDocs)
      .catch(() => setDocs([]));
  };
  useEffect(() => {
    if (streaming) return;
    refresh();
  }, [currentId, messageN, streaming]);
  useEffect(() => {
    const onDocs = () => refresh();
    window.addEventListener("spur-documents", onDocs);
    return () => window.removeEventListener("spur-documents", onDocs);
  }, []);

  if (!usesChatPy()) return null;

  return (
    <SidebarSection
      id="documents"
      title="Documents"
      defaultOpen={false}
      badge={docs.length || undefined}
    >
      <div className="space-y-2">
      {docs.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          Attach a file with the paperclip. After the turn it lives here.
          Mention the name to load it again.
        </p>
      ) : (
        <ul className="space-y-1">
          {docs.map((doc) => (
            <li
              key={doc.name}
              className="flex items-center gap-2 rounded-sm bg-secondary px-2 py-1.5 text-xs"
            >
              <FileText className="size-3 shrink-0 text-muted-foreground" />
              <span className="min-w-0 flex-1 truncate" title={doc.name}>
                {doc.name}
              </span>
              <span className="font-mono text-[10px] tabular-nums text-muted-foreground/50">
                {fmtChars(doc.chars)}
              </span>
              <button
                type="button"
                className="relative size-8 text-muted-foreground after:absolute after:left-1/2 after:top-1/2 after:size-10 after:-translate-x-1/2 after:-translate-y-1/2 hover:text-destructive"
                aria-label={`Delete ${doc.name}`}
                onClick={async () => {
                  if (!window.confirm(`Remove ${doc.name} from gold?`)) return;
                  const result = await deleteDocument(doc.name);
                  if (!result.ok) {
                    toast.error(result.error || `Could not delete ${doc.name}`);
                    return;
                  }
                  toast.success(`Deleted ${doc.name}`);
                  refresh();
                }}
              >
                <Trash2 className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
      <p className="text-[10px] leading-relaxed text-muted-foreground/70">
        Files available for recall by the AI.
      </p>
      </div>
    </SidebarSection>
  );
}
