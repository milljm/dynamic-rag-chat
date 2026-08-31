import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  BookOpen,
  Bot,
  Download,
  FileText,
  FolderCode,
  GitBranch,
  Lock,
  PanelLeftClose,
  Play,
  Plus,
  RotateCcw,
  Trash2,
  Undo2,
  Wrench,
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
import { previewCharsForWidth } from "@/lib/chat/preview-chars";
import {
  deleteDocument,
  deleteProjectFile,
  deleteTool,
  getProjectFile,
  getToolFile,
  addProjectDir,
  listDocuments,
  listProjects,
  removeProject,
  runProjectFile,
  runTool,
  selectProject,
  usesChatPy,
} from "@/lib/chat/remote";
import type { GoldDocument, ProjectFile, ProjectRecord, ProjectSnapshot } from "@/lib/chat/remote";
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
  const asideRef = useRef<HTMLElement>(null);
  const [barWidth, setBarWidth] = useState(350);
  const previewChars = previewCharsForWidth(barWidth);

  useEffect(() => {
    const el = asideRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setBarWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

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
      ref={asideRef}
      className={cn(
        "flex h-full min-h-0 min-w-0 w-full flex-col overflow-hidden bg-card paper text-card-foreground",
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

      <ScrollArea className="min-h-0 min-w-0 flex-1">
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
          <ul className="min-w-0 space-y-1">
            {list.map((branch) => (
              <BranchRow
                key={branch.id}
                branch={branch}
                previewChars={previewChars}
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
        <ProjectFiles
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
  previewChars,
  onSwitch,
  onDelete,
}: {
  branch: Branch;
  active: boolean;
  previewChars: number;
  onSwitch: () => void;
  onDelete: () => void;
}) {
  const turns = turnCount(branch.messages);
  const mode = modeOf(branch);
  const preview = lastAssistantPreview(branch, previewChars);
  const locked = isLockedBranch(branch.id);
  const canDelete = !active && !locked;

  return (
    <li
      className={cn(
        "flex min-w-0 items-stretch gap-0.5 overflow-hidden rounded-md transition-[background-color,box-shadow] duration-150",
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
            <span className="mt-1 block min-w-0 max-w-full truncate text-xs text-muted-foreground/80">
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

function lastAssistantPreview(branch: Branch, maxChars: number): string {
  const last = [...branch.messages]
    .reverse()
    .find((m) => m.role === "assistant" && m.content);
  if (!last) return "";
  const flat = last.content.replace(/\s+/g, " ").trim();
  return flat.length > maxChars ? `${flat.slice(0, maxChars)}…` : flat;
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
      <div className="flex min-w-0 gap-2">
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="alt-ending or testing@5"
          aria-label="Branch name"
          className="min-w-0"
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
        Files available for download from the last {ARTIFACT_TTL} turns.
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

function runnable(path: string): boolean {
  return /\.(py|js|mjs)$/i.test(path);
}

const SHOW_PROJECT_FILES = 80;

const EMPTY_PROJECTS: ProjectSnapshot = {
  active: "workspace",
  projects: [],
  files: [],
  tools: [],
};

function ProjectFiles({
  currentId,
  messageN,
  streaming,
}: {
  currentId: string;
  messageN: number;
  streaming: boolean;
}) {
  const [files, setFiles] = useState<ProjectFile[]>([]);
  const [tools, setTools] = useState<ProjectFile[]>([]);
  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [active, setActive] = useState("workspace");
  const [truncated, setTruncated] = useState(false);
  const [output, setOutput] = useState("");
  const [dirPath, setDirPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [filesOpen, setFilesOpen] = useState(true);

  const apply = (snap: ProjectSnapshot) => {
    setFiles(snap.files);
    setTools(snap.tools);
    setProjects(snap.projects);
    setActive(snap.active);
    setTruncated(Boolean(snap.truncated));
    setFilesOpen(true);
  };

  const refresh = () => {
    if (!usesChatPy()) {
      apply(EMPTY_PROJECTS);
      return;
    }
    listProjects()
      .then(apply)
      .catch(() => apply(EMPTY_PROJECTS));
  };
  useEffect(() => {
    if (streaming) return;
    refresh();
  }, [currentId, messageN, streaming]);
  useEffect(() => {
    const onProj = () => refresh();
    window.addEventListener("spur-project", onProj);
    return () => window.removeEventListener("spur-project", onProj);
  }, []);

  if (!usesChatPy()) return null;

  const shown = files.slice(0, SHOW_PROJECT_FILES);
  const locked = streaming || busy;
  const others = projects.filter((project) => project.kind !== "scratch");
  const visible =
    others.length === 0
      ? projects
      : projects.filter((project) => project.kind !== "scratch" || project.id === active);

  return (
    <SidebarSection
      id="projects"
      title="Projects"
      defaultOpen={false}
      badge={visible.length || undefined}
    >
      <div className="space-y-2">
        {visible.length > 0 ? (
          <ul className="space-y-1">
            {visible.map((project) => {
              const isActive = project.id === active;
              const canRemove = project.kind !== "scratch";
              return (
                <li
                  key={project.id}
                  className={cn(
                    "min-w-0 rounded-sm",
                    isActive ? "bg-accent" : "hover:bg-accent/70",
                  )}
                >
                  <div className="flex min-w-0 items-center gap-0.5">
                    <button
                      type="button"
                      disabled={locked}
                      onClick={async () => {
                        if (isActive) {
                          setFilesOpen((open) => !open);
                          return;
                        }
                        setBusy(true);
                        const result = await selectProject(project.id);
                        setBusy(false);
                        if (!result.ok) {
                          toast.error(result.error || `Could not select ${project.name}`);
                          return;
                        }
                        setOutput("");
                        setFilesOpen(true);
                        refresh();
                      }}
                      aria-current={isActive ? "true" : undefined}
                      aria-expanded={isActive ? filesOpen : undefined}
                      className="flex min-w-0 flex-1 items-center gap-1.5 px-2 py-1.5 text-left text-xs"
                      title={project.path}
                    >
                      <span
                        className={cn(
                          "size-1.5 shrink-0 rounded-full",
                          isActive ? "bg-primary" : "bg-muted-foreground/40",
                        )}
                      />
                      {project.git ? (
                        <GitBranch className="size-3 shrink-0 text-muted-foreground" />
                      ) : (
                        <FolderCode className="size-3 shrink-0 text-muted-foreground" />
                      )}
                      <span className="min-w-0 flex-1 truncate font-medium">
                        {project.name}
                      </span>
                      {isActive && !filesOpen && files.length > 0 ? (
                        <span className="shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground/70">
                          {files.length}
                        </span>
                      ) : null}
                      {project.git ? null : (
                        <span className="shrink-0 text-[10px] text-muted-foreground/70">
                          no git
                        </span>
                      )}
                    </button>
                    {canRemove ? (
                      <button
                        type="button"
                        disabled={locked}
                        className="relative mr-0.5 size-8 shrink-0 text-muted-foreground after:absolute after:left-1/2 after:top-1/2 after:size-10 after:-translate-x-1/2 after:-translate-y-1/2 hover:text-destructive"
                        aria-label={`Remove ${project.name}`}
                        onClick={async () => {
                          if (
                            !window.confirm(
                              `Unregister ${project.name}? Files on disk are not deleted.`,
                            )
                          ) {
                            return;
                          }
                          setBusy(true);
                          const result = await removeProject(project.id);
                          setBusy(false);
                          if (!result.ok) {
                            toast.error(result.error || `Could not remove ${project.name}`);
                            return;
                          }
                          toast.success(`Removed ${project.name}`);
                          setOutput("");
                          refresh();
                        }}
                      >
                        <X className="size-3.5" />
                      </button>
                    ) : null}
                  </div>
                  {isActive && filesOpen ? (
                    <div className="space-y-1 px-2 pb-2">
                      {files.length === 0 ? (
                        <p className="text-[10px] leading-relaxed text-muted-foreground/70">
                          No files yet.
                        </p>
                      ) : (
                        <ul className="space-y-1">
                          {shown.map((file) => (
                            <li
                              key={file.path}
                              className="flex items-center gap-1 rounded-sm bg-secondary px-2 py-1.5 text-xs"
                            >
                              <FolderCode className="size-3 shrink-0 text-muted-foreground" />
                              <span
                                className="min-w-0 flex-1 truncate font-mono"
                                title={file.path}
                              >
                                {file.path}
                              </span>
                              <span className="font-mono text-[10px] tabular-nums text-muted-foreground/50">
                                {fmtChars(file.chars)}
                              </span>
                              {runnable(file.path) ? (
                                <button
                                  type="button"
                                  className="relative size-8 text-muted-foreground after:absolute after:left-1/2 after:top-1/2 after:size-10 after:-translate-x-1/2 after:-translate-y-1/2 hover:text-foreground"
                                  aria-label={`Run ${file.path}`}
                                  onClick={async () => {
                                    const result = await runProjectFile(file.path);
                                    const body = (
                                      result.stdout ||
                                      result.stderr ||
                                      result.error ||
                                      `exit ${result.code}`
                                    ).trim();
                                    setOutput(`$ ${result.cmd || file.path}\n${body}`);
                                    if (result.ok) toast.success(`Ran ${file.path}`);
                                    else
                                      toast.error(
                                        result.stderr || result.error || "Run failed",
                                      );
                                  }}
                                >
                                  <Play className="size-3.5" />
                                </button>
                              ) : null}
                              <button
                                type="button"
                                className="relative size-8 text-muted-foreground after:absolute after:left-1/2 after:top-1/2 after:size-10 after:-translate-x-1/2 after:-translate-y-1/2 hover:text-foreground"
                                aria-label={`Download ${file.path}`}
                                onClick={async () => {
                                  const text = await getProjectFile(file.path);
                                  if (text == null) {
                                    toast.error(`Could not read ${file.path}`);
                                    return;
                                  }
                                  downloadTextFile(
                                    file.path.split("/").pop() || file.path,
                                    text,
                                  );
                                }}
                              >
                                <Download className="size-3.5" />
                              </button>
                              <button
                                type="button"
                                className="relative size-8 text-muted-foreground after:absolute after:left-1/2 after:top-1/2 after:size-10 after:-translate-x-1/2 after:-translate-y-1/2 hover:text-destructive"
                                aria-label={`Delete ${file.path}`}
                                onClick={async () => {
                                  if (
                                    !window.confirm(
                                      `Remove ${file.path} from the project?`,
                                    )
                                  ) {
                                    return;
                                  }
                                  const result = await deleteProjectFile(file.path);
                                  if (!result.ok) {
                                    toast.error(
                                      result.error || `Could not delete ${file.path}`,
                                    );
                                    return;
                                  }
                                  toast.success(`Deleted ${file.path}`);
                                  refresh();
                                }}
                              >
                                <Trash2 className="size-3.5" />
                              </button>
                            </li>
                          ))}
                        </ul>
                      )}
                      {files.length > SHOW_PROJECT_FILES ? (
                        <p className="text-[10px] text-muted-foreground/70">
                          Showing {SHOW_PROJECT_FILES} of {files.length}
                          {truncated ? "+" : ""}
                        </p>
                      ) : null}
                    </div>
                  ) : null}
                </li>
              );
            })}
          </ul>
        ) : null}

        <form
          className="space-y-1.5"
          onSubmit={async (e) => {
            e.preventDefault();
            const raw = dirPath.trim();
            if (!raw || locked) return;
            setBusy(true);
            const result = await addProjectDir(raw);
            setBusy(false);
            if (!result.ok) {
              toast.error(result.error || "Could not add that directory");
              return;
            }
            toast.success("Added project dir");
            setDirPath("");
            setOutput("");
            refresh();
          }}
        >
          <label className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <FolderCode className="size-3.5" />
            Add project dir
          </label>
          <div className="flex min-w-0 gap-2">
            <Input
              value={dirPath}
              onChange={(e) => setDirPath(e.target.value)}
              placeholder="~/src/my-repo"
              aria-label="Project directory"
              className="min-w-0 font-mono text-xs"
              disabled={locked}
            />
            <Button
              type="submit"
              size="icon"
              aria-label="Add project dir"
              disabled={locked || !dirPath.trim()}
            >
              <Plus />
            </Button>
          </div>
          <p className="text-[10px] leading-relaxed text-muted-foreground/70">
            Absolute path on this machine. Files stay in place.
          </p>
        </form>

        <details key={`tools-${tools.length}`} className="rounded-sm" defaultOpen={tools.length > 0}>
          <summary className="cursor-pointer text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
            Tools
            {tools.length ? (
              <span className="ml-2 font-mono text-[10px] font-normal normal-case tabular-nums tracking-normal">
                {tools.length}
              </span>
            ) : null}
          </summary>
          <div className="mt-2 space-y-1">
            {tools.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                Coding writes helpers here (
                <code className="font-mono">{"tool:name.py"}</code>
                ), then runs them. They stick around, outside the project.
              </p>
            ) : (
              <ul className="space-y-1">
                {tools.map((file) => (
                  <li
                    key={file.path}
                    className="flex items-center gap-1 rounded-sm bg-secondary px-2 py-1.5 text-xs"
                  >
                    <Wrench className="size-3 shrink-0 text-muted-foreground" />
                    <span className="min-w-0 flex-1 truncate font-mono" title={file.path}>
                      {file.path}
                    </span>
                    {runnable(file.path) ? (
                      <button
                        type="button"
                        className="relative size-8 text-muted-foreground after:absolute after:left-1/2 after:top-1/2 after:size-10 after:-translate-x-1/2 after:-translate-y-1/2 hover:text-foreground"
                        aria-label={`Run ${file.path}`}
                        onClick={async () => {
                          const result = await runTool(file.path);
                          const body = (
                            result.stdout ||
                            result.stderr ||
                            result.error ||
                            `exit ${result.code}`
                          ).trim();
                          setOutput(`$ ${result.cmd || file.path}\n${body}`);
                          if (result.ok) toast.success(`Ran ${file.path}`);
                          else toast.error(result.stderr || result.error || "Run failed");
                          refresh();
                        }}
                      >
                        <Play className="size-3.5" />
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className="relative size-8 text-muted-foreground after:absolute after:left-1/2 after:top-1/2 after:size-10 after:-translate-x-1/2 after:-translate-y-1/2 hover:text-foreground"
                      aria-label={`Download ${file.path}`}
                      onClick={async () => {
                        const text = await getToolFile(file.path);
                        if (text == null) {
                          toast.error(`Could not read ${file.path}`);
                          return;
                        }
                        downloadTextFile(file.path.split("/").pop() || file.path, text);
                      }}
                    >
                      <Download className="size-3.5" />
                    </button>
                    <button
                      type="button"
                      className="relative size-8 text-muted-foreground after:absolute after:left-1/2 after:top-1/2 after:size-10 after:-translate-x-1/2 after:-translate-y-1/2 hover:text-destructive"
                      aria-label={`Delete ${file.path}`}
                      onClick={async () => {
                        if (!window.confirm(`Remove tool ${file.path}?`)) return;
                        const result = await deleteTool(file.path);
                        if (!result.ok) {
                          toast.error(result.error || `Could not delete ${file.path}`);
                          return;
                        }
                        toast.success(`Deleted ${file.path}`);
                        refresh();
                      }}
                    >
                      <Trash2 className="size-3.5" />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </details>
        {output ? (
          <pre className="max-h-32 overflow-auto rounded-sm bg-background px-2 py-1.5 font-mono text-[10px] leading-relaxed text-muted-foreground">
            {output}
          </pre>
        ) : null}
        <p className="text-[10px] leading-relaxed text-muted-foreground/70">
          Click a project to browse its files. New apps get their own name —
          not workspace. Imported dirs that are not git: the model asks first.
        </p>
      </div>
    </SidebarSection>
  );
}
