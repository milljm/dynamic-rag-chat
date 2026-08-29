import type { Branch, ChatSnapshot, Message, Mode, RagChunk } from "./types";

export const RESERVED_NAMES = new Set([
  "current",
  "assistant",
  "story",
  "default",
  "assistant_mode",
  "branch_modes",
]);

export const LOCKED_MODE: Record<string, Mode> = {
  assistant: "assistant",
  story: "story",
};

export function canonicalMode(branchId: string): Mode | null {
  return LOCKED_MODE[branchId] ?? null;
}

export function isLockedBranch(branchId: string): boolean {
  return Object.prototype.hasOwnProperty.call(LOCKED_MODE, branchId);
}

/** Effective mode for a branch. Locked branches ignore stored mode. */
export function modeOf(branch: Pick<Branch, "id" | "mode">): Mode {
  return canonicalMode(branch.id) ?? branch.mode;
}

/**
 * Active conversation is ALWAYS `currentId`.
 * Mode must never select the branch — that was the Streamlit bug.
 */
export function activeBranch(state: ChatSnapshot): Branch | undefined {
  return state.branches[state.currentId];
}

export function turnCount(messages: Message[]): number {
  return messages.filter((m) => m.role === "user").length;
}

export function lastUserContent(messages: Message[]): string {
  return lastUserMessage(messages)?.content ?? "";
}

export function lastUserMessage(messages: Message[]): Message | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") return messages[i];
  }
  return undefined;
}

export function lastUserInputs(messages: Message[], n = 5): string[] {
  const out: string[] = [];
  for (let i = messages.length - 1; i >= 0 && out.length < n; i--) {
    const msg = messages[i];
    if (msg.role === "user" && msg.content) out.push(msg.content);
  }
  return out.reverse();
}

/** Flatten a branch the way Chat.stringify does for \\include. */
export function stringifyBranch(branch: Pick<Branch, "name" | "messages">): string {
  const lines = branch.messages.map((m) => {
    const role = m.role === "user" ? "USER" : "AI";
    return `${role}: ${m.content}`;
  });
  return `=== include:${branch.name} ===\n${lines.join("\n")}`;
}

export function parseBranchInput(raw: string):
  | { name: string; cutTurns: number | null }
  | { error: string } {
  const trimmed = raw.trim();
  if (!trimmed) return { error: "Branch name cannot be empty." };

  let name = trimmed;
  let cutTurns: number | null = null;

  const at = trimmed.lastIndexOf("@");
  if (at > 0) {
    name = trimmed.slice(0, at).trim();
    const suffix = trimmed.slice(at + 1).trim();
    const parsed = Number.parseInt(suffix, 10);
    if (!Number.isFinite(parsed) || parsed <= 0 || String(parsed) !== suffix) {
      return { error: "Invalid @N suffix — use a positive integer." };
    }
    cutTurns = parsed;
  }

  if (!name) return { error: "Branch name cannot be empty." };
  if (RESERVED_NAMES.has(name.toLowerCase())) {
    return { error: `'${name}' is a reserved branch name.` };
  }
  return { name, cutTurns };
}

export function slugify(name: string): string {
  const s = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return s || "branch";
}

export function uniqueBranchId(name: string, existing: Iterable<string>): string {
  const taken = new Set(existing);
  const base = slugify(name);
  if (!taken.has(base) && !RESERVED_NAMES.has(base)) return base;
  let i = 2;
  while (taken.has(`${base}-${i}`)) i += 1;
  return `${base}-${i}`;
}

export function takeTurns(messages: Message[], turns: number): Message[] {
  if (turns <= 0) return [];
  let seen = 0;
  const out: Message[] = [];
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    out.push(msg);
    if (msg.role === "user") {
      seen += 1;
      if (seen >= turns) {
        const next = messages[i + 1];
        if (next && next.role === "assistant") out.push(next);
        break;
      }
    }
  }
  return out;
}

export function rebuildRagFromMessages(messages: Message[]): RagChunk[] {
  const chunks: RagChunk[] = [];
  for (const msg of messages) {
    for (const att of msg.attachments ?? []) {
      if (att.kind === "text" && att.text) {
        chunks.push({
          id: `${att.id}-full`,
          source: att.name,
          text: att.text,
        });
      }
    }
  }
  return chunks;
}

export function applySwitch(state: ChatSnapshot, id: string): ChatSnapshot {
  if (!state.branches[id] || id === state.currentId) return state;
  return { ...state, currentId: id };
}

export function applySetMode(state: ChatSnapshot, mode: Mode): ChatSnapshot {
  const current = state.branches[state.currentId];
  if (!current || isLockedBranch(current.id)) return state;
  if (current.mode === mode) return state;
  return {
    ...state,
    branches: {
      ...state.branches,
      [current.id]: { ...current, mode, updatedAt: Date.now() },
    },
  };
}

export function applyCreateBranch(
  state: ChatSnapshot,
  rawName: string,
  cutTurns: number | null,
): { ok: true; state: ChatSnapshot; id: string } | { ok: false; error: string } {
  const name = rawName.trim();
  if (!name) return { ok: false, error: "Branch name cannot be empty." };
  if (RESERVED_NAMES.has(name.toLowerCase())) {
    return { ok: false, error: `'${name}' is a reserved branch name.` };
  }

  const src = state.branches[state.currentId];
  if (!src) return { ok: false, error: "No active branch." };

  const id = uniqueBranchId(name, Object.keys(state.branches));
  const now = Date.now();

  let messages: Message[];
  let rag: RagChunk[];
  if (cutTurns != null) {
    messages = takeTurns(src.messages, cutTurns).map(cloneMessage);
    rag = rebuildRagFromMessages(messages);
  } else {
    messages = src.messages.map(cloneMessage);
    rag = src.rag.map((c) => ({ ...c }));
  }

  const branch: Branch = {
    id,
    name,
    mode: modeOf(src),
    locked: false,
    messages,
    rag,
    createdAt: now,
    updatedAt: now,
  };

  return {
    ok: true,
    id,
    state: {
      ...state,
      currentId: id,
      branches: { ...state.branches, [id]: branch },
    },
  };
}

export function applyDeleteBranch(
  state: ChatSnapshot,
  id: string,
): { ok: true; state: ChatSnapshot } | { ok: false; error: string } {
  if (isLockedBranch(id) || RESERVED_NAMES.has(id)) {
    return { ok: false, error: `Cannot delete protected branch '${id}'.` };
  }
  if (state.currentId === id) {
    return {
      ok: false,
      error: "Cannot delete the branch you are on. Switch first, or reset it.",
    };
  }
  if (!state.branches[id]) {
    return { ok: false, error: `Unknown branch '${id}'.` };
  }
  const branches = { ...state.branches };
  delete branches[id];
  return { ok: true, state: { ...state, branches } };
}

export function applyResetBranch(
  state: ChatSnapshot,
): { ok: true; state: ChatSnapshot } | { ok: false; error: string } {
  const current = state.branches[state.currentId];
  if (!current) return { ok: false, error: "No active branch." };
  return {
    ok: true,
    state: {
      ...state,
      pendingAttachments: [],
      branches: {
        ...state.branches,
        [current.id]: {
          ...current,
          messages: [],
          rag: [],
          updatedAt: Date.now(),
        },
      },
    },
  };
}

export function applyDeleteLastTurn(
  state: ChatSnapshot,
): { ok: true; state: ChatSnapshot } | { ok: false; error: string } {
  const current = state.branches[state.currentId];
  if (!current) return { ok: false, error: "No active branch." };
  if (!current.messages.length) return { ok: false, error: "History empty." };
  const next = [...current.messages];
  if (next.at(-1)?.role === "assistant") next.pop();
  if (next.at(-1)?.role === "user") next.pop();
  return {
    ok: true,
    state: {
      ...state,
      branches: {
        ...state.branches,
        [current.id]: { ...current, messages: next, updatedAt: Date.now() },
      },
    },
  };
}

export function applyRewind(
  state: ChatSnapshot,
  n: number,
): { ok: true; state: ChatSnapshot } | { ok: false; error: string } {
  const current = state.branches[state.currentId];
  if (!current) return { ok: false, error: "No active branch." };
  const total = turnCount(current.messages);
  if (!Number.isInteger(n) || n < 1 || n > total) {
    return { ok: false, error: `Rewind needs 1 ≤ N ≤ ${total}.` };
  }
  return {
    ok: true,
    state: {
      ...state,
      branches: {
        ...state.branches,
        [current.id]: {
          ...current,
          messages: takeTurns(current.messages, n),
          updatedAt: Date.now(),
        },
      },
    },
  };
}

export function applyPopLastAssistant(state: ChatSnapshot): ChatSnapshot {
  const current = state.branches[state.currentId];
  if (!current || current.messages.at(-1)?.role !== "assistant") return state;
  return {
    ...state,
    branches: {
      ...state.branches,
      [current.id]: {
        ...current,
        messages: current.messages.slice(0, -1),
        updatedAt: Date.now(),
      },
    },
  };
}

export function applyAppendMessage(
  state: ChatSnapshot,
  message: Message,
  ragExtra: RagChunk[] = [],
  branchId = state.currentId,
): ChatSnapshot {
  const current = state.branches[branchId];
  if (!current) return state;
  return {
    ...state,
    pendingAttachments: branchId === state.currentId ? [] : state.pendingAttachments,
    branches: {
      ...state.branches,
      [current.id]: {
        ...current,
        messages: [...current.messages, message],
        rag: ragExtra.length ? [...current.rag, ...ragExtra] : current.rag,
        updatedAt: Date.now(),
      },
    },
  };
}

export function applyReplaceMessage(
  state: ChatSnapshot,
  messageId: string,
  patch: Partial<Message>,
  branchId = state.currentId,
): ChatSnapshot {
  const current = state.branches[branchId];
  if (!current) return state;
  return {
    ...state,
    branches: {
      ...state.branches,
      [current.id]: {
        ...current,
        messages: current.messages.map((m) =>
          m.id === messageId ? { ...m, ...patch } : m,
        ),
        updatedAt: Date.now(),
      },
    },
  };
}

export function applySetForceAgent(
  state: ChatSnapshot,
  enabled: boolean,
): ChatSnapshot {
  const current = state.branches[state.currentId];
  const allowed = current ? modeOf(current) === "assistant" : false;
  return { ...state, forceAgent: allowed && enabled };
}

function cloneMessage(m: Message): Message {
  return {
    ...m,
    attachments: m.attachments?.map((a) => ({ ...a })),
    metrics: m.metrics ? { ...m.metrics } : undefined,
    flags: m.flags ? { ...m.flags } : undefined,
  };
}

function lockedBranch(
  id: "story" | "assistant",
  now: number,
  extra?: Partial<Branch>,
): Branch {
  return {
    id,
    name: id,
    mode: LOCKED_MODE[id],
    locked: true,
    messages: [],
    rag: [],
    createdAt: now,
    updatedAt: now,
    ...extra,
  };
}

export function emptySnapshot(): ChatSnapshot {
  const now = Date.now();
  return {
    currentId: "story",
    pendingAttachments: [],
    forceAgent: false,
    pendingOoc: "",
    branches: {
      story: lockedBranch("story", now),
      assistant: lockedBranch("assistant", now),
    },
  };
}

/** Normalize persisted state: `default` → `story`, lock protected modes. */
export function migrateSnapshot(raw: ChatSnapshot): ChatSnapshot {
  const now = Date.now();
  const branches: Record<string, Branch> = { ...raw.branches };

  if (branches.default && !branches.story) {
    const d = branches.default;
    branches.story = {
      ...d,
      id: "story",
      name: d.name === "default" ? "story" : d.name,
      mode: "story",
      locked: true,
    };
  }
  delete branches.default;

  if (!branches.story) branches.story = lockedBranch("story", now);
  else {
    branches.story = {
      ...branches.story,
      id: "story",
      mode: "story",
      locked: true,
    };
  }

  if (!branches.assistant) branches.assistant = lockedBranch("assistant", now);
  else {
    branches.assistant = {
      ...branches.assistant,
      id: "assistant",
      mode: "assistant",
      locked: true,
    };
  }

  let currentId = raw.currentId === "default" ? "story" : raw.currentId;
  if (!branches[currentId]) currentId = "story";

  return {
    currentId,
    branches,
    pendingAttachments: raw.pendingAttachments ?? [],
    forceAgent: false,
    pendingOoc: "",
  };
}
