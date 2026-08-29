import type { Attachment, Branch, ChatSnapshot, Message, Mode, StreamMetrics } from "./types";

export function chatPyOrigin(): string {
  const raw = (import.meta.env.VITE_CHAT_API as string | undefined) ?? "";
  return raw.trim().replace(/\/+$/, "");
}

export function usesChatPy(): boolean {
  return Boolean(chatPyOrigin());
}

function url(path: string): string {
  return `${chatPyOrigin()}${path}`;
}

export type RemoteOp = { ok: boolean; error?: string; id?: string };

type RemoteMessage = {
  role: "user" | "assistant";
  content: unknown;
  reasoning?: string;
  attachments?: Attachment[];
  metrics?: StreamMetrics;
};

type RemoteBranch = {
  id: string;
  name: string;
  mode: Mode;
  locked: boolean;
  messages: RemoteMessage[];
};

export type RemoteSession = {
  currentId: string;
  branches: Record<string, RemoteBranch>;
};

export function coerceContent(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") {
          const rec = item as { text?: unknown; content?: unknown };
          return coerceContent(rec.text ?? rec.content ?? "");
        }
        return "";
      })
      .join("");
  }
  if (typeof value === "object") {
    const rec = value as { text?: unknown; content?: unknown };
    if (rec.text != null || rec.content != null) {
      return coerceContent(rec.text ?? rec.content);
    }
  }
  return String(value);
}

function fenceScore(text: string): number {
  return (text.match(/```/g) ?? []).length + (text.match(/~~~/g) ?? []).length;
}

export function mergeMessages(local: Message[], remote: Message[]): Message[] {
  const n = Math.max(local.length, remote.length);
  const out: Message[] = [];
  for (let i = 0; i < n; i++) {
    const l = local[i];
    const r = remote[i];
    if (l && r && l.role === r.role) {
      const localWins =
        fenceScore(l.content) > fenceScore(r.content) ||
        (fenceScore(l.content) === fenceScore(r.content) &&
          l.content.length >= r.content.length);
      out.push({
        ...r,
        id: l.id,
        content: localWins ? l.content : r.content,
        reasoning: l.reasoning || r.reasoning,
        attachments: l.attachments?.length ? l.attachments : r.attachments,
        metrics: l.metrics ?? r.metrics,
        flags: l.flags ?? r.flags,
      });
    } else {
      out.push(l ?? r);
    }
  }
  return out;
}

export function mergeRemoteSnapshot(
  local: Pick<ChatSnapshot, "currentId" | "branches">,
  remote: Pick<ChatSnapshot, "currentId" | "branches">,
): Pick<ChatSnapshot, "currentId" | "branches"> {
  const branches: Record<string, Branch> = {};
  for (const [id, remoteBranch] of Object.entries(remote.branches)) {
    const loc = local.branches[id];
    branches[id] = loc
      ? {
          ...remoteBranch,
          messages: mergeMessages(loc.messages, remoteBranch.messages),
          rag: loc.rag.length ? loc.rag : remoteBranch.rag,
          createdAt: loc.createdAt,
        }
      : remoteBranch;
  }
  return { currentId: remote.currentId, branches };
}

export function sessionToSnapshot(
  raw: RemoteSession,
): Pick<ChatSnapshot, "currentId" | "branches"> {
  const now = Date.now();
  const branches: Record<string, Branch> = {};
  for (const [id, b] of Object.entries(raw.branches ?? {})) {
    if (!b || typeof b !== "object" || Array.isArray(b)) continue;
    const messages: Message[] = (b.messages ?? []).map((m, i) => ({
      id: `${id}-${i}`,
      role: m.role === "user" ? "user" : "assistant",
      content: coerceContent(m.content),
      reasoning: m.reasoning || undefined,
      attachments: m.attachments?.length ? m.attachments : undefined,
      metrics: m.metrics,
      createdAt: now,
    }));
    branches[id] = {
      id,
      name: b.name || id,
      mode: b.mode === "assistant" ? "assistant" : "story",
      locked: Boolean(b.locked),
      messages,
      rag: [],
      createdAt: now,
      updatedAt: now,
    };
  }
  const currentId =
    raw.currentId && branches[raw.currentId]
      ? raw.currentId
      : Object.keys(branches)[0] || "story";
  return { currentId, branches };
}

export async function getSession(): Promise<RemoteSession> {
  const res = await fetch(url("/api/session"));
  if (!res.ok) throw new Error("chat.py is not reachable.");
  return (await res.json()) as RemoteSession;
}

export async function postOp(
  path: string,
  body: Record<string, unknown> = {},
): Promise<RemoteOp> {
  const res = await fetch(url(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let json: RemoteOp = { ok: res.ok };
  try {
    json = (await res.json()) as RemoteOp;
  } catch {
    /* empty body */
  }
  if (!res.ok && !json.error) json.error = `Request failed (${res.status})`;
  json.ok = Boolean(json.ok);
  return json;
}

export type GoldDocument = { name: string; chars: number };

export async function listDocuments(): Promise<GoldDocument[]> {
  const res = await fetch(url("/api/documents"));
  if (!res.ok) return [];
  const json = (await res.json()) as { documents?: GoldDocument[] };
  return Array.isArray(json.documents) ? json.documents : [];
}

export async function deleteDocument(name: string): Promise<RemoteOp> {
  return postOp("/api/documents/delete", { name });
}
