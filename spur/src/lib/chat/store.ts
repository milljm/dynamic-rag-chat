import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { fileToAttachment, isAllowedFile } from "./attachments";
import {
  applyAppendMessage,
  applyCreateBranch,
  applyDeleteBranch,
  applyDeleteLastTurn,
  applyPopLastAssistant,
  applyReplaceMessage,
  applyResetBranch,
  applyRewind,
  applySetForceAgent,
  applySetMode,
  applySwitch,
  emptySnapshot,
  isLockedBranch,
  migrateSnapshot,
  parseBranchInput,
} from "./branch-mode";
import { chunkText } from "./rag";
import {
  getSession,
  mergeRemoteSnapshot,
  postOp,
  sessionToSnapshot,
  usesChatPy,
} from "./remote";
import { seedSnapshot } from "./seed";
import type {
  Attachment,
  ChatSnapshot,
  Message,
  Mode,
  RagChunk,
} from "./types";

type ChatActions = {
  switchBranch: (id: string) => boolean;
  setMode: (mode: Mode) => boolean;
  setForceAgent: (enabled: boolean) => void;
  createBranch: (
    raw: string,
  ) => { ok: true; id: string } | { ok: false; error: string };
  deleteBranch: (id: string) => { ok: true } | { ok: false; error: string };
  resetBranch: () => { ok: true } | { ok: false; error: string };
  deleteLastTurn: () => { ok: true } | { ok: false; error: string };
  rewindTo: (n: number) => { ok: true } | { ok: false; error: string };
  popLastAssistant: () => boolean;
  setPendingOoc: (text: string) => void;
  addFiles: (files: File[]) => Promise<{
    added: number;
    skipped: number;
    unreadable: number;
    reason: string;
  }>;
  removePending: (id: string) => void;
  clearPending: () => void;
  appendMessage: (
    message: Message,
    ragExtra?: RagChunk[],
    branchId?: string,
  ) => void;
  replaceMessage: (
    id: string,
    patch: Partial<Message>,
    branchId?: string,
  ) => void;
  hydrateFromServer: () => Promise<boolean>;
};

export type ChatStore = ChatSnapshot & ChatActions;

const memoryStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
};

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      ...(usesChatPy() ? emptySnapshot() : seedSnapshot()),

      switchBranch: (id) => {
        if (usesChatPy()) {
          set((s) => applySwitch(s, id));
          void postOp("/api/branches/switch", { id }).then(() =>
            get().hydrateFromServer(),
          );
          return get().currentId === id;
        }
        const before = get().currentId;
        set((s) => applySwitch(s, id));
        return get().currentId !== before && get().currentId === id;
      },

      setMode: (mode) => {
        if (isLockedBranch(get().currentId)) return false;
        if (usesChatPy()) {
          set((s) => applySetMode(s, mode));
          void postOp("/api/session/mode", { mode }).then(() =>
            get().hydrateFromServer(),
          );
          return get().branches[get().currentId]?.mode === mode;
        }
        set((s) => applySetMode(s, mode));
        return get().branches[get().currentId]?.mode === mode;
      },

      setForceAgent: (enabled) => set((s) => applySetForceAgent(s, enabled)),

      createBranch: (raw) => {
        const parsed = parseBranchInput(raw);
        if ("error" in parsed) return { ok: false, error: parsed.error };
        if (usesChatPy()) {
          void postOp("/api/branches", {
            name: parsed.name,
            cutTurns: parsed.cutTurns ?? null,
          }).then(() => get().hydrateFromServer());
          return { ok: true, id: parsed.name };
        }
        const result = applyCreateBranch(get(), parsed.name, parsed.cutTurns);
        if (!result.ok) return result;
        set(result.state);
        return { ok: true, id: result.id };
      },

      deleteBranch: (id) => {
        if (usesChatPy()) {
          const result = applyDeleteBranch(get(), id);
          if (result.ok) set(result.state);
          void postOp("/api/branches/delete", { id }).then(() =>
            get().hydrateFromServer(),
          );
          return result.ok ? { ok: true } : result;
        }
        const result = applyDeleteBranch(get(), id);
        if (!result.ok) return result;
        set(result.state);
        return { ok: true };
      },

      resetBranch: () => {
        if (usesChatPy()) {
          const result = applyResetBranch(get());
          if (result.ok) set(result.state);
          void postOp("/api/history/reset").then(() => get().hydrateFromServer());
          return result.ok ? { ok: true } : result;
        }
        const result = applyResetBranch(get());
        if (!result.ok) return result;
        set(result.state);
        return { ok: true };
      },

      deleteLastTurn: () => {
        if (usesChatPy()) {
          const result = applyDeleteLastTurn(get());
          if (result.ok) set(result.state);
          void postOp("/api/history/delete-last").then(() =>
            get().hydrateFromServer(),
          );
          return result.ok ? { ok: true } : result;
        }
        const result = applyDeleteLastTurn(get());
        if (!result.ok) return result;
        set(result.state);
        return { ok: true };
      },

      rewindTo: (n) => {
        if (usesChatPy()) {
          const result = applyRewind(get(), n);
          if (result.ok) set(result.state);
          void postOp("/api/history/rewind", { n }).then(() =>
            get().hydrateFromServer(),
          );
          return result.ok ? { ok: true } : result;
        }
        const result = applyRewind(get(), n);
        if (!result.ok) return result;
        set(result.state);
        return { ok: true };
      },

      popLastAssistant: () => {
        const before = get().branches[get().currentId]?.messages.length ?? 0;
        set((s) => applyPopLastAssistant(s));
        const after = get().branches[get().currentId]?.messages.length ?? 0;
        if (usesChatPy() && after < before) {
          void postOp("/api/history/pop-assistant");
        }
        return after < before;
      },

      setPendingOoc: (text) => set({ pendingOoc: text }),

      addFiles: async (files) => {
        let added = 0;
        let skipped = 0;
        let unreadable = 0;
        let reason = "";
        const next: Attachment[] = [];
        for (const file of files) {
          if (!isAllowedFile(file)) {
            skipped += 1;
            continue;
          }
          try {
            next.push(await fileToAttachment(file));
            added += 1;
          } catch (err) {
            unreadable += 1;
            reason = err instanceof Error ? err.message : "unknown error";
          }
        }
        if (next.length) {
          set((s) => ({
            pendingAttachments: [...s.pendingAttachments, ...next],
          }));
        }
        return { added, skipped, unreadable, reason };
      },

      removePending: (id) =>
        set((s) => ({
          pendingAttachments: s.pendingAttachments.filter((a) => a.id !== id),
        })),

      clearPending: () => set({ pendingAttachments: [] }),

      appendMessage: (message, ragExtra = [], branchId) =>
        set((s) => applyAppendMessage(s, message, ragExtra, branchId)),

      replaceMessage: (id, patch, branchId) =>
        set((s) => applyReplaceMessage(s, id, patch, branchId)),

      hydrateFromServer: async () => {
        if (!usesChatPy()) return false;
        try {
          const snap = sessionToSnapshot(await getSession());
          if (!Object.keys(snap.branches).length) return false;
          const merged = mergeRemoteSnapshot(get(), snap);
          set({ currentId: merged.currentId, branches: merged.branches });
          return true;
        } catch {
          return false;
        }
      },
    }),
    {
      name: "spur-chat-v1",
      version: 2,
      storage: createJSONStorage(() =>
        typeof window === "undefined" || usesChatPy()
          ? memoryStorage
          : localStorage,
      ),
      partialize: (s) => ({
        currentId: s.currentId,
        branches: s.branches,
      }),
      merge: (persisted, current) => {
        try {
          const p = (persisted ?? {}) as Partial<ChatSnapshot>;
          const migrated = migrateSnapshot({
            currentId: p.currentId ?? current.currentId,
            branches: p.branches ?? current.branches,
            pendingAttachments: current.pendingAttachments,
            forceAgent: false,
            pendingOoc: current.pendingOoc ?? "",
          });
          return { ...current, ...migrated };
        } catch {
          return current;
        }
      },
      skipHydration: true,
    },
  ),
);

export function ragFromPending(pending: Attachment[]): RagChunk[] {
  const chunks: RagChunk[] = [];
  for (const att of pending) {
    if (att.kind === "text" && att.text) {
      chunks.push(...chunkText(att.name, att.text));
    }
  }
  return chunks;
}
