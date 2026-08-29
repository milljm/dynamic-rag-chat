import type { Branch, ChatSnapshot, Message } from "./types";

function msg(
  role: Message["role"],
  content: string,
  createdAt: number,
): Message {
  return {
    id: `${role}-${createdAt}`,
    role,
    content,
    createdAt,
  };
}

function branch(partial: Branch): Branch {
  return partial;
}

export function seedSnapshot(): ChatSnapshot {
  const t0 = Date.UTC(2026, 7, 20, 14, 0, 0);
  const research: Branch = branch({
    id: "research-notes",
    name: "research-notes",
    mode: "assistant",
    locked: false,
    createdAt: t0,
    updatedAt: t0 + 36 * 60_000,
    rag: [
      {
        id: "notes-0",
        source: "retrieval.md",
        text: "Spur scores attached documents with term overlap against the current question, then injects the top passages into the system prompt. A full clone copies the collection; a cut at turn N rebuilds from the kept messages only. Agent turns use live web search. \\no-context skips retrieval.",
      },
    ],
    messages: [
      msg(
        "user",
        "Summarize the retrieval approach we're using, and call out what happens on a branch cut.",
        t0 + 5 * 60_000,
      ),
      msg(
        "assistant",
        "We chunk each attached document, score overlap with the current question, and inject the top passages into the system prompt.\n\n- **Full clone** copies the collection as-is.\n- **Cut at turn N** rebuilds retrieval from the kept messages only, so later attachments do not leak into the fork.\n- **Agent** turns add live web search. **\\no-context** skips retrieval for one message.",
        t0 + 6 * 60_000,
      ),
    ],
  });

  const alt: Branch = branch({
    id: "alt-ending",
    name: "alt-ending",
    mode: "story",
    locked: false,
    createdAt: t0 + 2 * 60 * 60_000,
    updatedAt: t0 + 2 * 60 * 60_000 + 12 * 60_000,
    rag: [],
    messages: [
      msg("user", "She opens the door anyway.", t0 + 2 * 60 * 60_000),
      msg(
        "assistant",
        "Rain came in first. Then the smell of wet iron, and a hallway that had not been there yesterday — wallpaper blistered, a single coat hook holding nothing, the runner on the floor worn in a path that stopped at her feet.\n\nBehind her the old kitchen kept making kitchen sounds, as if it had not noticed the house had grown a throat.",
        t0 + 2 * 60 * 60_000 + 90_000,
      ),
    ],
  });

  const now = t0;
  return {
    currentId: "story",
    pendingAttachments: [],
    forceAgent: false,
    pendingOoc: "",
    branches: {
      story: {
        id: "story",
        name: "story",
        mode: "story",
        locked: true,
        messages: [],
        rag: [],
        createdAt: now - 86_400_000,
        updatedAt: now - 86_400_000,
      },
      assistant: {
        id: "assistant",
        name: "assistant",
        mode: "assistant",
        locked: true,
        messages: [],
        rag: [],
        createdAt: now - 86_400_000,
        updatedAt: now - 80_000_000,
      },
      "research-notes": research,
      "alt-ending": alt,
    },
  };
}
