import assert from "node:assert/strict";
import { test } from "node:test";
import {
  activeBranch,
  applyCreateBranch,
  applyDeleteBranch,
  applyDeleteLastTurn,
  applyPopLastAssistant,
  applyReplaceMessage,
  applyResetBranch,
  applyRewind,
  applySetMode,
  applySwitch,
  emptySnapshot,
  isLockedBranch,
  lastUserMessage,
  migrateSnapshot,
  modeOf,
  parseBranchInput,
  takeTurns,
} from "./branch-mode.ts";
import type { ChatSnapshot, Message } from "./types.ts";

function userBranch(
  state: ChatSnapshot,
  id: string,
  mode: "assistant" | "story",
): ChatSnapshot {
  return {
    ...state,
    branches: {
      ...state.branches,
      [id]: {
        id,
        name: id,
        mode,
        locked: false,
        messages: [],
        rag: [],
        createdAt: 1,
        updatedAt: 1,
      },
    },
  };
}

function withTurns(state: ChatSnapshot, id: string, n: number): ChatSnapshot {
  const branch = state.branches[id]!;
  const messages: Message[] = [];
  for (let i = 1; i <= n; i++) {
    messages.push({
      id: `u${i}`,
      role: "user",
      content: `q${i}`,
      createdAt: i,
    });
    messages.push({
      id: `a${i}`,
      role: "assistant",
      content: `A${i}`,
      createdAt: i + 0.5,
    });
  }
  return {
    ...state,
    branches: {
      ...state.branches,
      [id]: { ...branch, messages },
    },
  };
}

test("mode does not hijack the current branch", () => {
  let s = userBranch(emptySnapshot(), "research-notes", "assistant");
  s = applySwitch(s, "research-notes");
  assert.equal(s.currentId, "research-notes");
  assert.equal(modeOf(s.branches[s.currentId]!), "assistant");
  assert.notEqual(s.currentId, "assistant");
  assert.equal(activeBranch(s)?.id, "research-notes");
});

test("switching restores the destination branch mode", () => {
  let s = userBranch(emptySnapshot(), "research-notes", "assistant");
  s = userBranch(s, "alt-ending", "story");
  s = applySwitch(s, "research-notes");
  assert.equal(modeOf(activeBranch(s)!), "assistant");
  s = applySwitch(s, "alt-ending");
  assert.equal(s.currentId, "alt-ending");
  assert.equal(modeOf(activeBranch(s)!), "story");
  s = applySwitch(s, "story");
  assert.equal(s.currentId, "story");
  assert.equal(modeOf(activeBranch(s)!), "story");
  assert.equal(isLockedBranch(s.currentId), true);
});

test("setMode on a locked branch is a no-op", () => {
  const s = applySetMode(emptySnapshot(), "assistant");
  assert.equal(s.currentId, "story");
  assert.equal(modeOf(s.branches.story), "story");
});

test("setMode on a user branch persists and does not change currentId", () => {
  let s = userBranch(emptySnapshot(), "alt-ending", "story");
  s = applySwitch(s, "alt-ending");
  s = applySetMode(s, "assistant");
  assert.equal(s.currentId, "alt-ending");
  assert.equal(s.branches["alt-ending"]!.mode, "assistant");
  assert.equal(isLockedBranch(s.currentId), false);
});

test("create branch from assistant lands on the new branch", () => {
  let s = emptySnapshot();
  s = applySwitch(s, "assistant");
  const result = applyCreateBranch(s, "experiment", null);
  assert.equal(result.ok, true);
  if (!result.ok) return;
  assert.equal(result.state.currentId, result.id);
  assert.notEqual(result.state.currentId, "assistant");
  assert.equal(modeOf(result.state.branches[result.id]!), "assistant");
  assert.equal(isLockedBranch(result.state.currentId), false);
});

test("create branch copies source mode, not a global flag", () => {
  let s = userBranch(emptySnapshot(), "alt-ending", "story");
  s = applySwitch(s, "alt-ending");
  const result = applyCreateBranch(s, "fork-a", null);
  assert.equal(result.ok, true);
  if (!result.ok) return;
  assert.equal(modeOf(result.state.branches[result.id]!), "story");
});

test("reserved names are rejected", () => {
  const s = emptySnapshot();
  assert.equal(applyCreateBranch(s, "assistant", null).ok, false);
  assert.equal(applyCreateBranch(s, "story", null).ok, false);
  assert.equal(applyCreateBranch(s, "current", null).ok, false);
  assert.equal(applyCreateBranch(s, "  ", null).ok, false);
});

test("parseBranchInput reads @N cuts", () => {
  assert.deepEqual(parseBranchInput("testing@5"), {
    name: "testing",
    cutTurns: 5,
  });
  assert.equal("error" in parseBranchInput("testing@nope"), true);
  assert.equal("error" in parseBranchInput("assistant"), true);
  assert.equal("error" in parseBranchInput("story"), true);
});

test("takeTurns keeps the assistant reply of the last included turn", () => {
  const messages: Message[] = [
    { id: "u1", role: "user", content: "a", createdAt: 1 },
    { id: "a1", role: "assistant", content: "A", createdAt: 2 },
    { id: "u2", role: "user", content: "b", createdAt: 3 },
    { id: "a2", role: "assistant", content: "B", createdAt: 4 },
    { id: "u3", role: "user", content: "c", createdAt: 5 },
    { id: "a3", role: "assistant", content: "C", createdAt: 6 },
  ];
  const cut = takeTurns(messages, 2);
  assert.deepEqual(
    cut.map((m) => m.id),
    ["u1", "a1", "u2", "a2"],
  );
});

test("unknown switch is a no-op", () => {
  const s = emptySnapshot();
  assert.equal(applySwitch(s, "missing").currentId, "story");
});

test("cannot delete the active or locked branch", () => {
  let s = userBranch(emptySnapshot(), "scratch", "story");
  s = applySwitch(s, "scratch");
  assert.equal(applyDeleteBranch(s, "scratch").ok, false);
  assert.equal(applyDeleteBranch(s, "story").ok, false);
  s = applySwitch(s, "story");
  const gone = applyDeleteBranch(s, "scratch");
  assert.equal(gone.ok, true);
  if (gone.ok) assert.equal(gone.state.branches.scratch, undefined);
});

test("rewind and delete-last keep branch identity", () => {
  let s = userBranch(emptySnapshot(), "scratch", "assistant");
  s = applySwitch(s, "scratch");
  s = withTurns(s, "scratch", 3);
  const rewound = applyRewind(s, 1);
  assert.equal(rewound.ok, true);
  if (rewound.ok) {
    assert.equal(rewound.state.currentId, "scratch");
    assert.deepEqual(
      rewound.state.branches.scratch!.messages.map((m) => m.id),
      ["u1", "a1"],
    );
  }
  const trimmed = applyDeleteLastTurn(s);
  assert.equal(trimmed.ok, true);
  if (trimmed.ok) {
    assert.equal(trimmed.state.branches.scratch!.messages.at(-1)?.id, "a2");
  }
});

test("replaceMessage pins writes to the origin branch", () => {
  let s = userBranch(emptySnapshot(), "a", "assistant");
  s = userBranch(s, "b", "story");
  s = applySwitch(s, "a");
  s = {
    ...s,
    branches: {
      ...s.branches,
      a: {
        ...s.branches.a!,
        messages: [
          { id: "m1", role: "assistant", content: "", createdAt: 1 },
        ],
      },
    },
  };
  s = applySwitch(s, "b");
  s = applyReplaceMessage(s, "m1", { content: "hello" }, "a");
  assert.equal(s.branches.a!.messages[0]!.content, "hello");
  assert.equal(s.currentId, "b");
});

test("reset clears messages and rag on the current branch only", () => {
  let s = userBranch(emptySnapshot(), "scratch", "story");
  s = applySwitch(s, "scratch");
  s = withTurns(s, "scratch", 1);
  const reset = applyResetBranch(s);
  assert.equal(reset.ok, true);
  if (reset.ok) {
    assert.equal(reset.state.branches.scratch!.messages.length, 0);
    assert.equal(reset.state.currentId, "scratch");
    assert.ok(reset.state.branches.story);
  }
});

test("pop last assistant leaves the user turn", () => {
  let s = userBranch(emptySnapshot(), "scratch", "story");
  s = applySwitch(s, "scratch");
  s = withTurns(s, "scratch", 1);
  s = applyPopLastAssistant(s);
  assert.equal(s.branches.scratch!.messages.length, 1);
  assert.equal(s.branches.scratch!.messages[0]!.role, "user");
});

test("lastUserMessage keeps attachments for regenerate", () => {
  const last = lastUserMessage([
    {
      id: "u",
      role: "user",
      content: "look at this",
      attachments: [
        {
          id: "f",
          name: "notes.txt",
          mime: "text/plain",
          kind: "text",
          text: "hello",
          size: 5,
        },
      ],
      createdAt: 1,
    },
    { id: "a", role: "assistant", content: "ok", createdAt: 2 },
  ]);
  assert.equal(last?.content, "look at this");
  assert.equal(last?.attachments?.[0]?.name, "notes.txt");
});

test("migrate remaps default to story without dropping user branches", () => {
  const raw = {
    currentId: "default",
    pendingAttachments: [],
    forceAgent: true,
    pendingOoc: "",
    branches: {
      default: {
        id: "default",
        name: "default",
        mode: "story" as const,
        locked: true,
        messages: [],
        rag: [],
        createdAt: 1,
        updatedAt: 1,
      },
      "alt-ending": {
        id: "alt-ending",
        name: "alt-ending",
        mode: "story" as const,
        locked: false,
        messages: [],
        rag: [],
        createdAt: 1,
        updatedAt: 1,
      },
    },
  };
  const next = migrateSnapshot(raw);
  assert.equal(next.currentId, "story");
  assert.equal(next.branches.default, undefined);
  assert.equal(next.branches.story?.mode, "story");
  assert.ok(next.branches["alt-ending"]);
  assert.ok(next.branches.assistant);
  assert.equal(next.forceAgent, false);
});
