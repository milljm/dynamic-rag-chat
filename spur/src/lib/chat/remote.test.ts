import assert from "node:assert/strict";
import { test } from "node:test";
import { mergeMessages, mergeRemoteSnapshot, sessionToSnapshot } from "./remote.ts";
import type { Branch, Message } from "./types.ts";

test("session snapshot maps pickle-shaped branches", () => {
  const snap = sessionToSnapshot({
    currentId: "story",
    branches: {
      story: {
        id: "story",
        name: "story",
        mode: "story",
        locked: true,
        messages: [{ role: "user", content: "hi" }],
      },
      "alt-ending": {
        id: "alt-ending",
        name: "alt-ending",
        mode: "story",
        locked: false,
        messages: [],
      },
    },
  });
  assert.equal(snap.currentId, "story");
  assert.equal(snap.branches.story.locked, true);
  assert.equal(snap.branches.story.messages[0].content, "hi");
  assert.equal(snap.branches["alt-ending"].mode, "story");
});

test("session snapshot ignores non-object branch values", () => {
  const snap = sessionToSnapshot({
    currentId: "story",
    branches: {
      story: {
        id: "story",
        name: "story",
        mode: "story",
        locked: true,
        messages: [],
      },
      // old adapter sent turn counts instead of branch objects
      stale: 4 as unknown as {
        id: string;
        name: string;
        mode: "story";
        locked: boolean;
        messages: [];
      },
    },
  });
  assert.equal("stale" in snap.branches, false);
  assert.equal("story" in snap.branches, true);
});

test("coerceContent flattens langchain-style arrays", () => {
  const snap = sessionToSnapshot({
    currentId: "story",
    branches: {
      story: {
        id: "story",
        name: "story",
        mode: "story",
        locked: true,
        messages: [
          {
            role: "assistant",
            content: [{ type: "text", text: "```js\n1\n```" }] as unknown as string,
          },
        ],
      },
    },
  });
  assert.equal(snap.branches.story.messages[0].content.includes("```js"), true);
});

test("merge keeps local fenced content when pickle dropped it", () => {
  const fenced = "here\n```python\nprint(1)\n```\n";
  const local: Message[] = [
    { id: "u", role: "user", content: "code", createdAt: 1 },
    { id: "a", role: "assistant", content: fenced, createdAt: 2 },
  ];
  const remote: Message[] = [
    { id: "story-0", role: "user", content: "code", createdAt: 3 },
    { id: "story-1", role: "assistant", content: "here print(1)", createdAt: 3 },
  ];
  const merged = mergeMessages(local, remote);
  assert.equal(merged[1].content, fenced);
  assert.equal(merged[1].id, "a");
});

test("mergeRemoteSnapshot does not drop a branch the client already streamed", () => {
  const localBranch = {
    id: "story",
    name: "story",
    mode: "story" as const,
    locked: true,
    messages: [
      { id: "a", role: "assistant" as const, content: "```js\nx\n```", createdAt: 1 },
    ],
    rag: [],
    createdAt: 1,
    updatedAt: 1,
  } satisfies Branch;
  const remote = sessionToSnapshot({
    currentId: "assistant",
    branches: {
      story: {
        id: "story",
        name: "story",
        mode: "story",
        locked: true,
        messages: [{ role: "assistant", content: "x" }],
      },
      assistant: {
        id: "assistant",
        name: "assistant",
        mode: "assistant",
        locked: true,
        messages: [],
      },
    },
  });
  const merged = mergeRemoteSnapshot(
    { currentId: "story", branches: { story: localBranch } },
    remote,
  );
  assert.equal(merged.currentId, "assistant");
  assert.equal(merged.branches.story.messages[0].content.includes("```js"), true);
});
