import assert from "node:assert/strict";
import { test } from "node:test";
import { feedThink } from "./think.ts";

test("splits think tags out of the visible answer", () => {
  const state = { inThink: false };
  const a = feedThink("Hello <think>secret", state);
  assert.equal(a.content, "Hello ");
  assert.equal(a.reasoning, "secret");
  assert.equal(state.inThink, true);
  const b = feedThink(" plan</think>\n```js\n1\n```", state);
  assert.equal(b.reasoning, " plan");
  assert.equal(b.content, "\n```js\n1\n```");
  assert.equal(state.inThink, false);
  assert.equal(state.neverThink, true);
});

test("plain tokens pass through", () => {
  const state = { inThink: false };
  const a = feedThink("no tags here", state);
  assert.equal(a.content, "no tags here");
  assert.equal(a.reasoning, "");
  assert.equal(state.neverThink, true);
});

test("mm:think ignores nested <think> mentions (MiniMax)", () => {
  const state = { inThink: false };
  const raw = [
    "<mm:think>If a `<think>` tag gets split mid-token, this could misclassify.",
    " handles `<think>...</think>` style tags.",
    "</mm:think>Took a read-through — solid little FastAPI shim.",
  ].join("");
  const out = feedThink(raw, state);
  assert.equal(state.inThink, false);
  assert.equal(state.neverThink, true);
  assert.match(out.reasoning, /If a `<think>` tag gets split/);
  assert.match(out.reasoning, /<\/think>` style tags/);
  assert.equal(
    out.content,
    "Took a read-through — solid little FastAPI shim.",
  );
});

test("after thinking closes, later <think> mentions stay in the answer", () => {
  const state = { inThink: false };
  const first = feedThink(
    "<think>Let me give my honest take.</think>\n\nOh, spur-server.py — clean move.\n",
    state,
  );
  assert.equal(state.neverThink, true);
  assert.match(first.reasoning, /honest take/);
  assert.match(first.content, /Oh, spur-server/);
  const later = feedThink(
    "4. split_think() to handle `<think>` / `</thinking>` blocks is neat.",
    state,
  );
  assert.equal(later.reasoning, "");
  assert.match(later.content, /`<think>` \/ `<\/thinking>`/);
});

test("blank first tokens latch neverThink on the first non-blank (gpt-oss)", () => {
  const state = { inThink: false };
  const a = feedThink("", state);
  assert.equal(a.content, "");
  assert.equal(state.shadowThink, true);
  feedThink("", state);
  assert.equal(state.shadowThink, true);
  const c = feedThink(
    "Oh, spur-server.py — handle `<think>` / `</thinking>` blocks.",
    state,
  );
  assert.equal(state.neverThink, true);
  assert.equal(c.reasoning, "");
  assert.match(c.content, /`<think>` \/ `<\/thinking>`/);
});

test("literal <think> chunk inside mm:think stays reasoning (MiniMax-M3)", () => {
  const state = { inThink: false };
  feedThink("<mm:think>The file parses ", state);
  assert.equal(state.inThink, true);
  assert.equal(state.ns, "mm:");
  const inner = feedThink("<think>", state);
  assert.equal(inner.content, "");
  assert.equal(inner.reasoning, "<think>");
  assert.equal(state.inThink, true);
  assert.equal(state.ns, "mm:");
  assert.equal(state.neverThink, undefined);
  const more = feedThink(" tags.</mm:think>Solid.", state);
  assert.equal(more.content, "Solid.");
  assert.equal(state.inThink, false);
  assert.equal(state.neverThink, true);
});

test("shadow then <mm:think> still opens namespaced reasoning", () => {
  const state = { inThink: false };
  feedThink("", state);
  assert.equal(state.shadowThink, true);
  const out = feedThink("<mm:think>real reasoning", state);
  assert.equal(state.shadowThink, false);
  assert.equal(state.inThink, true);
  assert.equal(state.ns, "mm:");
  assert.equal(out.content, "");
  assert.equal(out.reasoning, "real reasoning");
});
