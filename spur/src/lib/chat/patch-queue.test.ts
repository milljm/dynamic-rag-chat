import assert from "node:assert/strict";
import { test } from "node:test";
import { createPatchQueue } from "./patch-queue.ts";
import type { Message } from "./types.ts";

/** Manual scheduler: tests decide when the "frame" fires. */
function manualScheduler() {
  const callbacks: Array<() => void> = [];
  return {
    schedule: (cb: () => void) => {
      callbacks.push(cb);
      return () => {
        const i = callbacks.indexOf(cb);
        if (i >= 0) callbacks.splice(i, 1);
      };
    },
    frame: () => {
      const pending = callbacks.splice(0);
      for (const cb of pending) cb();
    },
    get size() {
      return callbacks.length;
    },
  };
}

test("coalesces many pushes into one store write per frame", () => {
  const timer = manualScheduler();
  const writes: Partial<Message>[] = [];
  const queue = createPatchQueue((partial) => writes.push(partial), timer.schedule);
  queue.push({ content: "a" });
  queue.push({ status: "Streaming…" });
  queue.push({ content: "ab" });
  assert.equal(timer.size, 1, "only one frame scheduled");
  timer.frame();
  assert.equal(writes.length, 1);
  assert.deepEqual(writes[0], { content: "ab", status: "Streaming…" });
});

test("later pushes override earlier fields", () => {
  const timer = manualScheduler();
  const writes: Partial<Message>[] = [];
  const queue = createPatchQueue((partial) => writes.push(partial), timer.schedule);
  queue.push({ content: "first", status: "A" });
  queue.push({ content: "second" });
  timer.frame();
  assert.deepEqual(writes[0], { content: "second", status: "A" });
});

test("flush applies immediately and un-schedules the frame", () => {
  const timer = manualScheduler();
  const writes: Partial<Message>[] = [];
  const queue = createPatchQueue((partial) => writes.push(partial), timer.schedule);
  queue.push({ content: "x" });
  queue.flush();
  assert.equal(writes.length, 1);
  assert.deepEqual(writes[0], { content: "x" });
  assert.equal(timer.size, 0);
  // A push after flush schedules a fresh frame.
  queue.push({ content: "y" });
  assert.equal(timer.size, 1);
  timer.frame();
  assert.equal(writes.length, 2);
});

test("cancel drops the pending patch without touching the store", () => {
  const timer = manualScheduler();
  let writes = 0;
  const queue = createPatchQueue(() => {
    writes += 1;
  }, timer.schedule);
  queue.push({ content: "discarded" });
  queue.cancel();
  assert.equal(timer.size, 0);
  timer.frame();
  assert.equal(writes, 0);
});

test("push after cancel schedules a fresh frame", () => {
  const timer = manualScheduler();
  const writes: Partial<Message>[] = [];
  const queue = createPatchQueue((partial) => writes.push(partial), timer.schedule);
  queue.push({ content: "a" });
  queue.cancel();
  queue.push({ content: "b" });
  assert.equal(timer.size, 1);
  timer.frame();
  assert.deepEqual(writes, [{ content: "b" }]);
});

test("empty flush is a no-op", () => {
  const timer = manualScheduler();
  let writes = 0;
  const queue = createPatchQueue(() => {
    writes += 1;
  }, timer.schedule);
  queue.flush();
  assert.equal(writes, 0);
});
