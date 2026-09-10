import type { Message } from "./types";

/**
 * Coalesces high-frequency streaming patches into at most one store write
 * per animation frame. LLM token events arrive far faster than the browser
 * paints; without batching, every token re-clones the branch's message
 * array and churns React reconciliation for the whole thread.
 */
export type PatchQueue = {
  /** Queue a partial patch; merged over any pending patch. */
  push: (partial: Partial<Message>) => void;
  /** Apply the pending patch immediately, if any. */
  flush: () => void;
  /** Drop the pending patch without touching the store. */
  cancel: () => void;
};

type Scheduler = (callback: () => void) => () => void;

const defaultScheduler: Scheduler =
  typeof requestAnimationFrame === "function"
    ? (callback) => {
        const id = requestAnimationFrame(callback);
        return () => cancelAnimationFrame(id);
      }
    : (callback) => {
        const id = setTimeout(callback, 16);
        return () => clearTimeout(id);
      };

export function createPatchQueue(
  replace: (partial: Partial<Message>) => void,
  schedule: Scheduler = defaultScheduler,
): PatchQueue {
  let pending: Partial<Message> = {};
  let cancelScheduled: (() => void) | null = null;

  const run = () => {
    cancelScheduled = null;
    const partial = pending;
    pending = {};
    if (Object.keys(partial).length > 0) replace(partial);
  };

  return {
    push(partial) {
      pending = { ...pending, ...partial };
      if (!cancelScheduled) cancelScheduled = schedule(run);
    },
    flush() {
      if (cancelScheduled) {
        cancelScheduled();
        cancelScheduled = null;
      }
      run();
    },
    cancel() {
      if (cancelScheduled) {
        cancelScheduled();
        cancelScheduled = null;
      }
      pending = {};
    },
  };
}
