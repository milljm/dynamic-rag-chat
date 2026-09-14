import { useEffect, useState } from "react";

// UI behavior toggles live in localStorage (per-browser, applies instantly)
// rather than .chat.yaml — they describe how this browser renders the
// thread, not what the server should run.
export const BEHAVIORS_KEY = "spur-behaviors";

export type Behaviors = {
  clearScreen: boolean;
  autoScroll: boolean;
};

export const DEFAULT_BEHAVIORS: Behaviors = {
  clearScreen: false,
  autoScroll: true,
};

export function normalizeBehaviors(value: unknown): Behaviors {
  const source = (typeof value === "object" && value !== null ? value : {}) as Record<
    string,
    unknown
  >;
  return {
    clearScreen:
      typeof source.clearScreen === "boolean" ? source.clearScreen : DEFAULT_BEHAVIORS.clearScreen,
    autoScroll:
      typeof source.autoScroll === "boolean" ? source.autoScroll : DEFAULT_BEHAVIORS.autoScroll,
  };
}

export function readBehaviors(): Behaviors {
  if (typeof window === "undefined") return DEFAULT_BEHAVIORS;
  try {
    const raw = window.localStorage.getItem(BEHAVIORS_KEY);
    if (!raw) return DEFAULT_BEHAVIORS;
    return normalizeBehaviors(JSON.parse(raw) as unknown);
  } catch {
    /* ignore */
  }
  return DEFAULT_BEHAVIORS;
}

const listeners = new Set<(behaviors: Behaviors) => void>();

export function persistBehaviors(next: Behaviors) {
  try {
    window.localStorage.setItem(BEHAVIORS_KEY, JSON.stringify(next));
  } catch {
    /* ignore */
  }
  listeners.forEach((fn) => fn(next));
}

export function useBehaviors(): [Behaviors, (next: Partial<Behaviors>) => void] {
  const [behaviors, setBehaviors] = useState(DEFAULT_BEHAVIORS);

  useEffect(() => {
    setBehaviors(readBehaviors());
    const onChange = (next: Behaviors) => {
      setBehaviors(next);
    };
    listeners.add(onChange);
    return () => {
      listeners.delete(onChange);
    };
  }, []);

  function update(next: Partial<Behaviors>) {
    persistBehaviors({ ...readBehaviors(), ...next });
  }

  return [behaviors, update];
}
