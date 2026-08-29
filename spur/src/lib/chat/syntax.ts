import { useEffect, useState } from "react";
import type { ResolvedTheme } from "@/lib/theme";
import {
  PYGMENTS_STYLES,
  SYNTAX_AUTO,
  SYNTAX_KEY,
  findPalette,
  paletteFor,
  type PygmentsPalette,
} from "./pygments-styles";

export {
  PYGMENTS_STYLES,
  SYNTAX_AUTO,
  findPalette,
  paletteFor,
  type PygmentsPalette,
};

const listeners = new Set<(id: string) => void>();

export function listPygmentsStyles(): PygmentsPalette[] {
  return PYGMENTS_STYLES;
}

export function readSyntaxPref(): string {
  if (typeof window === "undefined") return SYNTAX_AUTO;
  try {
    const stored = window.localStorage.getItem(SYNTAX_KEY);
    if (stored === SYNTAX_AUTO || findPalette(stored)) return stored!;
  } catch {
    /* ignore */
  }
  return SYNTAX_AUTO;
}

export function persistSyntaxPref(id: string) {
  try {
    window.localStorage.setItem(SYNTAX_KEY, id);
  } catch {
    /* ignore */
  }
}

export function applySyntax(id: string, resolved: ResolvedTheme) {
  if (typeof document === "undefined") return;
  const pal = paletteFor(id, resolved);
  const root = document.documentElement;
  root.style.setProperty("--spur-hx-kw", pal.kw);
  root.style.setProperty("--spur-hx-st", pal.st);
  root.style.setProperty("--spur-hx-cm", pal.cm);
  root.style.setProperty("--spur-hx-nu", pal.nu);
  root.style.setProperty("--spur-hx-fn", pal.fn);
  root.style.setProperty("--spur-code-fg", pal.fg);
  root.style.setProperty("--spur-code", pal.bg);
  root.dataset.syntax = id || SYNTAX_AUTO;
}

export function setSyntaxPref(id: string, resolved: ResolvedTheme) {
  const next = findPalette(id) ? id : SYNTAX_AUTO;
  persistSyntaxPref(next);
  applySyntax(next, resolved);
  listeners.forEach((fn) => fn(next));
}

export function useSyntaxPref(): [string, (id: string) => void] {
  const [id, setId] = useState(SYNTAX_AUTO);

  useEffect(() => {
    setId(readSyntaxPref());
    listeners.add(setId);
    return () => {
      listeners.delete(setId);
    };
  }, []);

  function choose(next: string) {
    const resolved =
      document.documentElement.dataset.theme === "light" ? "light" : "dark";
    setSyntaxPref(next, resolved);
  }

  return [id, choose];
}
