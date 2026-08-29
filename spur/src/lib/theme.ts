export type ThemePref = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

export const THEME_KEY = "spur-theme";
export const THEME_META = {
  dark: "#0a0b0a",
  light: "#f1eee6",
} as const;

export function isThemePref(value: unknown): value is ThemePref {
  return value === "light" || value === "dark" || value === "system";
}

export function readThemePref(): ThemePref {
  if (typeof window === "undefined") return "dark";
  try {
    const stored = window.localStorage.getItem(THEME_KEY);
    if (isThemePref(stored)) return stored;
  } catch {
    /* ignore */
  }
  return "dark";
}

export function systemWantsLight(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.matchMedia("(prefers-color-scheme: light)").matches;
  } catch {
    return false;
  }
}

export function resolveTheme(pref: ThemePref): ResolvedTheme {
  if (pref === "system") return systemWantsLight() ? "light" : "dark";
  return pref;
}

export function applyTheme(pref: ThemePref): ResolvedTheme {
  const resolved = resolveTheme(pref);
  const root = document.documentElement;
  root.dataset.theme = resolved;
  root.style.colorScheme = resolved;
  root.classList.toggle("dark", resolved === "dark");
  root.classList.toggle("light", resolved === "light");
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", THEME_META[resolved]);
  return resolved;
}

export function persistThemePref(pref: ThemePref) {
  try {
    window.localStorage.setItem(THEME_KEY, pref);
  } catch {
    /* ignore */
  }
}

/** Inline boot — keep in sync with applyTheme. Avoids a dark→light flash. */
export const THEME_BOOT_SCRIPT = `(function(){try{var k=${JSON.stringify(THEME_KEY)};var p=localStorage.getItem(k)||"dark";if(p!=="light"&&p!=="dark"&&p!=="system")p="dark";var r=p==="system"?(matchMedia("(prefers-color-scheme: light)").matches?"light":"dark"):p;var e=document.documentElement;e.setAttribute("data-theme",r);e.style.colorScheme=r;e.classList.toggle("dark",r==="dark");e.classList.toggle("light",r==="light");}catch(e){document.documentElement.setAttribute("data-theme","dark");}})();`;
