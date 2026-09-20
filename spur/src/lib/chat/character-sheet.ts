import { useEffect, useState } from "react";
import { chatPyOrigin, usesChatPy } from "./remote";

export type CharacterSheetFile = {
  ok: boolean;
  enabled?: boolean;
  path?: string;
  content?: string;
  error?: string;
  message?: string;
};

function url(path: string): string {
  return `${chatPyOrigin()}${path}`;
}

export async function fetchCharacterSheet(): Promise<CharacterSheetFile> {
  const res = await fetch(url("/api/character-sheet"));
  const json = (await res.json()) as CharacterSheetFile;
  json.ok = Boolean(json.ok);
  if (!res.ok && !json.error) json.error = `Request failed (${res.status})`;
  return json;
}

export async function saveCharacterSheet(
  content: string,
): Promise<CharacterSheetFile> {
  const res = await fetch(url("/api/character-sheet"), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  const json = (await res.json()) as CharacterSheetFile;
  json.ok = Boolean(json.ok);
  if (!res.ok && !json.error) json.error = `Request failed (${res.status})`;
  return json;
}

/** Emitted by the settings panel after a successful save (chat rebuilds from yaml). */
export const SETTINGS_SAVED_EVENT = "spur-settings";

/**
 * Whether the running chat has a character sheet configured
 * (`--character-sheet` or `character_sheet:` in `.chat.yaml`).
 */
export function useCharacterSheetEnabled(): boolean {
  const [enabled, setEnabled] = useState(false);
  useEffect(() => {
    if (!usesChatPy()) return;
    let cancelled = false;
    const load = () => {
      void fetchCharacterSheet().then((sheet) => {
        if (!cancelled) setEnabled(Boolean(sheet.enabled));
      });
    };
    load();
    window.addEventListener(SETTINGS_SAVED_EVENT, load);
    return () => {
      cancelled = true;
      window.removeEventListener(SETTINGS_SAVED_EVENT, load);
    };
  }, []);
  return enabled;
}
