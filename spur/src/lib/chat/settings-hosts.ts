/** Cache key for a typed OpenAI-compatible host. */
export function normalizeHost(host: string): string {
  return (host || "").trim().replace(/\/+$/, "");
}

/** Role-specific server, or the main generator server when blank. */
export function hostForRole(roleServer: string, llmServer: string): string {
  return normalizeHost(roleServer) || normalizeHost(llmServer);
}

/** Unique non-empty hosts, first spelling wins. */
export function uniqueHosts(hosts: Iterable<string>): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of hosts) {
    const trimmed = (raw || "").trim();
    const normalized = normalizeHost(trimmed);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(trimmed);
  }
  return out;
}

export type ModelRow = { id: string; loaded?: boolean | null };

/** Prefer mixed-case (server `/v1/models` spelling) over an all-lowercase copy. */
export function preferModelId(a: string, b: string): string {
  if (!a) return b;
  if (!b) return a;
  if (a === b) return a;
  const aFolded = a === a.toLowerCase();
  const bFolded = b === b.toLowerCase();
  if (aFolded !== bFolded) return bFolded ? a : b;
  let aUp = 0;
  let bUp = 0;
  for (const ch of a) if (ch >= "A" && ch <= "Z") aUp++;
  for (const ch of b) if (ch >= "A" && ch <= "Z") bUp++;
  return bUp > aUp ? b : a;
}

function mergeLoaded(
  a: boolean | null | undefined,
  b: boolean | null | undefined,
): boolean | null {
  if (a === true || b === true) return true;
  if (a === false || b === false) return false;
  return a ?? b ?? null;
}

/** Drop case-insensitive duplicates, keeping mixed-case ids. */
export function dedupeModelRows(rows: ModelRow[]): ModelRow[] {
  const map = new Map<string, ModelRow>();
  for (const row of rows) {
    const id = (row.id || "").trim();
    if (!id) continue;
    const key = id.toLowerCase();
    const prev = map.get(key);
    if (!prev) {
      map.set(key, { id, loaded: row.loaded ?? null });
      continue;
    }
    map.set(key, {
      id: preferModelId(prev.id, id),
      loaded: mergeLoaded(prev.loaded, row.loaded),
    });
  }
  return [...map.values()];
}

/** Map a saved id onto the catalog spelling when they differ only by case. */
export function canonicalModelId(value: string, ids: Iterable<string>): string {
  const want = (value || "").trim();
  if (!want) return want;
  for (const id of ids) {
    if (id.toLowerCase() === want.toLowerCase()) return id;
  }
  return want;
}

