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
