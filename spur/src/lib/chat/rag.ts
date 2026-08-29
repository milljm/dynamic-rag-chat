import type { RagChunk } from "./types";

const STOP = new Set([
  "the",
  "and",
  "for",
  "that",
  "with",
  "this",
  "from",
  "have",
  "are",
  "was",
  "but",
  "not",
  "you",
  "your",
  "our",
  "its",
]);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 2 && !STOP.has(t));
}

export function chunkText(
  source: string,
  text: string,
  size = 900,
  overlap = 120,
): RagChunk[] {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return [];
  const chunks: RagChunk[] = [];
  let i = 0;
  let n = 0;
  while (i < clean.length) {
    const end = Math.min(i + size, clean.length);
    chunks.push({
      id: `${source}-${n}`,
      source,
      text: clean.slice(i, end),
    });
    n += 1;
    if (end >= clean.length) break;
    i = Math.max(end - overlap, i + 1);
  }
  return chunks;
}

export function retrieve(chunks: RagChunk[], query: string, k = 4): RagChunk[] {
  if (!chunks.length || !query.trim()) return [];
  const terms = tokenize(query);
  if (!terms.length) return [];

  const scored = chunks.map((chunk) => {
    const hay = tokenize(chunk.text);
    if (!hay.length) return { chunk, score: 0 };
    let hits = 0;
    for (const t of terms) {
      for (const h of hay) {
        if (h === t) hits += 1;
      }
    }
    return { chunk, score: hits / Math.sqrt(hay.length) };
  });

  return scored
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((s) => s.chunk);
}

export function formatRagBlock(chunks: RagChunk[]): string {
  if (!chunks.length) return "";
  const body = chunks
    .map((c, i) => `[${i + 1}] (${c.source})\n${c.text}`)
    .join("\n\n");
  return `Retrieved context:\n${body}`;
}
