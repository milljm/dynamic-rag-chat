/** GFM table parse. Spur's markdown renderer is custom; tables need this. */

export type TableAlign = "left" | "center" | "right" | null;

export type MdTable = {
  headers: string[];
  align: TableAlign[];
  rows: string[][];
};

const DELIM_CELL = /^:?-{3,}:?$/;

function pad(cells: string[], cols: number): string[] {
  const out = cells.slice(0, cols);
  while (out.length < cols) out.push("");
  return out;
}

/** Split a GFM table row into cells. Null if the line is not a row. */
export function splitRow(line: string): string[] | null {
  const trimmed = line.trim();
  if (!trimmed.includes("|")) return null;
  if (/^[-*+]\s/.test(trimmed) || /^\d+[.)]\s/.test(trimmed)) return null;
  let body = trimmed;
  if (body.startsWith("|")) body = body.slice(1);
  if (body.endsWith("|")) body = body.slice(0, -1);
  return body.split("|").map((c) => c.trim());
}

export function isDelimiterRow(line: string): boolean {
  const cells = splitRow(line);
  if (!cells || !cells.length) return false;
  const real = cells.filter((c) => c.length > 0);
  return real.length > 0 && real.every((c) => DELIM_CELL.test(c.replace(/\s/g, "")));
}

function parseAlign(cell: string): TableAlign {
  const t = cell.replace(/\s/g, "");
  const left = t.startsWith(":");
  const right = t.endsWith(":");
  if (left && right) return "center";
  if (right) return "right";
  if (left) return "left";
  return null;
}

/**
 * If `lines[start]` is a header followed by a delimiter row, return the table
 * and how many lines it ate. Otherwise null.
 */
export function parseTableAt(
  lines: string[],
  start: number,
): { table: MdTable; consumed: number } | null {
  if (start + 1 >= lines.length) return null;
  const header = splitRow(lines[start] ?? "");
  if (!header || !header.length) return null;
  if (!isDelimiterRow(lines[start + 1] ?? "")) return null;
  const delim = splitRow(lines[start + 1] ?? "") ?? [];
  const cols = Math.max(header.length, delim.length);
  if (cols < 1) return null;
  const align = Array.from({ length: cols }, (_, i) =>
    delim[i] ? parseAlign(delim[i] ?? "") : null,
  );
  const rows: string[][] = [];
  let i = start + 2;
  while (i < lines.length) {
    const line = lines[i] ?? "";
    if (line.trim() === "") break;
    if (isDelimiterRow(line)) break;
    const cells = splitRow(line);
    if (!cells) break;
    rows.push(pad(cells, cols));
    i += 1;
  }
  return {
    table: { headers: pad(header, cols), align, rows },
    consumed: i - start,
  };
}
