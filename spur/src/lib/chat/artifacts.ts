export type Artifact = {
  file: string;
  lang: string;
  text: string;
};

export type LivedArtifact = Artifact & {
  /** Turns left in the Downloadable Files window (1–ARTIFACT_TTL). */
  remaining: number;
};

export const ARTIFACT_TTL = 4;

const FILE_TOKEN = /^(?:\.\/)?[\w.@+-]+(?:\/[\w.@+-]+)*\.[A-Za-z0-9]{1,8}$/;

export function isFilename(raw: string): boolean {
  const token = raw.trim().replace(/^["'`]+|["'`]+$/g, "");
  if (!token || token.length > 180 || /\s/.test(token)) return false;
  return FILE_TOKEN.test(token);
}

export function parseFenceInfo(
  info: string,
  prevLine = "",
): { lang: string; file: string | null } {
  const raw = info.trim();
  let file: string | null = null;
  let lang = "";

  const named = raw.match(
    /(?:filename|file|title|path)\s*[:=]\s*["']?([^\s"']+)/i,
  );
  if (named && isFilename(named[1] ?? "")) file = stripName(named[1] ?? "");

  if (!file) {
    const colon = raw.match(/^([A-Za-z0-9_+-]+)\s*:\s*(\S+)$/);
    if (colon && isFilename(colon[2] ?? "")) {
      lang = colon[1] ?? "";
      file = stripName(colon[2] ?? "");
    }
  }

  if (!file || !lang) {
    const parts = raw.split(/\s+/).filter(Boolean);
    if (!file) {
      const hit = parts.find((p) => isFilename(p));
      if (hit) file = stripName(hit);
    }
    if (!lang) {
      const first = parts[0];
      if (first && !isFilename(first)) lang = first;
    }
  }

  if (!file && prevLine) {
    const heading = prevLine
      .trim()
      .replace(/^#{1,6}\s+/, "")
      .replace(/^\*{1,2}(.+?)\*{1,2}$/, "$1")
      .replace(/^`(.+?)`$/, "$1")
      .replace(/^(?:file|filename)\s*:\s*/i, "");
    if (isFilename(heading)) file = stripName(heading);
  }

  if (!lang && file) {
    lang = file.split(".").pop() ?? "";
  }
  return { lang, file };
}

function stripName(value: string): string {
  return value.trim().replace(/^["'`]+|["'`]+$/g, "");
}

const LANG_EXT: Record<string, string> = {
  python: "py",
  py: "py",
  python3: "py",
  javascript: "js",
  js: "js",
  jsx: "jsx",
  typescript: "ts",
  ts: "ts",
  tsx: "tsx",
  rust: "rs",
  rs: "rs",
  go: "go",
  golang: "go",
  ruby: "rb",
  rb: "rb",
  bash: "sh",
  sh: "sh",
  shell: "sh",
  zsh: "zsh",
  json: "json",
  yaml: "yml",
  yml: "yml",
  html: "html",
  css: "css",
  sql: "sql",
  c: "c",
  cpp: "cpp",
  "c++": "cpp",
  java: "java",
  kt: "kt",
  kotlin: "kt",
  swift: "swift",
  php: "php",
  r: "r",
  lua: "lua",
  toml: "toml",
  xml: "xml",
  md: "md",
  markdown: "md",
  text: "txt",
  txt: "txt",
};

function nameFromBody(body: string): string | null {
  const first = (body.split("\n")[0] ?? "").trim();
  const hit = first.match(/^(?:#|\/\/|--)\s*([^\s]+\.[A-Za-z0-9]{1,8})\s*$/);
  if (hit && isFilename(hit[1] ?? "")) return stripName(hit[1] ?? "");
  return null;
}

function looksLikeFile(body: string): boolean {
  const lines = body.split("\n").filter((l) => l.trim());
  if (lines.length >= 4) return true;
  if (/^#!/m.test(body)) return true;
  return /\b(def |class |function |fn |pub fn |package |fn main)/m.test(body);
}

function fallbackName(lang: string, body: string, used: Set<string>): string | null {
  const fromBody = nameFromBody(body);
  if (fromBody) return uniqueName(fromBody, used);
  if (!looksLikeFile(body)) return null;
  const ext = LANG_EXT[lang.toLowerCase()];
  if (!ext) return null;
  return uniqueName(`untitled.${ext}`, used);
}

function uniqueName(base: string, used: Set<string>): string {
  if (!used.has(base.toLowerCase())) return base;
  const dot = base.lastIndexOf(".");
  const stem = dot > 0 ? base.slice(0, dot) : base;
  const ext = dot > 0 ? base.slice(dot) : "";
  let i = 2;
  let next = `${stem}-${i}${ext}`;
  while (used.has(next.toLowerCase())) {
    i += 1;
    next = `${stem}-${i}${ext}`;
  }
  return next;
}

const OPEN_FENCE = /^( {0,3})(`{3,}|~{3,})([^\n]*)$/;

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function extractArtifacts(text: string): Artifact[] {
  const source = typeof text === "string" ? text : String(text ?? "");
  const lines = source.split("\n");
  const out: Artifact[] = [];
  const used = new Set<string>();
  let i = 0;
  let prev = "";
  while (i < lines.length) {
    const open = lines[i].match(OPEN_FENCE);
    if (open) {
      const marker = open[2] ?? "```";
      const meta = parseFenceInfo(open[3] ?? "", prev);
      const close = new RegExp(`^ {0,3}${escapeRe(marker)}[ \\t]*$`);
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !close.test(lines[i])) {
        body.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      const code = body.join("\n").replace(/\n$/, "");
      const file = meta.file || fallbackName(meta.lang, code, used);
      if (file && code.trim()) {
        used.add(file.toLowerCase());
        out.push({ file, lang: meta.lang, text: code });
      }
      prev = "";
      continue;
    }
    prev = lines[i];
    i += 1;
  }
  return out;
}

export function artifactsFromMessages(
  messages: { role: string; content: string }[],
): LivedArtifact[] {
  const currentTurn = messages.filter((m) => m.role === "user").length;
  const windowStart = Math.max(1, currentTurn - ARTIFACT_TTL + 1);
  const byName = new Map<string, LivedArtifact>();
  let turn = 0;
  for (const m of messages) {
    if (m.role === "user") turn += 1;
    if (m.role !== "assistant" || !m.content) continue;
    if (turn < windowStart) continue;
    const remaining = ARTIFACT_TTL - (currentTurn - turn);
    for (const art of extractArtifacts(m.content)) {
      byName.set(art.file, { ...art, remaining });
    }
  }
  return [...byName.values()];
}

export function downloadTextFile(name: string, text: string) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name.split("/").pop() || name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
