const KEYWORDS: Record<string, string[]> = {
  python: [
    "and","as","assert","async","await","break","class","continue","def","del",
    "elif","else","except","False","finally","for","from","global","if","import",
    "in","is","lambda","None","nonlocal","not","or","pass","raise","return",
    "True","try","while","with","yield",
  ],
  javascript: [
    "async","await","break","case","catch","class","const","continue","debugger",
    "default","delete","else","export","extends","false","finally","for","from",
    "function","if","import","in","instanceof","let","new","null","of","return",
    "static","super","switch","this","throw","true","try","typeof","undefined",
    "var","void","while","yield",
  ],
  typescript: [
    "as","async","await","break","case","catch","class","const","continue",
    "debugger","default","else","enum","export","extends","false","finally",
    "for","from","function","if","implements","import","in","interface",
    "let","new","null","of","private","protected","public","return","static",
    "super","switch","this","throw","true","try","type","typeof","undefined",
    "var","void","while","yield",
  ],
  bash: [
    "alias","break","case","do","done","elif","else","esac","export","fi","for",
    "function","if","in","local","return","then","until","while",
  ],
  rust: [
    "as","async","await","break","const","continue","crate","else","enum","extern",
    "false","fn","for","if","impl","in","let","loop","match","mod","move","mut",
    "pub","ref","return","self","Self","static","struct","super","trait","true",
    "type","unsafe","use","where","while",
  ],
  go: [
    "break","case","chan","const","continue","default","defer","else","fallthrough",
    "false","for","func","go","goto","if","import","interface","map","nil",
    "package","range","return","select","struct","switch","true","type","var",
  ],
  json: ["true","false","null"],
  sql: [
    "and","as","asc","by","create","delete","desc","from","group","having",
    "insert","into","join","left","limit","not","null","on","or","order",
    "right","select","set","table","update","values","where",
  ],
};

const ALIAS: Record<string, string> = {
  js: "javascript",
  jsx: "javascript",
  ts: "typescript",
  tsx: "typescript",
  py: "python",
  sh: "bash",
  shell: "bash",
  zsh: "bash",
  rs: "rust",
  yml: "yaml",
};

export function normalizeLang(raw: string): string {
  const key = raw.trim().toLowerCase();
  return ALIAS[key] || key || "text";
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&")
    .replace(/</g, "<")
    .replace(/>/g, ">");
}

export function highlightCode(code: string, langRaw: string): string {
  const lang = normalizeLang(langRaw);
  const escaped = escapeHtml(code);
  const words = KEYWORDS[lang];
  const comment =
    lang === "python" || lang === "bash" || lang === "yaml"
      ? /(#.*)$/gm
      : lang === "sql"
        ? /(--.*)$/gm
        : /(\/\/.*$|\/\*[\s\S]*?\*\/)/gm;
  const stringRe = /(`(?:\\.|[^`\\])*`|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/g;

  const tokens: { start: number; end: number; cls: string; text: string }[] = [];
  const take = (re: RegExp, cls: string) => {
    re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(escaped))) {
      tokens.push({
        start: m.index,
        end: m.index + m[0].length,
        cls,
        text: m[0],
      });
    }
  };
  take(comment, "cm");
  take(stringRe, "st");
  take(/\b\d+(?:\.\d+)?\b/g, "nu");
  if (words) {
    take(new RegExp(`\\b(?:${words.join("|")})\\b`, "g"), "kw");
  }
  const fnRe = /\b(?:def|fn|func|function|class)\s+([A-Za-z_][\w]*)/g;
  fnRe.lastIndex = 0;
  let fm: RegExpExecArray | null;
  while ((fm = fnRe.exec(escaped))) {
    const name = fm[1];
    const start = fm.index + fm[0].lastIndexOf(name);
    tokens.push({
      start,
      end: start + name.length,
      cls: "fn",
      text: name,
    });
  }
  take(/\b([A-Za-z_][\w]*)(?=\s*\()/g, "fn");

  tokens.sort((a, b) => a.start - b.start || b.end - a.end);
  const kept: typeof tokens = [];
  let cursor = 0;
  for (const t of tokens) {
    if (t.start < cursor) continue;
    kept.push(t);
    cursor = t.end;
  }

  let out = "";
  let i = 0;
  for (const t of kept) {
    out += escaped.slice(i, t.start);
    out += `<span class="hx-${t.cls}">${t.text}</span>`;
    i = t.end;
  }
  out += escaped.slice(i);
  return out;
}
