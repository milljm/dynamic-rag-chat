export type SlashCommand =
  | "help"
  | "turn"
  | "branch"
  | "regenerate"
  | "delete-last"
  | "rewind"
  | "reset"
  | "dbranch"
  | "history";

export type SlashResult =
  | { kind: "command"; command: SlashCommand; args: string }
  | {
      kind: "inline";
      agent: boolean;
      noContext: boolean;
      text: string;
      rare: string[];
      ooc: boolean;
    }
  | { kind: "include"; branch: string; text: string; rare: string[]; ooc: boolean }
  | { kind: "message"; text: string; rare: string[]; ooc: boolean };

const CMD_RE =
  /^[ \t]*\\(?<cmd>[A-Za-z0-9_\-\?]+)(?:[ \t]+(?<args>.*))?$/;

const LOCAL_COMMANDS = new Set<string>([
  "help",
  "?",
  "turn",
  "branch",
  "regenerate",
  "delete-last",
  "rewind",
  "reset",
  "dbranch",
  "history",
]);

export const RARE_TOKENS = [
  "[RARE NOW]",
  "[RARE USED]",
  "[RARE RESET]",
  "[SAFE MODE]",
] as const;

const RARE_RE = new RegExp(
  RARE_TOKENS.map((t) => t.replace(/[[\]]/g, "\\$&")).join("|"),
  "g",
);

const OOC_RE = /^\s*(?:OOC:|SYSTEM:|OOC>)/i;

export function stripRare(raw: string): { text: string; rare: string[] } {
  const rare = raw.match(RARE_RE) ?? [];
  const text = rare.length ? raw.replace(RARE_RE, "").replace(/\s+/g, " ").trim() : raw;
  return { text, rare };
}

export function isOoc(text: string): boolean {
  return OOC_RE.test(text);
}

export function parseComposerInput(raw: string): SlashResult {
  const trimmed = raw.trim();
  const lines = trimmed.split(/\n/);
  const first = lines[0] ?? "";
  const matched = first.match(CMD_RE);
  if (!matched?.groups) {
    const { text, rare } = stripRare(raw);
    return { kind: "message", text, rare, ooc: isOoc(text) };
  }

  const cmd = matched.groups.cmd.toLowerCase();
  const argLine = (matched.groups.args || "").trim();
  const rest = lines.slice(1).join("\n").trim();
  const body = [argLine, rest].filter(Boolean).join("\n");

  if (cmd === "agent" || cmd === "no-context") {
    const { text, rare } = stripRare(body);
    return {
      kind: "inline",
      agent: cmd === "agent",
      noContext: cmd === "no-context",
      text,
      rare,
      ooc: isOoc(text),
    };
  }

  if (cmd === "include") {
    const [branch, ...msg] = body.split(/\s+/);
    const joined = msg.join(" ").trim();
    const { text, rare } = stripRare(joined);
    return {
      kind: "include",
      branch: (branch || "").trim(),
      text,
      rare,
      ooc: isOoc(text),
    };
  }

  if (LOCAL_COMMANDS.has(cmd)) {
    const command = (cmd === "?" ? "help" : cmd) as SlashCommand;
    return { kind: "command", command, args: body };
  }

  const { text, rare } = stripRare(raw);
  return { kind: "message", text, rare, ooc: isOoc(text) };
}

export function parseIncludes(raw: string): {
  urls: string[];
  paths: string[];
} {
  const urls: string[] = [];
  const paths: string[] = [];
  const re = /\{\{\s*([^}]+?)\s*\}\}/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(raw))) {
    const inner = match[1].trim();
    if (/^https?:\/\//i.test(inner)) urls.push(inner);
    else if (inner.startsWith("/") || inner.startsWith("file:")) {
      paths.push(inner);
    }
  }
  return { urls, paths };
}

export const SLASH_HELP = [
  { cmd: "\\regenerate", hint: "Redo the last assistant reply" },
  { cmd: "\\reset", hint: "Clear history and retrieval for this branch" },
  { cmd: "\\delete-last", hint: "Drop the last user/assistant turn" },
  { cmd: "\\rewind N", hint: "Keep only the first N turns" },
  { cmd: "\\agent msg", hint: "Force web search, then answer" },
  { cmd: "\\no-context msg", hint: "Skip retrieval this turn" },
  { cmd: "\\include NAME msg", hint: "Attach another branch as context" },
  { cmd: "\\history N", hint: "Show the last N user inputs" },
  { cmd: "\\branch NAME@N", hint: "Fork a branch, optional cut at turn N" },
  { cmd: "\\dbranch NAME", hint: "Delete a non-active branch" },
  { cmd: "\\turn", hint: "Show the current turn count" },
] as const;
