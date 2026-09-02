import { Fragment, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { Check, Copy, Download } from "lucide-react";
import { downloadTextFile, parseFenceInfo } from "@/lib/chat/artifacts";
import { highlightCode, normalizeLang } from "@/lib/chat/highlight";
import {
  PYGMENTS_STYLES,
  SYNTAX_AUTO,
  useSyntaxPref,
} from "@/lib/chat/syntax";
import { parseTableAt, type MdTable, type TableAlign } from "@/lib/chat/md-table";
import { ChatImage } from "@/components/chat/chat-image";
import { cn } from "@/lib/utils";

type MdCtx = {
  jump: (id: string) => void;
  slug: (text: string) => string;
  notes: Map<string, string>;
};

function safeHref(raw: string): string | null {
  const url = raw.trim();
  if (!url) return null;
  if (/^(https?:\/\/|mailto:|\/|#)/i.test(url)) return url;
  return null;
}

function safeImg(raw: string): string | null {
  const url = raw.trim();
  if (!url) return null;
  if (/^https?:\/\//i.test(url) || url.startsWith("data:image/")) return url;
  return null;
}

/** `#intro` or `http://host/#intro` (path /) — stay inside this markdown. */
function hashTarget(href: string): string | null {
  const raw = href.trim();
  if (raw.startsWith("#") && raw.length > 1) {
    try {
      return decodeURIComponent(raw.slice(1));
    } catch {
      return raw.slice(1);
    }
  }
  try {
    const u = new URL(raw);
    if (!u.hash || u.hash.length < 2) return null;
    if ((u.pathname || "/") === "/") {
      try {
        return decodeURIComponent(u.hash.slice(1));
      } catch {
        return u.hash.slice(1);
      }
    }
  } catch {
    return null;
  }
  return null;
}

function githubSlug(text: string): string {
  return (
    text
      .toLowerCase()
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/!\[[^\]]*\]\([^)]+\)/g, "")
      .replace(/\[\^[^\]]+\]/g, "")
      .replace(/[`*_~]+/g, "")
      .replace(/[^\p{L}\p{N}\s-]/gu, "")
      .trim()
      .replace(/\s+/g, "-") || "section"
  );
}

function MdLink({
  href,
  title,
  ctx,
  children,
}: {
  href: string;
  title?: string;
  ctx: MdCtx;
  children: ReactNode;
}) {
  const hash = hashTarget(href);
  if (hash) {
    return (
      <a
        href={`#${hash}`}
        title={title}
        className="underline underline-offset-2 hover:text-foreground"
        onClick={(e: MouseEvent<HTMLAnchorElement>) => {
          e.preventDefault();
          ctx.jump(hash);
        }}
      >
        {children}
      </a>
    );
  }
  return (
    <a
      href={href}
      title={title}
      target="_blank"
      rel="noreferrer"
      className="underline underline-offset-2 hover:text-foreground"
    >
      {children}
    </a>
  );
}

function inline(text: string, ctx: MdCtx): ReactNode[] {
  const parts: ReactNode[] = [];
  const re =
    /(`[^`]+`)|(!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+"([^"]*)")?\s*\))|(\[\^([^\]]+)\])|(\[([^\]]+)\]\(\s*<?([^)\s>]+)>?(?:\s+"([^"]*)")?\s*\))|(\*\*[^*]+\*\*)|(\*[^*]+\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[1]) {
      parts.push(
        <code
          key={k++}
          className="rounded-xs bg-code px-1 py-0.5 font-mono text-xs"
        >
          {m[1].slice(1, -1)}
        </code>,
      );
    } else if (m[2]) {
      const src = safeImg(m[4] ?? "");
      const alt = m[3] ?? "";
      if (src) {
        parts.push(
          <ChatImage key={k++} src={src} alt={alt} name={m[5] || alt} />,
        );
      } else {
        parts.push(m[2]);
      }
    } else if (m[6]) {
      const id = m[7] ?? "";
      parts.push(
        <a
          key={k++}
          id={`fnref-${id}`}
          href={`#fn-${id}`}
          className="align-super text-[0.7em] text-muted-foreground hover:text-foreground"
          onClick={(e: MouseEvent<HTMLAnchorElement>) => {
            e.preventDefault();
            ctx.jump(`fn-${id}`);
          }}
        >
          [{id}]
        </a>,
      );
    } else if (m[8]) {
      const href = safeHref(m[10] ?? "");
      const label = m[9] ?? "";
      if (href) {
        parts.push(
          <MdLink key={k++} href={href} title={m[11] || undefined} ctx={ctx}>
            {label}
          </MdLink>,
        );
      } else {
        parts.push(m[8]);
      }
    } else if (m[12]) {
      parts.push(
        <strong key={k++} className="font-medium text-foreground">
          {m[12].slice(2, -2)}
        </strong>,
      );
    } else {
      parts.push(
        <em key={k++} className="italic">
          {(m[13] ?? "").slice(1, -1)}
        </em>,
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

const HEADING_CLASS = [
  "mt-4 mb-2 scroll-mt-2 text-xl font-semibold tracking-tight text-foreground first:mt-0",
  "mt-4 mb-2 scroll-mt-2 text-lg font-semibold text-foreground first:mt-0",
  "mt-3 mb-1.5 scroll-mt-2 text-base font-semibold text-foreground",
  "mt-3 mb-1 scroll-mt-2 text-sm font-semibold text-foreground",
  "mt-2 mb-1 scroll-mt-2 text-sm font-medium text-foreground",
  "mt-2 mb-1 scroll-mt-2 text-sm font-medium text-muted-foreground",
];

function renderProse(block: string, nodes: ReactNode[], ctx: MdCtx) {
  const lines = block.split("\n");
  let para: string[] = [];
  let list: string[] = [];
  let ordered: string[] = [];
  let tasks: { checked: boolean; text: string }[] = [];
  let quote: string[] = [];
  const flushPara = () => {
    if (!para.length) return;
    nodes.push(
      <p
        key={`p${nodes.length}`}
        className="my-2 whitespace-pre-wrap leading-relaxed"
      >
        {inline(para.join("\n"), ctx)}
      </p>,
    );
    para = [];
  };
  const flushList = () => {
    if (!list.length) return;
    nodes.push(
      <ul key={`l${nodes.length}`} className="my-2 list-disc space-y-1 pl-5">
        {list.map((item, idx) => (
          <li key={idx}>{inline(item, ctx)}</li>
        ))}
      </ul>,
    );
    list = [];
  };
  const flushOrdered = () => {
    if (!ordered.length) return;
    nodes.push(
      <ol key={`o${nodes.length}`} className="my-2 list-decimal space-y-1 pl-5">
        {ordered.map((item, idx) => (
          <li key={idx}>{inline(item, ctx)}</li>
        ))}
      </ol>,
    );
    ordered = [];
  };
  const flushTasks = () => {
    if (!tasks.length) return;
    nodes.push(
      <ul key={`k${nodes.length}`} className="my-2 space-y-1 pl-0">
        {tasks.map((item, idx) => (
          <li key={idx} className="flex items-start gap-2">
            <input
              type="checkbox"
              checked={item.checked}
              readOnly
              disabled
              className="mt-1 size-3.5 shrink-0"
            />
            <span
              className={item.checked ? "text-muted-foreground line-through" : ""}
            >
              {inline(item.text, ctx)}
            </span>
          </li>
        ))}
      </ul>,
    );
    tasks = [];
  };
  const flushQuote = () => {
    if (!quote.length) return;
    nodes.push(
      <blockquote
        key={`q${nodes.length}`}
        className="my-2 border-l-2 border-border pl-3 text-muted-foreground"
      >
        {inline(quote.join("\n"), ctx)}
      </blockquote>,
    );
    quote = [];
  };
  const flushAll = () => {
    flushPara();
    flushList();
    flushOrdered();
    flushTasks();
    flushQuote();
  };

  for (let i = 0; i < lines.length; i++) {
    const table = parseTableAt(lines, i);
    if (table) {
      flushAll();
      nodes.push(
        <MarkdownTable key={`t${nodes.length}`} table={table.table} ctx={ctx} />,
      );
      i += table.consumed - 1;
      continue;
    }
    const line = lines[i] ?? "";
    const note = line.match(/^\[\^([^\]]+)\]:\s*(.*)$/);
    if (note) {
      flushAll();
      ctx.notes.set(note[1], note[2] ?? "");
      continue;
    }
    const heading = line.match(/^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (heading) {
      flushAll();
      const level = Math.min(6, heading[1].length);
      const raw = heading[2] ?? "";
      const Tag = (`h${level}` as unknown) as "h1";
      nodes.push(
        <Tag
          key={`h${nodes.length}`}
          id={ctx.slug(raw)}
          className={HEADING_CLASS[level - 1]}
        >
          {inline(raw, ctx)}
        </Tag>,
      );
      continue;
    }
    if (/^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flushAll();
      nodes.push(
        <hr key={`r${nodes.length}`} className="my-4 border-border" />,
      );
      continue;
    }
    const quoted = line.match(/^ {0,3}>\s?(.*)$/);
    if (quoted) {
      flushPara();
      flushList();
      flushOrdered();
      flushTasks();
      quote.push(quoted[1] ?? "");
      continue;
    }
    const task = line.match(/^\s*(?:[-*+]\s+)?\[([ xX])\]\s+(.*)$/);
    if (task) {
      flushPara();
      flushList();
      flushOrdered();
      flushQuote();
      tasks.push({ checked: task[1] !== " ", text: task[2] ?? "" });
      continue;
    }
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    if (bullet) {
      flushPara();
      flushOrdered();
      flushTasks();
      flushQuote();
      list.push(bullet[1] ?? "");
      continue;
    }
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (numbered) {
      flushPara();
      flushList();
      flushTasks();
      flushQuote();
      ordered.push(numbered[1] ?? "");
      continue;
    }
    if (line.trim() === "") {
      flushAll();
      continue;
    }
    flushList();
    flushOrdered();
    flushTasks();
    flushQuote();
    para.push(line);
  }
  flushAll();
}

function alignClass(align: TableAlign | undefined): string {
  if (align === "center") return "text-center";
  if (align === "right") return "text-right";
  return "text-left";
}

function MarkdownTable({ table, ctx }: { table: MdTable; ctx: MdCtx }) {
  return (
    <div className="my-3 overflow-x-auto rounded-md outline outline-1 -outline-offset-1 outline-white/10">
      <table className="w-full min-w-max border-collapse text-sm">
        <thead>
          <tr className="border-b border-border bg-secondary/70">
            {table.headers.map((h, i) => (
              <th
                key={i}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium tracking-wide text-foreground",
                  alignClass(table.align[i]),
                )}
              >
                {inline(h, ctx)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, r) => (
            <tr
              key={r}
              className="border-b border-border/50 last:border-0"
            >
              {row.map((cell, c) => (
                <td
                  key={c}
                  className={cn(
                    "px-3 py-1.5 text-foreground/85",
                    alignClass(table.align[c]),
                  )}
                >
                  {inline(cell, ctx)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CodeBlock({
  lang,
  body,
  file,
}: {
  lang: string;
  body: string;
  file: string | null;
}) {
  const name = normalizeLang(lang);
  const [copied, setCopied] = useState(false);
  const [syntax, setSyntax] = useSyntaxPref();
  const label = file || (name && name !== "text" ? name : "code");

  async function copy() {
    try {
      await navigator.clipboard.writeText(body);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = body;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  return (
    <div className="my-3 overflow-hidden rounded-md bg-code text-code-fg outline outline-1 -outline-offset-1 outline-foreground/10">
      <div className="flex items-center justify-between gap-2 border-b border-border px-2 py-0.5">
        <span className="min-w-0 truncate px-1 font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
        <div className="flex items-center">
          <select
            aria-label="Syntax highlighting theme"
            value={syntax}
            onChange={(e) => setSyntax(e.target.value)}
            className="mr-0.5 h-8 max-w-[8.5rem] cursor-pointer truncate rounded-sm bg-transparent px-1 font-mono text-[10px] text-muted-foreground outline-none hover:text-foreground"
          >
            <option value={SYNTAX_AUTO}>auto</option>
            {PYGMENTS_STYLES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id}
              </option>
            ))}
          </select>
          {file && (
            <button
              type="button"
              aria-label={`Download ${file}`}
              className="inline-flex size-8 items-center justify-center rounded-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              onClick={() => downloadTextFile(file, body)}
            >
              <Download className="size-3.5" />
            </button>
          )}
          <button
            type="button"
            aria-label={copied ? "Copied" : "Copy code"}
            className="inline-flex size-8 items-center justify-center rounded-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            onClick={() => void copy()}
          >
            {copied ? (
              <Check className="size-3.5" />
            ) : (
              <Copy className="size-3.5" />
            )}
          </button>
        </div>
      </div>
      <pre className="overflow-x-auto p-3 font-mono text-xs leading-relaxed">
        <code
          dangerouslySetInnerHTML={{ __html: highlightCode(body, name) }}
        />
      </pre>
    </div>
  );
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const OPEN_FENCE = /^( {0,3})(`{3,}|~{3,})([^\n]*)$/;

export function Markdown({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  const source = typeof text === "string" ? text : String(text ?? "");
  const lines = source.split("\n");
  const nodes: ReactNode[] = [];
  let prose: string[] = [];
  let i = 0;
  let prevLine = "";
  const rootRef = useRef<HTMLDivElement>(null);
  const seen = new Map<string, number>();
  const notes = new Map<string, string>();
  const ctx: MdCtx = {
    notes,
    slug(text: string) {
      const base = githubSlug(text);
      const n = seen.get(base) ?? 0;
      seen.set(base, n + 1);
      return n ? `${base}-${n}` : base;
    },
    jump(id: string) {
      const root = rootRef.current;
      if (!root) return;
      const el = root.querySelector(`#${CSS.escape(id)}`);
      el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    },
  };

  const flushProse = () => {
    if (!prose.length) return;
    prevLine = prose[prose.length - 1] ?? "";
    renderProse(prose.join("\n"), nodes, ctx);
    prose = [];
  };

  while (i < lines.length) {
    const open = lines[i].match(OPEN_FENCE);
    if (open) {
      flushProse();
      const marker = open[2];
      const meta = parseFenceInfo(open[3] ?? "", prevLine);
      const close = new RegExp(`^ {0,3}${escapeRe(marker)}[ \\t]*$`);
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !close.test(lines[i])) {
        body.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      nodes.push(
        <CodeBlock
          key={`c${nodes.length}`}
          lang={meta.lang}
          file={meta.file}
          body={body.join("\n")}
        />,
      );
      prevLine = "";
      continue;
    }
    prose.push(lines[i]);
    i += 1;
  }
  flushProse();

  if (notes.size) {
    nodes.push(<hr key="fn-rule" className="my-4 border-border" />);
    nodes.push(
      <ol
        key="fn-list"
        className="my-2 list-decimal space-y-1 pl-5 text-xs text-muted-foreground"
      >
        {[...notes.entries()].map(([id, body]) => (
          <li key={id} id={`fn-${id}`} className="scroll-mt-2">
            {inline(body, ctx)}{" "}
            <a
              href={`#fnref-${id}`}
              className="text-muted-foreground hover:text-foreground"
              onClick={(e: MouseEvent<HTMLAnchorElement>) => {
                e.preventDefault();
                ctx.jump(`fnref-${id}`);
              }}
            >
              ↩
            </a>
          </li>
        ))}
      </ol>,
    );
  }

  return (
    <div
      ref={rootRef}
      className={cn("text-pretty text-sm text-foreground/90", className)}
    >
      {nodes.length ? nodes : <Fragment>{inline(source, ctx)}</Fragment>}
    </div>
  );
}
