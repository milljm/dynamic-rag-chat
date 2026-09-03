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
      .replace(/<[^>]+>/g, "")
      .replace(/[`*_~]+/g, "")
      .replace(/[^\p{L}\p{N}\s-]/gu, "")
      .trim()
      .replace(/\s+/g, "-") || "section"
  );
}

function attrOf(attrs: string, name: string): string | undefined {
  return attrs.match(new RegExp(`(?:^|\\s)${name}\\s*=\\s*["']([^"']+)["']`, "i"))?.[1];
}

function parseHtmlA(html: string): { href?: string; name?: string; id?: string; inner: string } | null {
  const m = html.match(/^<a\s+([^>]*?)(?:\/>|>([\s\S]*)<\/a>)\s*$/i);
  if (!m) return null;
  return {
    href: attrOf(m[1], "href"),
    name: attrOf(m[1], "name"),
    id: attrOf(m[1], "id"),
    inner: (m[2] ?? "").trim(),
  };
}

function takeNamedIds(line: string): { text: string; ids: string[] } {
  const ids: string[] = [];
  const text = line
    .replace(/<a\s+([^>]*?)(?:\/>|><\/a>|>\s*<\/a>)/gi, (full, attrs: string) => {
      if (attrOf(attrs, "href")) return full;
      const id = attrOf(attrs, "name") || attrOf(attrs, "id");
      if (id) {
        ids.push(id);
        return "";
      }
      return full;
    })
    .replace(/[ \t]+$/g, "")
    .trimEnd();
  return { text, ids };
}

const KEYCAP = /^((?:[0-9]\uFE0F?\u20E3)|🔟)\s+(.+)$/u;

const SHORTCODES: Record<string, string> = {
  smile: "😄",
  grinning: "😀",
  rocket: "🚀",
  "+1": "👍",
  "-1": "👎",
  thumbsup: "👍",
  thumbsdown: "👎",
  heart: "❤️",
  fire: "🔥",
  tada: "🎉",
  check: "✅",
  x: "❌",
  warning: "⚠️",
  eyes: "👀",
  think: "🤔",
  thinking: "🤔",
  wave: "👋",
  star: "⭐",
  sparkles: "✨",
  zap: "⚡",
  bulb: "💡",
  laugh: "😂",
  wink: "😉",
  cry: "😢",
  pray: "🙏",
  clap: "👏",
  ok: "👌",
};

const SAFE_COLOR = /^(?:[a-z]+|#(?:[0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})|rgb\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*\))$/i;


function decodeEntities(s: string): string {
  const amp = "\u0026";
  return s
    .replace(new RegExp(amp + "nbsp;", "gi"), "\u00a0")
    .replace(new RegExp(amp + "amp;", "gi"), amp)
    .replace(new RegExp(amp + "lt;", "gi"), "<")
    .replace(new RegExp(amp + "gt;", "gi"), ">")
    .replace(new RegExp(amp + "quot;", "gi"), '"')
    .replace(new RegExp(amp + "apos;", "gi"), "'")
    .replace(new RegExp(amp + "#39;", "gi"), "'")
    .replace(new RegExp(amp + "mdash;", "gi"), "\u2014")
    .replace(new RegExp(amp + "ndash;", "gi"), "\u2013")
    .replace(new RegExp(amp + "hellip;", "gi"), "\u2026")
    .replace(new RegExp(amp + "#x([0-9a-f]+);", "gi"), (_, h) => {
      const n = Number.parseInt(h, 16);
      return Number.isFinite(n) ? String.fromCodePoint(n) : _;
    })
    .replace(new RegExp(amp + "#(\\d+);", "g"), (_, d) => {
      const n = Number(d);
      return Number.isFinite(n) ? String.fromCodePoint(n) : _;
    });
}

function expandShortcodes(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const re = /:([a-z0-9_+-]+):/gi;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const glyph = SHORTCODES[m[1].toLowerCase()];
    parts.push(glyph ?? m[0]);
    last = m.index + m[0].length;
    k += 1;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.length ? parts : [text];
}

type ListItem = {
  indent: number;
  ordered: boolean;
  task: boolean;
  checked: boolean;
  text: string;
};

function indentOf(line: string): number {
  let n = 0;
  for (const ch of line) {
    if (ch === " " || ch === "\u00a0") n += 1;
    else if (ch === "\t") n += 4;
    else break;
  }
  return n;
}

function renderListTree(items: ListItem[], ctx: MdCtx, key: number): ReactNode {
  function walk(start: number, indent: number): { node: ReactNode; next: number } {
    const ordered = items[start]?.ordered ?? false;
    const tasky = items[start]?.task ?? false;
    const lis: ReactNode[] = [];
    let i = start;
    while (i < items.length) {
      const it = items[i];
      if (it.indent < indent) break;
      if (it.indent > indent) break;
      if (it.ordered !== ordered || it.task !== tasky) break;
      let next = i + 1;
      let nested: ReactNode = null;
      if (next < items.length && items[next].indent > indent) {
        const child = walk(next, items[next].indent);
        nested = child.node;
        next = child.next;
      }
      lis.push(
        <li key={i} className={it.task ? "flex items-start gap-2" : undefined}>
          {it.task ? (
            <>
              <input type="checkbox" checked={it.checked} readOnly disabled className="mt-1 size-3.5 shrink-0" />
              <span className={it.checked ? "text-muted-foreground line-through" : ""}>
                {inline(it.text, ctx)}
                {nested}
              </span>
            </>
          ) : (
            <>
              {inline(it.text, ctx)}
              {nested}
            </>
          )}
        </li>,
      );
      i = next;
    }
    const Tag = ordered ? "ol" : "ul";
    return {
      node: (
        <Tag
          key={`${key}-${start}`}
          className={
            tasky ? "my-2 space-y-1 pl-0" : ordered ? "my-2 list-decimal space-y-1 pl-5" : "my-2 list-disc space-y-1 pl-5"
          }
        >
          {lis}
        </Tag>
      ),
      next: i,
    };
  }
  const out: ReactNode[] = [];
  let i = 0;
  while (i < items.length) {
    const piece = walk(i, items[i].indent);
    out.push(piece.node);
    i = piece.next;
  }
  return <div key={key}>{out}</div>;
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

function htmlAnchor(html: string, ctx: MdCtx, key: number): ReactNode {
  const parsed = parseHtmlA(html.trim());
  if (!parsed) return html;
  const id = parsed.name || parsed.id;
  if (parsed.href) {
    const href = safeHref(parsed.href);
    if (!href) return parsed.inner || html;
    const link = (
      <MdLink key={key} href={href} ctx={ctx}>
        {parsed.inner || href}
      </MdLink>
    );
    return id ? (
      <span id={id} className="scroll-mt-2">
        {link}
      </span>
    ) : (
      link
    );
  }
  if (id) {
    return parsed.inner ? (
      <span id={id} className="scroll-mt-2">
        {parsed.inner}
      </span>
    ) : (
      <span id={id} className="scroll-mt-2" />
    );
  }
  return parsed.inner || html;
}


function parseSafeStyle(raw: string): { color?: string; backgroundColor?: string } | null {
  const out: { color?: string; backgroundColor?: string } = {};
  for (const part of raw.split(";")) {
    const cut = part.indexOf(":");
    if (cut < 0) continue;
    const key = part.slice(0, cut).trim().toLowerCase();
    const val = part.slice(cut + 1).trim();
    if (!val || /url\s*\(|expression|javascript/i.test(val)) continue;
    if (!SAFE_COLOR.test(val)) continue;
    if (key === "color") out.color = val;
    if (key === "background" || key === "background-color") out.backgroundColor = val;
  }
  return out.color || out.backgroundColor ? out : null;
}

function htmlInline(html: string, ctx: MdCtx, key: number): ReactNode {
  if (/^<br\s*\/?>$/i.test(html.trim())) return <br key={key} />;
  const span = html.trim().match(/^<span\s+([^>]*)>([\s\S]*)<\/span>$/i);
  if (span) {
    const style = parseSafeStyle(attrOf(span[1] ?? "", "style") || "");
    const inner = inlineMd(span[2] ?? "", ctx, key + 1);
    if (style) {
      return (
        <span key={key} style={style}>
          {inner}
        </span>
      );
    }
    return <Fragment key={key}>{inner}</Fragment>;
  }
  return htmlAnchor(html, ctx, key);
}

function inline(text: string, ctx: MdCtx): ReactNode[] {
  const source = decodeEntities(text);
  const parts: ReactNode[] = [];
  const htmlRe =
    /(<a\s+[^>]*?\/>|<a\s+[^>]*?>[\s\S]*?<\/a>|<span\s+[^>]*>[\s\S]*?<\/span>|<br\s*\/?>)/gi;
  let last = 0;
  let k = 0;
  let tag: RegExpExecArray | null;
  while ((tag = htmlRe.exec(source))) {
    if (tag.index > last) parts.push(...inlineMd(source.slice(last, tag.index), ctx, k));
    k += 32;
    parts.push(htmlInline(tag[0], ctx, k++));
    last = tag.index + tag[0].length;
  }
  if (last < source.length) parts.push(...inlineMd(source.slice(last), ctx, k));
  return parts;
}

function inlineMd(text: string, ctx: MdCtx, seed: number): ReactNode[] {
  const parts: ReactNode[] = [];
  const re =
    /(`[^`]+`)|(!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+"([^"]*)")?\s*\))|(\[\^([^\]]+)\])|(\[([^\]]+)\]\(\s*<?([^)\s>]+)>?(?:\s+"([^"]*)")?\s*\))|(\*\*[^*]+\*\*)|(~~[^~]+~~)|(\*[^*]+\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = seed;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(...expandShortcodes(text.slice(last, m.index)));
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
    } else if (m[13]) {
      parts.push(
        <del key={k++} className="text-muted-foreground">
          {m[13].slice(2, -2)}
        </del>,
      );
    } else {
      parts.push(
        <em key={k++} className="italic">
          {(m[14] ?? "").slice(1, -1)}
        </em>,
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(...expandShortcodes(text.slice(last)));
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
  const capTotal = new Map<string, number>();
  const capSeen = new Map<string, number>();
  for (const raw of lines) {
    const peeled = takeNamedIds(decodeEntities(raw));
    const cap = peeled.text.match(KEYCAP);
    if (cap) {
      const slug = githubSlug(cap[2] ?? "");
      capTotal.set(slug, (capTotal.get(slug) ?? 0) + 1);
    }
  }
  let para: string[] = [];
  let listItems: ListItem[] = [];
  let defs: { term: string; defs: string[] }[] = [];
  let quote: string[] = [];
  const flushPara = () => {
    if (!para.length) return;
    nodes.push(
      <p key={`p${nodes.length}`} className="my-2 whitespace-pre-wrap leading-relaxed">
        {inline(para.join("\n"), ctx)}
      </p>,
    );
    para = [];
  };
  const flushLists = () => {
    if (!listItems.length) return;
    nodes.push(renderListTree(listItems, ctx, nodes.length));
    listItems = [];
  };
  const flushDefs = () => {
    if (!defs.length) return;
    nodes.push(
      <dl key={`d${nodes.length}`} className="my-2 space-y-1">
        {defs.map((row, idx) => (
          <Fragment key={idx}>
            <dt className="font-medium text-foreground">{inline(row.term, ctx)}</dt>
            {row.defs.map((dd, j) => (
              <dd key={j} className="mb-2 ml-4 text-muted-foreground">
                {inline(dd, ctx)}
              </dd>
            ))}
          </Fragment>
        ))}
      </dl>,
    );
    defs = [];
  };
  const flushQuote = () => {
    if (!quote.length) return;
    nodes.push(
      <blockquote key={`q${nodes.length}`} className="my-2 border-l-2 border-border pl-3 text-muted-foreground">
        {inline(quote.join("\n"), ctx)}
      </blockquote>,
    );
    quote = [];
  };
  const flushAll = () => {
    flushPara();
    flushLists();
    flushDefs();
    flushQuote();
  };

  for (let i = 0; i < lines.length; i++) {
    const table = parseTableAt(lines, i);
    if (table) {
      flushAll();
      nodes.push(<MarkdownTable key={`t${nodes.length}`} table={table.table} ctx={ctx} />);
      i += table.consumed - 1;
      continue;
    }
    const peeled = takeNamedIds(decodeEntities(lines[i] ?? ""));
    const line = peeled.text;
    const extraId = peeled.ids[0];
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
        <Tag key={`h${nodes.length}`} id={extraId || ctx.slug(raw)} className={HEADING_CLASS[level - 1]}>
          {inline(raw, ctx)}
        </Tag>,
      );
      continue;
    }
    if (/^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flushAll();
      nodes.push(<hr key={`r${nodes.length}`} className="my-4 border-border" />);
      continue;
    }
    const quoted = line.match(/^ {0,3}>\s?(.*)$/);
    if (quoted) {
      flushPara();
      flushLists();
      flushDefs();
      quote.push(quoted[1] ?? "");
      continue;
    }
    const dd = line.match(/^:\s+(.*)$/);
    if (dd) {
      flushLists();
      flushQuote();
      if (para.length) {
        const term = para.pop() ?? "";
        flushPara();
        defs.push({ term, defs: [dd[1] ?? ""] });
      } else if (defs.length) {
        defs[defs.length - 1].defs.push(dd[1] ?? "");
      } else {
        para.push(line);
      }
      continue;
    }
    const listLine = line.match(
      /^([ \t\u00a0]*)(?:([-*+])\s+(?:\[([ xX])\]\s+)?(.*)|(\d+)[.)]\s+(.*))$/,
    );
    if (listLine && !/^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flushPara();
      flushDefs();
      flushQuote();
      const taskMark = listLine[3];
      const bulletText = listLine[2] != null ? (listLine[4] ?? "") : (listLine[6] ?? "");
      listItems.push({
        indent: indentOf(line),
        ordered: listLine[5] != null,
        task: taskMark != null,
        checked: Boolean(taskMark && taskMark !== " "),
        text: bulletText,
      });
      continue;
    }
    if (line.trim() === "") {
      flushAll();
      continue;
    }
    const keycap = line.match(KEYCAP);
    if (keycap) {
      flushAll();
      const title = keycap[2] ?? "";
      const slug = extraId || githubSlug(title);
      const n = capSeen.get(slug) ?? 0;
      capSeen.set(slug, n + 1);
      const toc = !extraId && (capTotal.get(githubSlug(title)) ?? 0) > 1 && n === 0;
      nodes.push(
        <p
          key={`p${nodes.length}`}
          id={toc ? undefined : slug}
          className="my-2 scroll-mt-2 whitespace-pre-wrap leading-relaxed"
        >
          {toc ? (
            <MdLink href={`#${slug}`} ctx={ctx}>
              {line}
            </MdLink>
          ) : (
            inline(line, ctx)
          )}
        </p>,
      );
      continue;
    }
    if (extraId) {
      flushAll();
      nodes.push(
        <p
          key={`p${nodes.length}`}
          id={extraId}
          className="my-2 scroll-mt-2 whitespace-pre-wrap leading-relaxed"
        >
          {inline(line, ctx)}
        </p>,
      );
      continue;
    }
    flushLists();
    flushDefs();
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
      const safe = CSS.escape(id);
      let el: Element | null = root.querySelector(`#${safe}`) || root.querySelector(`[name="${safe}"]`);
      if (!el) {
        const want = id.toLowerCase();
        el =
          [...root.querySelectorAll("h1,h2,h3,h4,h5,h6,p,[id]")].find((n) => {
            const nid = n.getAttribute("id") || "";
            if (nid === id) return true;
            return githubSlug(n.textContent || "") === want;
          }) ?? null;
      }
      el?.scrollIntoView({ behavior: "smooth", block: "start" });
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
