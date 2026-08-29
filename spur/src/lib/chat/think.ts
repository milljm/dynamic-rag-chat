const NS_START = /<\s*mm:(think|thinking|reasoning)\s*>/i;
const NS_END = /<\/\s*mm:(think|thinking|reasoning)\s*>/i;
const BARE_START = /<\s*(think|thinking|reasoning)\s*>/i;
const BARE_END = /<\/\s*(think|thinking|reasoning)\s*>/i;

export type ThinkState = {
  inThink: boolean;
  ns?: string;
  neverThink?: boolean;
  shadowThink?: boolean;
};

function earliest(rest: string, ...patterns: RegExp[]): RegExpExecArray | null {
  let best: RegExpExecArray | null = null;
  for (const re of patterns) {
    re.lastIndex = 0;
    const m = re.exec(rest);
    re.lastIndex = 0;
    if (!m || m.index === undefined) continue;
    if (!best || m.index < best.index) best = m;
  }
  return best;
}

function isNsOpen(tag: string): boolean {
  return /<\s*mm:/i.test(tag);
}

export function feedThink(
  chunk: string,
  state: ThinkState,
): { content: string; reasoning: string } {
  if (state.neverThink) {
    return { content: chunk, reasoning: "" };
  }
  if (!state.inThink && !state.shadowThink && !chunk) {
    state.shadowThink = true;
    return { content: "", reasoning: "" };
  }
  if (state.shadowThink) {
    if (!chunk) return { content: "", reasoning: "" };
    const trimmed = chunk.trimStart();
    NS_START.lastIndex = 0;
    const nsOpen = NS_START.exec(trimmed);
    NS_START.lastIndex = 0;
    if (nsOpen && nsOpen.index === 0) {
      state.shadowThink = false;
    } else {
      state.shadowThink = false;
      state.inThink = false;
      state.ns = "";
      state.neverThink = true;
      return { content: chunk, reasoning: "" };
    }
  }
  let content = "";
  let reasoning = "";
  let rest = chunk;
  while (rest.length) {
    if (state.inThink) {
      const endRe = state.ns === "mm:" ? NS_END : BARE_END;
      endRe.lastIndex = 0;
      const m = endRe.exec(rest);
      endRe.lastIndex = 0;
      if (!m || m.index === undefined) {
        reasoning += rest;
        break;
      }
      reasoning += rest.slice(0, m.index);
      rest = rest.slice(m.index + m[0].length);
      state.inThink = false;
      state.ns = "";
      state.neverThink = true;
      content += rest;
      break;
    }
    const m = earliest(rest, NS_START, BARE_START);
    if (!m || m.index === undefined) {
      content += rest;
      state.neverThink = true;
      break;
    }
    content += rest.slice(0, m.index);
    rest = rest.slice(m.index + m[0].length);
    state.inThink = true;
    state.ns = isNsOpen(m[0]) ? "mm:" : "";
  }
  return { content, reasoning };
}
