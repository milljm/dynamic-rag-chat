/**
 * Per-turn performance samples powering the sidebar's metrics widget.
 *
 * One sample per completed assistant response in the active branch. Every
 * field mirrors the per-message footer in `thread.tsx` exactly:
 *
 *   footer          sample
 *   --------------  ----------------------------------------------
 *   TTFT 0.42s      ttft             (s)
 *   Gen 3.10s       gen              (s, decode window after first token)
 *   N tok           completionTokens
 *   N.N T/s         tokensPerSec
 *   DUP N           dupTokens
 *   CTX N           ctxTokens
 *
 * Reasoning models additionally get a non-reasoning split (the orange overlay
 * on the Gen / Tok charts). The share is derived from the response's stored
 * `reasoning` text against the total token count, so it is durable and works
 * for every turn — live or reloaded — without any backend change.
 *
 * `null` means "no data for this turn" (aborted stream, older backend) and
 * renders as a gap in the line.
 */
import type { Message, StreamMetrics } from "./types";

export type PerfTurn = {
  /** 1-based user turn this response answers. */
  turn: number;
  /** Time to first token, in seconds (footer TTFT). */
  ttft: number | null;
  /** Decode window after the first token, in seconds (footer Gen). */
  gen: number | null;
  /** Completion tokens (footer Tok). */
  completionTokens: number | null;
  /** Decode speed in tokens/second (footer T/s); null when the window is empty. */
  tokensPerSec: number | null;
  /** Tokens reclaimed by RAG dedupe (footer DUP). */
  dupTokens: number | null;
  /** Packed context-window tokens (footer CTX). */
  ctxTokens: number | null;
  /** Model that answered this turn (footer label). */
  model: string;
  /** Estimated reasoning tokens (0 when the model did not reason). */
  reasoningTokens: number;
  /** Reasoning share of the decode window, 0..1. */
  reasoningFraction: number;
  /** Completion tokens attributable to the answer (`Tok` × non-reasoning share). */
  nonReasoningTokens: number;
  /** Decode window attributable to the answer (`Gen` × non-reasoning share). */
  nonReasoningGen: number;
};

/**
 * Flattened `[turn, metrics, reasoningTokens, …]` triples. The `metrics`
 * references are kept (not copied) because they survive content-only stream
 * patches — `applyReplaceMessage` spreads the untouched metric object over —
 * so a shallow array compare stays stable per token and the widget only
 * re-renders when a response actually completes.
 */
export type PerfSample = number | StreamMetrics | undefined;

/** Rough token estimate (chars / 4), mirroring `use-send.ts`. */
function estimateTokens(text: string): number {
  return Math.max(1, Math.round(text.length / 4));
}

/** Stride of the flattened `perfSamples` array: turn, metrics, reasoning. */
const STRIDE = 3;

/** Flatten a branch's assistant responses into shallow-comparable samples. */
export function perfSamples(messages: Message[]): PerfSample[] {
  const out: PerfSample[] = [];
  let turn = 0;
  for (const m of messages) {
    if (m.role === "user") {
      turn += 1;
      continue;
    }
    out.push(turn, m.metrics, m.reasoning ? estimateTokens(m.reasoning) : 0);
  }
  return out;
}

/** One footer-parity sample for a finished response. */
function sampleTurn(turn: number, m: StreamMetrics, reasoningTokens: number): PerfTurn {
  const gen = Math.max(0, m.generationTime - m.ttft);
  const tokens = m.tokenCount;
  // Reasoning share of the response; clamped because the footer total is the
  // server's count while the reasoning side is our estimate.
  const fraction = tokens > 0 ? Math.min(1, Math.max(0, reasoningTokens / tokens)) : 0;
  const answerFraction = 1 - fraction;
  return {
    turn,
    ttft: m.ttft,
    gen,
    completionTokens: tokens,
    tokensPerSec: gen > 0 ? tokens / gen : null,
    dupTokens: m.tokenSavings,
    ctxTokens: m.promptTokens,
    model: m.model,
    reasoningTokens,
    reasoningFraction: fraction,
    nonReasoningTokens: tokens * answerFraction,
    nonReasoningGen: gen * answerFraction,
  };
}

/** Rebuild chart samples from `perfSamples` output (skips unanswered turns). */
export function historyFromSamples(samples: PerfSample[]): PerfTurn[] {
  const out: PerfTurn[] = [];
  for (let i = 0; i + STRIDE - 1 < samples.length; i += STRIDE) {
    const turn = samples[i] as number;
    const metrics = samples[i + 1] as StreamMetrics | undefined;
    const reasoningTokens = (samples[i + 2] as number) || 0;
    if (!metrics) continue;
    out.push(sampleTurn(turn, metrics, reasoningTokens));
  }
  return out;
}

/** Direct messages → chart samples (convenience for non-store callers/tests). */
export function perfHistory(messages: Message[]): PerfTurn[] {
  return historyFromSamples(perfSamples(messages));
}

/** True when any turn in the branch was produced by a reasoning model. */
export function hasReasoning(history: PerfTurn[]): boolean {
  return history.some((t) => t.reasoningTokens > 0);
}

export type MetricKey = "ttft" | "gen" | "tokens" | "tps" | "dup" | "ctx";

export type MetricDef = {
  key: MetricKey;
  /** Chip label. */
  label: string;
  /** Y-axis / readout unit. */
  unit: string;
  /** Metric value at a turn; null = no data (chart gap). */
  value: (t: PerfTurn) => number | null;
  /** Optional second (orange) series — the non-reasoning share. */
  alt?: (t: PerfTurn) => number | null;
  /** Compact readout formatting (hover tooltip / latest line). */
  fmt: (v: number) => string;
};

/** Compact token count (footer parity): 500 → "500", 1249 → "1.2k", 3.2M → "3.2M". */
export function fmtTok(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 === 0 ? 0 : 1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}k`;
  return n.toLocaleString("en-US");
}

export const METRICS: MetricDef[] = [
  {
    key: "ttft",
    label: "TTFT",
    unit: "s",
    value: (t) => t.ttft,
    fmt: (v) => `${v.toFixed(2)}s`,
  },
  {
    key: "gen",
    label: "Gen",
    unit: "s",
    value: (t) => t.gen,
    alt: (t) => t.nonReasoningGen,
    fmt: (v) => `${v.toFixed(2)}s`,
  },
  {
    key: "tokens",
    label: "Tok",
    unit: "tok",
    value: (t) => t.completionTokens,
    alt: (t) => t.nonReasoningTokens,
    fmt: (v) => v.toLocaleString("en-US", { maximumFractionDigits: 0 }),
  },
  {
    key: "tps",
    label: "T/s",
    unit: "tok/s",
    value: (t) => t.tokensPerSec,
    fmt: (v) => v.toFixed(1),
  },
  {
    key: "dup",
    label: "Dup",
    unit: "tok",
    value: (t) => t.dupTokens,
    fmt: fmtTok,
  },
  {
    key: "ctx",
    label: "CTX",
    unit: "tok",
    value: (t) => t.ctxTokens,
    fmt: fmtTok,
  },
];

/** One point per turn for a metric (null where the turn has no data). */
export function metricSeries(history: PerfTurn[], metric: MetricDef): (number | null)[] {
  return history.map((t) => metric.value(t));
}

/** One point per turn for a metric's optional orange (non-reasoning) series. */
export function altSeries(history: PerfTurn[], metric: MetricDef): (number | null)[] | null {
  const alt = metric.alt;
  return alt ? history.map((t) => alt(t)) : null;
}

/** Last non-null value in a series (for the "latest" readout). */
export function lastValue(series: (number | null)[]): number | null {
  for (let i = series.length - 1; i >= 0; i--) {
    const v = series[i];
    if (v != null && Number.isFinite(v)) return v;
  }
  return null;
}

/** Round a max up to a tidy grid top (1/2/5 × 10^n). */
export function niceMax(v: number): number {
  if (!Number.isFinite(v) || v <= 0) return 1;
  const exp = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v / exp;
  const step = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return step * exp;
}

/** Compact tick label (2 significant digits below 10; no float noise). */
export function fmtTick(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(v % 1000 === 0 ? 0 : 1)}k`;
  if (v >= 100) return v.toFixed(0);
  if (v >= 10) return v.toFixed(Number.isInteger(v) ? 0 : 1);
  return String(Number(v.toPrecision(2)));
}
