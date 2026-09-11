import assert from "node:assert/strict";
import { test } from "node:test";
import {
  altSeries,
  fmtTick,
  hasReasoning,
  historyFromSamples,
  lastValue,
  METRICS,
  metricSeries,
  niceMax,
  perfHistory,
  perfSamples,
} from "./perf.ts";
import type { Message, StreamMetrics } from "./types.ts";

function user(turn: number): Message {
  return { id: `u${turn}`, role: "user", content: "hi", createdAt: turn };
}

function answer(id: string, metrics?: StreamMetrics, reasoning?: string): Message {
  return { id, role: "assistant", content: "ok", createdAt: 0, metrics, reasoning };
}

function metrics(overrides: Partial<StreamMetrics> = {}): StreamMetrics {
  return {
    model: "m1",
    tokenCount: 100,
    generationTime: 5,
    promptTokens: 2000,
    tokenSavings: 30,
    ttft: 1,
    ...overrides,
  };
}

test("perfHistory maps footer metrics per response", () => {
  const history = perfHistory([user(1), answer("a1", metrics()), user(2), answer("a2")]);
  assert.equal(history.length, 1);
  assert.deepEqual(history[0], {
    turn: 1,
    ttft: 1,
    gen: 4,
    completionTokens: 100,
    tokensPerSec: 25,
    dupTokens: 30,
    ctxTokens: 2000,
    model: "m1",
    reasoningTokens: 0,
    reasoningFraction: 0,
    nonReasoningTokens: 100,
    nonReasoningGen: 4,
  });
});

test("tokensPerSec is null when the decode window is empty", () => {
  const [turn] = perfHistory([
    user(1),
    answer("a", metrics({ generationTime: 2, ttft: 2, tokenSavings: 0 })),
  ]);
  assert.equal(turn.gen, 0);
  assert.equal(turn.tokensPerSec, null);
  assert.equal(turn.dupTokens, 0);
});

test("perfSamples yields [turn, metrics, reasoningTokens] triples", () => {
  const m1 = metrics();
  const samples = perfSamples([user(1), answer("a1", m1), user(2), answer("a2")]);
  assert.equal(samples.length, 6);
  assert.equal(samples[0], 1);
  assert.equal(samples[1], m1);
  assert.equal(samples[2], 0);
  assert.equal(samples[3], 2);
  assert.equal(samples[4], undefined);
  assert.equal(samples[5], 0);
  assert.deepEqual(
    historyFromSamples(samples),
    perfHistory([user(1), answer("a1", m1), user(2), answer("a2")]),
  );
});

test("reasoning models split the answer out of Tok and Gen", () => {
  // 2000 chars ≈ 500 estimated tokens against a 1000-token response.
  const history = perfHistory([
    user(1),
    answer("a1", metrics({ tokenCount: 1000, generationTime: 5, ttft: 1 }), "x".repeat(2000)),
  ]);
  const [turn] = history;
  assert.equal(turn.reasoningTokens, 500);
  assert.equal(turn.reasoningFraction, 0.5);
  assert.equal(turn.nonReasoningTokens, 500);
  assert.equal(turn.nonReasoningGen, 2); // gen 4s × 0.5
});

test("reasoningFraction is clamped when the estimate exceeds the total", () => {
  const [turn] = perfHistory([
    user(1),
    answer("a1", metrics({ tokenCount: 10 }), "x".repeat(4000)),
  ]);
  assert.equal(turn.reasoningFraction, 1);
  assert.equal(turn.nonReasoningTokens, 0);
  assert.equal(turn.nonReasoningGen, 0);
});

test("hasReasoning flags branches with at least one reasoner turn", () => {
  assert.equal(hasReasoning(perfHistory([user(1), answer("a1", metrics())])), false);
  assert.equal(
    hasReasoning(perfHistory([user(1), answer("a1", metrics(), "hmm")])),
    true,
  );
});

test("only Gen and Tok carry a non-reasoning line", () => {
  const withAlt = METRICS.filter((m) => m.alt).map((m) => m.key);
  assert.deepEqual(withAlt, ["gen", "tokens"]);
});

test("altSeries is null for metrics without a split", () => {
  const history = perfHistory([user(1), answer("a1", metrics())]);
  const ttft = METRICS.find((m) => m.key === "ttft")!;
  const ctx = METRICS.find((m) => m.key === "ctx")!;
  assert.equal(altSeries(history, ttft), null);
  assert.deepEqual(altSeries(history, ctx), null);
  const tokens = METRICS.find((m) => m.key === "tokens")!;
  assert.deepEqual(altSeries(history, tokens), [100]);
});

test("metricSeries + lastValue skip gaps", () => {
  const history = perfHistory([user(1), answer("a1", metrics())]);
  const ctx = METRICS.find((m) => m.key === "ctx")!;
  const series = metricSeries(history, ctx);
  assert.deepEqual(series, [2000]);
  assert.equal(lastValue(series), 2000);
  assert.equal(lastValue([null, null]), null);
  assert.equal(lastValue([]), null);
});

test("niceMax rounds to a tidy grid top", () => {
  assert.equal(niceMax(0), 1);
  assert.equal(niceMax(3), 5);
  assert.equal(niceMax(12), 20);
  assert.equal(niceMax(230), 500);
});

test("fmtTick compacts without float noise", () => {
  assert.equal(fmtTick(0), "0");
  assert.equal(fmtTick(150), "150");
  assert.equal(fmtTick(1500), "1.5k");
});
