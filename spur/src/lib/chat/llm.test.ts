import assert from "node:assert/strict";
import { test } from "node:test";
import { resolveLlm } from "./llm.ts";

test("local OpenAI-compatible server wins over xAI", () => {
  const llm = resolveLlm({
    LLM_BASE_URL: "http://127.0.0.1:1234/v1",
    LLM_MODEL: "gemma-3-27b",
    XAI_API_KEY: "should-not-use",
  });
  assert.equal(llm?.baseUrl, "http://127.0.0.1:1234/v1");
  assert.equal(llm?.model, "gemma-3-27b");
  assert.equal(llm?.webSearch, false);
  assert.equal(llm?.apiKey, "lm-studio");
});

test("xAI is used when no local server is configured", () => {
  const llm = resolveLlm({ XAI_API_KEY: "xai-test" });
  assert.equal(llm?.baseUrl, "https://api.x.ai/v1");
  assert.equal(llm?.model, "grok-4.5");
  assert.equal(llm?.webSearch, true);
});

test("missing both returns null", () => {
  assert.equal(resolveLlm({}), null);
});
