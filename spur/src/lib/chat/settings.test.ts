import assert from "node:assert/strict";
import { test } from "node:test";
import { hostForRole, normalizeHost, uniqueHosts } from "./settings-hosts.ts";

test("normalizeHost trims and strips trailing slashes", () => {
  assert.equal(normalizeHost("  http://127.0.0.1:1234/v1/  "), "http://127.0.0.1:1234/v1");
  assert.equal(normalizeHost(""), "");
});

test("hostForRole uses the role server when set", () => {
  assert.equal(
    hostForRole("http://coder:1234/v1", "http://main:1234/v1"),
    "http://coder:1234/v1",
  );
});

test("hostForRole falls back to the generator server", () => {
  assert.equal(hostForRole("", "http://main:1234/v1"), "http://main:1234/v1");
  assert.equal(hostForRole("   ", "http://main:1234/v1/"), "http://main:1234/v1");
});

test("uniqueHosts de-dupes by normalized spelling", () => {
  assert.deepEqual(
    uniqueHosts([
      "http://main:1234/v1",
      "http://main:1234/v1/",
      "",
      "http://coder:1234/v1",
      "http://embed:11434/v1",
    ]),
    [
      "http://main:1234/v1",
      "http://coder:1234/v1",
      "http://embed:11434/v1",
    ],
  );
});

test("uniqueHosts ignores blanks", () => {
  assert.deepEqual(uniqueHosts(["", "   ", "http://main:1234/v1"]), [
    "http://main:1234/v1",
  ]);
});
