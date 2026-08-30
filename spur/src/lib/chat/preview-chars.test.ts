import assert from "node:assert/strict";
import { test } from "node:test";
import {
  PREVIEW_CHARS_AT_MIN,
  previewCharsForWidth,
} from "./preview-chars.ts";

test("52 characters at the 350px minimum", () => {
  assert.equal(previewCharsForWidth(350), PREVIEW_CHARS_AT_MIN);
});

test("grows as the sidebar widens", () => {
  assert.ok(previewCharsForWidth(450) > PREVIEW_CHARS_AT_MIN);
  assert.ok(previewCharsForWidth(560) > previewCharsForWidth(450));
});

test("shrinks a little under the minimum (mobile drawer)", () => {
  const n = previewCharsForWidth(320);
  assert.ok(n < PREVIEW_CHARS_AT_MIN);
  assert.ok(n >= 24);
});
