import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { isThemePref, resolveTheme, THEME_KEY } from "./theme.ts";

describe("theme pref", () => {
  it("accepts light, dark, system", () => {
    assert.equal(isThemePref("light"), true);
    assert.equal(isThemePref("dark"), true);
    assert.equal(isThemePref("system"), true);
    assert.equal(isThemePref("auto"), false);
    assert.equal(THEME_KEY, "spur-theme");
  });

  it("system follows the mock media query", () => {
    assert.equal(resolveTheme("dark"), "dark");
    assert.equal(resolveTheme("light"), "light");
  });
});
