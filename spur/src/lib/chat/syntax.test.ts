import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  PYGMENTS_STYLES,
  findPalette,
  paletteFor,
} from "./pygments-styles.ts";

describe("pygments palettes", () => {
  it("includes the pygments.org set plus lilypond", () => {
    assert.ok(PYGMENTS_STYLES.length >= 49);
    assert.ok(findPalette("coffee"));
    assert.ok(findPalette("stata-light"));
    assert.ok(findPalette("monokai"));
  });

  it("auto follows spur theme", () => {
    assert.equal(paletteFor("auto", "dark").id, "coffee");
    assert.equal(paletteFor("auto", "light").id, "stata-light");
    assert.equal(paletteFor("monokai", "light").id, "monokai");
  });

  it("coffee matches pygments token colors", () => {
    const c = findPalette("coffee")!;
    assert.equal(c.kw, "#919191");
    assert.equal(c.bg, "#262220");
    assert.equal(c.fg, "#ddd0c0");
  });
});
