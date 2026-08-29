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
    assert.ok(findPalette("fruity"));
    assert.ok(findPalette("stata-light"));
    assert.ok(findPalette("monokai"));
  });

  it("auto follows spur theme", () => {
    assert.equal(paletteFor("auto", "dark").id, "fruity");
    assert.equal(paletteFor("auto", "light").id, "stata-light");
    assert.equal(paletteFor("monokai", "light").id, "monokai");
  });

  it("fruity matches pygments token colors", () => {
    const f = findPalette("fruity")!;
    assert.equal(f.kw, "#fb660a");
    assert.equal(f.fn, "#ff0086");
    assert.equal(f.bg, "#111111");
  });
});
