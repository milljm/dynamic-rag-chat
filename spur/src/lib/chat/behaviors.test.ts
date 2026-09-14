import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  BEHAVIORS_KEY,
  DEFAULT_BEHAVIORS,
  normalizeBehaviors,
  readBehaviors,
} from "./behaviors.ts";

describe("behaviors prefs", () => {
  it("defaults to auto scroll on and clear screen off", () => {
    assert.equal(BEHAVIORS_KEY, "spur-behaviors");
    assert.equal(DEFAULT_BEHAVIORS.autoScroll, true);
    assert.equal(DEFAULT_BEHAVIORS.clearScreen, false);
  });

  it("returns defaults without a DOM localStorage", () => {
    assert.deepEqual(readBehaviors(), DEFAULT_BEHAVIORS);
  });

  it("keeps valid stored values", () => {
    assert.deepEqual(normalizeBehaviors({ clearScreen: true, autoScroll: false }), {
      clearScreen: true,
      autoScroll: false,
    });
    assert.deepEqual(normalizeBehaviors({ clearScreen: true }), {
      clearScreen: true,
      autoScroll: true,
    });
  });

  it("coerces junk back to defaults", () => {
    assert.deepEqual(normalizeBehaviors("junk"), DEFAULT_BEHAVIORS);
    assert.deepEqual(normalizeBehaviors(null), DEFAULT_BEHAVIORS);
    assert.deepEqual(normalizeBehaviors({ clearScreen: "yes" }), DEFAULT_BEHAVIORS);
    assert.deepEqual(normalizeBehaviors({ autoScroll: 0 }), DEFAULT_BEHAVIORS);
  });
});
