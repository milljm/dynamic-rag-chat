import assert from "node:assert/strict";
import { test } from "node:test";
import { newId } from "./id.ts";

test("newId returns RFC-ish UUIDs without throwing", () => {
  const a = newId();
  const b = newId();
  assert.match(a, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  assert.notEqual(a, b);
});
