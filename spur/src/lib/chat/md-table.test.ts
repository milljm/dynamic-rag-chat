import assert from "node:assert/strict";
import { test } from "node:test";
import { isDelimiterRow, parseTableAt, splitRow } from "./md-table.ts";

test("splits pipe rows, including tight GFM", () => {
  assert.deepEqual(splitRow("| Feature | Status |"), ["Feature", "Status"]);
  assert.deepEqual(splitRow("|---|---|"), ["---", "---"]);
  assert.deepEqual(splitRow("| Bold text | Works fine |"), [
    "Bold text",
    "Works fine",
  ]);
});

test("delimiter rows match dashes with optional alignment", () => {
  assert.equal(isDelimiterRow("|---|---|"), true);
  assert.equal(isDelimiterRow("| :--- | ---: | :---: |"), true);
  assert.equal(isDelimiterRow("|-------|--------------|--------|"), true);
  assert.equal(isDelimiterRow("| Feature | Status |"), false);
  assert.equal(isDelimiterRow("not a table"), false);
});

test("parses the Spur regression table", () => {
  const src = [
    "Here's a quick example of a markdown table:",
    "",
    "| Feature | Status |",
    "|---|---|",
    "| Bold text | Works fine |",
    "| Italic | Also works |",
    "| Code spans | Yep |",
  ];
  const hit = parseTableAt(src, 2);
  assert.ok(hit);
  assert.equal(hit.consumed, 5);
  assert.deepEqual(hit.table.headers, ["Feature", "Status"]);
  assert.equal(hit.table.rows.length, 3);
  assert.deepEqual(hit.table.rows[0], ["Bold text", "Works fine"]);
  assert.deepEqual(hit.table.rows[2], ["Code spans", "Yep"]);
});

test("header without a delimiter is not a table", () => {
  const src = ["| Feature | Status |", "still waiting on dashes"];
  assert.equal(parseTableAt(src, 0), null);
});

test("stops at a blank line so the next paragraph is prose", () => {
  const src = [
    "| Repo | Lang |",
    "|---|---|",
    "| **dynamic-rag-chat** | Python |",
    "",
    "After the table.",
  ];
  const hit = parseTableAt(src, 0);
  assert.ok(hit);
  assert.equal(hit.consumed, 3);
  assert.equal(hit.table.rows.length, 1);
});

test("reads alignment from the delimiter row", () => {
  const src = ["| A | B | C |", "|:---|---:|:---:|", "| 1 | 2 | 3 |"];
  const hit = parseTableAt(src, 0);
  assert.ok(hit);
  assert.deepEqual(hit.table.align, ["left", "right", "center"]);
});
