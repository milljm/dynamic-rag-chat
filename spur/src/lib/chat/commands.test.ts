import assert from "node:assert/strict";
import { test } from "node:test";
import {
  isOoc,
  parseComposerInput,
  parseIncludes,
  SLASH_HELP,
  stripRare,
} from "./commands.ts";

test("plain text is a message", () => {
  assert.deepEqual(parseComposerInput("hello"), {
    kind: "message",
    text: "hello",
    rare: [],
    ooc: false,
  });
});

test("unknown slash stays a message", () => {
  const raw = "\\not-a-command hello";
  const parsed = parseComposerInput(raw);
  assert.equal(parsed.kind, "message");
  if (parsed.kind === "message") assert.equal(parsed.text, raw);
});

test("local commands parse with args", () => {
  assert.deepEqual(parseComposerInput("\\rewind 3"), {
    kind: "command",
    command: "rewind",
    args: "3",
  });
  assert.deepEqual(parseComposerInput("\\dbranch alt-ending"), {
    kind: "command",
    command: "dbranch",
    args: "alt-ending",
  });
  const help = parseComposerInput("\\?");
  assert.equal(help.kind, "command");
  if (help.kind === "command") {
    assert.equal(help.command, "help");
  }
  assert.equal(parseComposerInput("\\history 5").kind, "command");
});

test("inline agent and no-context strip the command", () => {
  assert.deepEqual(parseComposerInput("\\agent what shipped today?"), {
    kind: "inline",
    agent: true,
    image: false,
    coding: false,
    noContext: false,
    text: "what shipped today?",
    rare: [],
    ooc: false,
  });
  assert.deepEqual(parseComposerInput("\\no-context just riff"), {
    kind: "inline",
    agent: false,
    image: false,
    coding: false,
    noContext: true,
    text: "just riff",
    rare: [],
    ooc: false,
  });
  assert.deepEqual(parseComposerInput("\\image a red fox at dusk"), {
    kind: "inline",
    agent: false,
    image: true,
    coding: false,
    noContext: false,
    text: "a red fox at dusk",
    rare: [],
    ooc: false,
  });
  assert.deepEqual(parseComposerInput("\\coding write a hello script"), {
    kind: "inline",
    agent: false,
    image: false,
    coding: true,
    noContext: false,
    text: "write a hello script",
    rare: [],
    ooc: false,
  });
});

test("include command splits branch name from the rest", () => {
  const parsed = parseComposerInput("\\include research-notes summarize that");
  assert.deepEqual(parsed, {
    kind: "include",
    branch: "research-notes",
    text: "summarize that",
    rare: [],
    ooc: false,
  });
});

test("includes split urls from filesystem paths", () => {
  const found = parseIncludes(
    "see {{https://example.com/a}} and {{/tmp/notes.md}}",
  );
  assert.deepEqual(found.urls, ["https://example.com/a"]);
  assert.deepEqual(found.paths, ["/tmp/notes.md"]);
});

test("slash help lists url scrape", () => {
  assert.ok(SLASH_HELP.some((row) => row.cmd.includes("{{https://")));
});

test("rare tokens strip from the spoken line", () => {
  const { text, rare } = stripRare("She opens the door [RARE NOW]");
  assert.equal(text, "She opens the door");
  assert.deepEqual(rare, ["[RARE NOW]"]);
});

test("OOC prefix is detected", () => {
  assert.equal(isOoc("OOC: stay in first person"), true);
  assert.equal(isOoc("hello"), false);
});
