import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ARTIFACT_TTL,
  artifactsFromMessages,
  extractArtifacts,
  isFilename,
  parseFenceInfo,
} from "./artifacts.ts";

test("plain language one-liners are not files", () => {
  assert.equal(isFilename("python"), false);
  assert.equal(parseFenceInfo("python").file, null);
  assert.equal(extractArtifacts("```python\nprint(1)\n```").length, 0);
});

test("unnamed fences that look like files become untitled", () => {
  const a = extractArtifacts(
    "```python\ndef main():\n    print(1)\n\nif __name__ == '__main__':\n    main()\n```",
  );
  assert.equal(a[0]?.file, "untitled.py");
});

test("first-line comment can name an unnamed fence", () => {
  const a = extractArtifacts("```python\n# hello_world.py\nprint(1)\n```");
  assert.equal(a[0]?.file, "hello_world.py");
});

test("info-string filenames are harvested", () => {
  const a = extractArtifacts("```python app.py\nprint(1)\n```");
  assert.equal(a[0]?.file, "app.py");
  assert.equal(a[0]?.text, "print(1)");
  assert.equal(parseFenceInfo("js:src/main.ts").file, "src/main.ts");
  assert.equal(parseFenceInfo('filename="notes.md"').file, "notes.md");
});

test("heading before a fence names the file", () => {
  const a = extractArtifacts("**lib.rs**\n```rust\nfn main() {}\n```");
  assert.equal(a[0]?.file, "lib.rs");
});

test("latest assistant file of the same name wins", () => {
  const list = artifactsFromMessages([
    { role: "assistant", content: "```txt a.txt\none\n```" },
    { role: "user", content: "again" },
    { role: "assistant", content: "```txt a.txt\ntwo\n```" },
  ]);
  assert.equal(list.length, 1);
  assert.equal(list[0]?.text, "two");
});

test("only last ARTIFACT_TTL turns are listed", () => {
  const msgs = [];
  for (let t = 1; t <= 6; t++) {
    msgs.push({ role: "user", content: `t${t}` });
    msgs.push({
      role: "assistant",
      content: "```txt f" + t + ".txt\n" + t + "\n```",
    });
  }
  const list = artifactsFromMessages(msgs);
  assert.equal(ARTIFACT_TTL, 4);
  assert.equal(list.length, 4);
  const names = list.map((a) => a.file).sort();
  assert.deepEqual(names, ["f3.txt", "f4.txt", "f5.txt", "f6.txt"]);
  const by = Object.fromEntries(list.map((a) => [a.file, a.remaining]));
  assert.equal(by["f6.txt"], 4);
  assert.equal(by["f5.txt"], 3);
  assert.equal(by["f4.txt"], 2);
  assert.equal(by["f3.txt"], 1);
});
