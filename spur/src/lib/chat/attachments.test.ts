import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ensureFileType,
  guessImageMime,
  isAllowedFile,
  isImageFile,
} from "./attachments.ts";

function fake(name: string, type: string): File {
  return new File(["x"], name, { type });
}

test("empty MIME still allows images by extension (LAN / drag-drop)", () => {
  const png = fake("shot.png", "");
  assert.equal(isImageFile(png), true);
  assert.equal(isAllowedFile(png), true);
  assert.equal(isAllowedFile(fake("notes.md", "")), true);
  assert.equal(isAllowedFile(fake("photo.JPEG", "application/octet-stream")), true);
});

test("ensureFileType fills image MIME so the decoder will accept the blob", () => {
  const typed = ensureFileType(fake("shot.png", ""));
  assert.equal(typed.type, "image/png");
  assert.equal(ensureFileType(fake("pic.JPG", "")).type, "image/jpeg");
  assert.equal(
    ensureFileType(fake("photo.JPEG", "application/octet-stream")).type,
    "image/jpeg",
  );
  assert.equal(ensureFileType(fake("already.png", "image/png")).type, "image/png");
});

test("guessImageMime covers common photo extensions", () => {
  assert.equal(guessImageMime("a.webp"), "image/webp");
  assert.equal(guessImageMime("a.svg"), "image/svg+xml");
});

test("unknown binaries stay rejected", () => {
  assert.equal(isAllowedFile(fake("payload.bin", "")), false);
  assert.equal(isAllowedFile(fake("doc.docx", "")), false);
});
