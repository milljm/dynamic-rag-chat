import { newId } from "./id.ts";
import type { Attachment } from "./types";

const TEXT_TYPES = new Set([
  "text/plain",
  "text/markdown",
  "text/csv",
  "text/html",
  "application/json",
  "application/javascript",
  "text/javascript",
  "text/x-python",
]);

const TEXT_EXT =
  /\.(txt|md|py|json|csv|html|htm|js|ts|tsx|jsx|css|yml|yaml|toml|xml|sh|rs|go|rb|sql|log)$/i;
const IMAGE_EXT = /\.(png|jpe?g|gif|webp|bmp|svg|heic|heif|avif)$/i;

export function isImageFile(file: File): boolean {
  const mime = (file.type || "").toLowerCase();
  if (mime.startsWith("image/")) return true;
  return IMAGE_EXT.test(file.name);
}

export function isAllowedFile(file: File): boolean {
  if (isImageFile(file)) return true;
  if (TEXT_TYPES.has(file.type)) return true;
  return TEXT_EXT.test(file.name);
}

/** Browsers on LAN/drag-drop often omit File.type; Image() then refuses the blob. */
export function ensureFileType(file: File): File {
  const current = (file.type || "").toLowerCase();
  const useless =
    !current ||
    current === "application/octet-stream" ||
    current === "application/octetstream";
  if (!useless) return file;
  const mime = IMAGE_EXT.test(file.name)
    ? guessImageMime(file.name)
    : TEXT_EXT.test(file.name)
      ? "text/plain"
      : "";
  if (!mime) return file;
  return new File([file], file.name, {
    type: mime,
    lastModified: file.lastModified,
  });
}

export function guessImageMime(name: string): string {
  const ext = name.split(".").pop()?.toLowerCase();
  if (ext === "jpg" || ext === "jpeg") return "image/jpeg";
  if (ext === "svg") return "image/svg+xml";
  if (ext === "heic" || ext === "heif") return "image/heic";
  if (
    ext === "png" ||
    ext === "gif" ||
    ext === "webp" ||
    ext === "bmp" ||
    ext === "avif"
  ) {
    return `image/${ext}`;
  }
  return "image/png";
}

export async function fileToAttachment(file: File): Promise<Attachment> {
  const typed = ensureFileType(file);
  if (isImageFile(typed)) {
    const dataUrl = await imageDataUrl(typed);
    return {
      id: newId(),
      name: typed.name,
      mime: typed.type || guessImageMime(typed.name),
      kind: "image",
      dataUrl,
      size: typed.size,
    };
  }
  const text = await typed.text();
  return {
    id: newId(),
    name: typed.name,
    mime: typed.type || "text/plain",
    kind: "text",
    text,
    size: typed.size,
  };
}

function readAsDataUrl(file: File): Promise<string> {
  const mime = file.type || guessImageMime(file.name);
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => {
      void file
        .arrayBuffer()
        .then((buf) => resolve(stampMime(bytesToDataUrl(buf, mime), mime)))
        .catch(() => reject(reader.error ?? new Error("Could not read file")));
    };
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string" || !result.includes(",")) {
        reject(new Error("Could not read file"));
        return;
      }
      resolve(stampMime(result, mime));
    };
    reader.readAsDataURL(file);
  });
}

function bytesToDataUrl(buf: ArrayBuffer, mime: string): string {
  const bytes = new Uint8Array(buf);
  const chunk = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return `data:${mime || "application/octet-stream"};base64,${btoa(binary)}`;
}

function stampMime(dataUrl: string, mime: string): string {
  if (!mime) return dataUrl;
  return dataUrl.replace(/^data:[^;,]*/, `data:${mime}`);
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const timer = setTimeout(() => reject(new Error("timeout")), 8000);
    img.onload = () => {
      clearTimeout(timer);
      resolve(img);
    };
    img.onerror = () => {
      clearTimeout(timer);
      reject(new Error("Could not decode image"));
    };
    img.src = url;
  });
}

async function imageDataUrl(file: File): Promise<string> {
  const original = await readAsDataUrl(file);
  try {
    const img = await loadImage(original);
    const scale = Math.min(1, 1280 / Math.max(img.width, img.height));
    const w = Math.max(1, Math.round(img.width * scale));
    const h = Math.max(1, Math.round(img.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return original;
    ctx.drawImage(img, 0, 0, w, h);
    return canvas.toDataURL("image/jpeg", 0.86);
  } catch {
    return original;
  }
}
