/** `crypto.randomUUID` throws (or is missing) on insecure HTTP like http://llm:8080. */
export function newId(): string {
  const bytes = new Uint8Array(16);
  if (typeof crypto !== "undefined" && crypto.getRandomValues) {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256);
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function installIdPolyfill() {
  if (typeof crypto === "undefined") return;
  try {
    if (typeof crypto.randomUUID === "function") {
      crypto.randomUUID();
      return;
    }
  } catch {
    /* insecure origin — method exists but throws */
  }
  try {
    Object.defineProperty(crypto, "randomUUID", {
      value: newId,
      configurable: true,
    });
  } catch {
    /* ignore */
  }
}

installIdPolyfill();
