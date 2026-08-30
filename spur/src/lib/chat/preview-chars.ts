/** Branch preview fits 52 chars at the 350px sidebar minimum. */
export const SIDEBAR_MIN_WIDTH = 350;
export const PREVIEW_CHARS_AT_MIN = 52;
/** text-xs, ~5.4px per character in the preview row. */
export const PREVIEW_PX_PER_CHAR = 5.4;

export function previewCharsForWidth(width: number): number {
  const extra = width - SIDEBAR_MIN_WIDTH;
  return Math.max(
    24,
    PREVIEW_CHARS_AT_MIN + Math.round(extra / PREVIEW_PX_PER_CHAR),
  );
}
