/** Palettes extracted from Pygments styles (pygments.org/styles). */

export type PygmentsPalette = {
  id: string;
  kw: string;
  st: string;
  cm: string;
  nu: string;
  fn: string;
  fg: string;
  bg: string;
};

export const PYGMENTS_STYLES: PygmentsPalette[] = [
  { id: "bw", kw: "#000000", st: "#000000", cm: "#000000", nu: "#000000", fn: "#000000", fg: "#000000", bg: "#ffffff" },
  { id: "sas", kw: "#2c2cff", st: "#800080", cm: "#008800", nu: "#2c8553", fn: "#2c2cff", fg: "#000000", bg: "#ffffff" },
  { id: "staroffice", kw: "#000080", st: "#000080", cm: "#696969", nu: "#000080", fn: "#000080", fg: "#000080", bg: "#ffffff" },
  { id: "xcode", kw: "#a90d91", st: "#c41a16", cm: "#177500", nu: "#1c01ce", fn: "#000000", fg: "#000000", bg: "#ffffff" },
  { id: "monokai", kw: "#66d9ef", st: "#e6db74", cm: "#959077", nu: "#ae81ff", fn: "#a6e22e", fg: "#f8f8f2", bg: "#272822" },
  { id: "lightbulb", kw: "#5ccfe6", st: "#000000", cm: "#888888", nu: "#000000", fn: "#000000", fg: "#000000", bg: "#ffffff" },
  { id: "github-dark", kw: "#ff7b72", st: "#a5d6ff", cm: "#8b949e", nu: "#a5d6ff", fn: "#d2a8ff", fg: "#e6edf3", bg: "#0d1117" },
  { id: "rrt", kw: "#ff0000", st: "#87ceeb", cm: "#00ff00", nu: "#ff00ff", fn: "#ffff00", fg: "#dddddd", bg: "#000000" },
  { id: "abap", kw: "#0000ff", st: "#55aa22", cm: "#888888", nu: "#33aaff", fn: "#0000ff", fg: "#000000", bg: "#ffffff" },
  { id: "algol", kw: "#000000", st: "#666666", cm: "#888888", nu: "#000000", fn: "#666666", fg: "#000000", bg: "#ffffff" },
  { id: "algol_nu", kw: "#000000", st: "#666666", cm: "#888888", nu: "#000000", fn: "#666666", fg: "#000000", bg: "#ffffff" },
  { id: "arduino", kw: "#728e00", st: "#7f8c8d", cm: "#95a5a6", nu: "#8a7b52", fn: "#d35400", fg: "#000000", bg: "#ffffff" },
  { id: "autumn", kw: "#0000aa", st: "#aa5500", cm: "#aaaaaa", nu: "#009999", fn: "#00aa00", fg: "#000000", bg: "#ffffff" },
  { id: "borland", kw: "#000080", st: "#0000ff", cm: "#008800", nu: "#0000ff", fn: "#ff0000", fg: "#000000", bg: "#ffffff" },
  { id: "colorful", kw: "#008800", st: "#dd2200", cm: "#888888", nu: "#6600ee", fn: "#0066bb", fg: "#000000", bg: "#ffffff" },
  { id: "igor", kw: "#0000ff", st: "#009c00", cm: "#ff0000", nu: "#000000", fn: "#c34e00", fg: "#000000", bg: "#ffffff" },
  { id: "lovelace", kw: "#444444", st: "#000000", cm: "#888888", nu: "#444444", fn: "#000000", fg: "#000000", bg: "#ffffff" },
  { id: "murphy", kw: "#228899", st: "#ff8888", cm: "#666666", nu: "#6600ee", fn: "#55eedd", fg: "#000000", bg: "#ffffff" },
  { id: "pastie", kw: "#008800", st: "#dd2200", cm: "#888888", nu: "#0000dd", fn: "#0066bb", fg: "#000000", bg: "#ffffff" },
  { id: "rainbow_dash", kw: "#2c5dcd", st: "#00cc66", cm: "#0080ff", nu: "#5918bb", fn: "#ff8000", fg: "#000000", bg: "#ffffff" },
  { id: "stata-light", kw: "#353580", st: "#7a2424", cm: "#008800", nu: "#2c2cff", fn: "#2c2cff", fg: "#111111", bg: "#ffffff" },
  { id: "trac", kw: "#445588", st: "#bb8844", cm: "#999988", nu: "#009999", fn: "#990000", fg: "#000000", bg: "#ffffff" },
  { id: "vs", kw: "#0000ff", st: "#a31515", cm: "#008000", nu: "#000000", fn: "#0000ff", fg: "#000000", bg: "#ffffff" },
  { id: "emacs", kw: "#aa22ff", st: "#bb4444", cm: "#008800", nu: "#666666", fn: "#00a000", fg: "#000000", bg: "#f8f8f8" },
  { id: "tango", kw: "#204a87", st: "#4e9a06", cm: "#8f5902", nu: "#0000cf", fn: "#000000", fg: "#000000", bg: "#f8f8f8" },
  { id: "solarized-light", kw: "#859900", st: "#2aa198", cm: "#93a1a1", nu: "#2aa198", fn: "#268bd2", fg: "#657b83", bg: "#fdf6e3" },
  { id: "manni", kw: "#006699", st: "#cc3300", cm: "#0099ff", nu: "#ff6600", fn: "#cc00ff", fg: "#000000", bg: "#f0f3f3" },
  { id: "gruvbox-light", kw: "#9d0006", st: "#79740e", cm: "#928374", nu: "#8f3f71", fn: "#427b58", fg: "#000000", bg: "#fbf1c7" },
  { id: "friendly", kw: "#007020", st: "#4070a0", cm: "#60a0b0", nu: "#40a070", fn: "#06287e", fg: "#000000", bg: "#f0f0f0" },
  { id: "friendly_grayscale", kw: "#575757", st: "#717171", cm: "#959595", nu: "#888888", fn: "#3f3f3f", fg: "#000000", bg: "#f0f0f0" },
  { id: "perldoc", kw: "#8b008b", st: "#cd5555", cm: "#228b22", nu: "#b452cd", fn: "#008b45", fg: "#000000", bg: "#eeeedd" },
  { id: "paraiso-light", kw: "#815ba4", st: "#48b685", cm: "#8d8687", nu: "#f99b15", fn: "#06b6ef", fg: "#2f1e2e", bg: "#e7e9db" },
  { id: "zenburn", kw: "#efdcbc", st: "#cc9393", cm: "#7f9f7f", nu: "#8cd0d3", fn: "#efef8f", fg: "#dcdccc", bg: "#3f3f3f" },
  { id: "nord", kw: "#81a1c1", st: "#a3be8c", cm: "#616e87", nu: "#b48ead", fn: "#88c0d0", fg: "#d8dee9", bg: "#2e3440" },
  { id: "material", kw: "#000000", st: "#000000", cm: "#888888", nu: "#000000", fn: "#000000", fg: "#000000", bg: "#ffffff" },
  { id: "one-dark", kw: "#c678dd", st: "#98c379", cm: "#7f848e", nu: "#d19a66", fn: "#61afef", fg: "#abb2bf", bg: "#282c34" },
  { id: "dracula", kw: "#000000", st: "#000000", cm: "#888888", nu: "#000000", fn: "#000000", fg: "#000000", bg: "#ffffff" },
  { id: "nord-darker", kw: "#81a1c1", st: "#a3be8c", cm: "#616e87", nu: "#b48ead", fn: "#88c0d0", fg: "#d8dee9", bg: "#242933" },
  { id: "gruvbox-dark", kw: "#fb4934", st: "#b8bb26", cm: "#928374", nu: "#d3869b", fn: "#8ec07c", fg: "#dddddd", bg: "#282828" },
  { id: "stata-dark", kw: "#7686bb", st: "#51cc99", cm: "#777777", nu: "#4fb8cc", fn: "#6a6aff", fg: "#cccccc", bg: "#232629" },
  { id: "paraiso-dark", kw: "#815ba4", st: "#48b685", cm: "#776e71", nu: "#f99b15", fn: "#06b6ef", fg: "#e7e9db", bg: "#2f1e2e" },
  { id: "coffee", kw: "#919191", st: "#c9b98f", cm: "#70757a", nu: "#87afaf", fn: "#fdd0c0", fg: "#ddd0c0", bg: "#262220" },
  { id: "solarized-dark", kw: "#859900", st: "#2aa198", cm: "#586e75", nu: "#2aa198", fn: "#268bd2", fg: "#839496", bg: "#002b36" },
  { id: "native", kw: "#6ebf26", st: "#ed9d13", cm: "#ababab", nu: "#51b2fd", fn: "#71adff", fg: "#d0d0d0", bg: "#202020" },
  { id: "inkpot", kw: "#808bed", st: "#ffcd8b", cm: "#cd8b00", nu: "#f0ad6d", fn: "#c080d0", fg: "#cfbfad", bg: "#1e1e27" },
  { id: "night-owl", kw: "#000000", st: "#000000", cm: "#888888", nu: "#000000", fn: "#000000", fg: "#000000", bg: "#ffffff" },
  { id: "fruity", kw: "#c34b00", st: "#0086d2", cm: "#008800", nu: "#0086f7", fn: "#ff0086", fg: "#ffffff", bg: "#111111" },
  { id: "vim", kw: "#cdcd00", st: "#cd0000", cm: "#000080", nu: "#cb006c", fn: "#cdcd00", fg: "#cccccc", bg: "#000000" },
  { id: "lilypond", kw: "#000000", st: "#000000", cm: "#888888", nu: "#000000", fn: "#000000", fg: "#000000", bg: "#ffffff" },
];

export const SYNTAX_KEY = "spur-syntax";
export const SYNTAX_AUTO = "auto";

export function findPalette(id: string | null | undefined): PygmentsPalette | undefined {
  if (!id || id === SYNTAX_AUTO) return undefined;
  return PYGMENTS_STYLES.find((s) => s.id === id);
}

export function paletteFor(
  id: string,
  resolved: "light" | "dark",
): PygmentsPalette {
  const found = findPalette(id);
  if (found) return found;
  const fallback = resolved === "light" ? "stata-light" : "fruity";
  return findPalette(fallback) ?? PYGMENTS_STYLES[0];
}

