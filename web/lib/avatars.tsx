/**
 * The preset avatar set — drawn, not uploaded.
 *
 * WHY THESE ARE DRAWINGS. An uploaded avatar is user-supplied imagery on a
 * public profile and needs a moderation pipeline behind it: review queue,
 * reporting, takedown, and a person to run them. None of that exists, so
 * uploads are not offered. A preset carries none of the risk — the server
 * stores a key from a fixed allowlist (`server/core/avatars.py`, which is
 * the gate) and the artwork lives here as inline SVG.
 *
 * KEEP IN SYNC with `server/core/avatars.py`. A key here that the server
 * does not know is refused on save; a key there with no entry here renders
 * as the empty fallback.
 *
 * Every preset is a 3x3 cube face, because that is what this product is.
 * They differ by pattern and palette rather than by subject, which is also
 * what keeps any of them from reading as a depiction of a person.
 */

export type AvatarPreset = {
  key: string;
  label: string;
  /** Nine sticker colours, reading left-to-right, top-to-bottom. */
  stickers: string[];
};

// The six cube colours, matching the renderer's palette and the analytics
// page's swatches so a face here is the same yellow as a face there.
const R = "#D42A3D";
const O = "#FF5F0F";
const W = "#F4F6F9";
const Y = "#FFD41F";
const G = "#00A067";
const B = "#0A5AC4";

/** Shorthand: a 9-sticker face from a pattern string over two colours. */
function face(pattern: string, a: string, b: string): string[] {
  return pattern.split("").map((c) => (c === "x" ? a : b));
}

export const AVATAR_PRESETS: AvatarPreset[] = [
  { key: "cube-classic", label: "Solved", stickers: face("xxxxxxxxx", W, W) },
  { key: "cube-checker", label: "Checker", stickers: face("xoxoxoxox", Y, B) },
  { key: "cube-cross", label: "Cross", stickers: face("oxoxxxoxo", G, W) },
  { key: "cube-stripe", label: "Stripe", stickers: face("xxxoooxxx", R, W) },
  { key: "cube-corners", label: "Corners", stickers: face("xoxoooxox", B, W) },
  { key: "cube-spiral", label: "Spiral", stickers: face("xxoxooxxx", O, Y) },
  { key: "cube-sunset", label: "Sunset", stickers: [R, R, O, R, O, Y, O, Y, Y] },
  { key: "cube-ocean", label: "Ocean", stickers: [B, B, G, B, G, G, G, G, W] },
  { key: "cube-forest", label: "Forest", stickers: [G, G, Y, G, Y, G, Y, G, G] },
  { key: "cube-mono", label: "Mono", stickers: face("xoxoxoxox", W, "#2A2E35") },
  { key: "cube-neon", label: "Neon", stickers: [G, B, G, B, W, B, G, B, G] },
  { key: "cube-ember", label: "Ember", stickers: [O, R, O, R, Y, R, O, R, O] },
];

export const AVATAR_BY_KEY = Object.fromEntries(
  AVATAR_PRESETS.map((a) => [a.key, a]),
) as Record<string, AvatarPreset>;

/**
 * One preset, as an SVG cube face.
 *
 * `viewBox` is fixed and every dimension is relative to it, so a single
 * definition scales from the 28px nav button to the 128px profile card
 * without a second asset or a blurry upscale.
 */
export function PresetAvatar({
  preset,
  size = 64,
  className,
}: {
  preset: string;
  size?: number;
  className?: string;
}) {
  const spec = AVATAR_BY_KEY[preset];
  if (!spec) return null;

  const gap = 3;
  const cell = 28;
  const pad = 5;
  return (
    <svg
      viewBox="0 0 100 100"
      width={size}
      height={size}
      className={className}
      role="img"
      aria-label={`${spec.label} cube avatar`}
    >
      <rect x="0" y="0" width="100" height="100" rx="50" fill="#16181D" />
      {spec.stickers.map((colour, i) => (
        <rect
          key={i}
          x={pad + (i % 3) * (cell + gap)}
          y={pad + Math.floor(i / 3) * (cell + gap)}
          width={cell}
          height={cell}
          rx={6}
          fill={colour}
        />
      ))}
    </svg>
  );
}
