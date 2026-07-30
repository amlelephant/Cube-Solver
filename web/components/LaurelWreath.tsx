import { cn } from "@/lib/cn";

export type WreathTier = "gold" | "silver" | "bronze" | "founder";

/**
 * Each tier is a ramp, not a colour: every leaf is filled at its own point
 * between `deep` and `bright`. A wreath drawn in one flat tone reads as a
 * decal — the per-leaf variation is what makes it look like metal catching
 * light at different angles.
 *
 * `founder` is the exception and deliberately not a metal: a living olive
 * branch, olive-green with only the brightest few leaves reaching green-yellow,
 * on a woody stem. `fruit` gives it olives — unevenly ripened, as they are on
 * a real tree — and is what marks a tier as fruiting at all.
 */
const TIER_TONES: Record<
  WreathTier,
  { deep: string; bright: string; stem: string; fruit?: [string, string] }
> = {
  gold: { deep: "#8a6a1c", bright: "#f0cc63", stem: "#7d5f18" },
  silver: { deep: "#6e7683", bright: "#c9d0da", stem: "#636a76" },
  bronze: { deep: "#7d4a20", bright: "#d9955a", stem: "#6f4220" },
  founder: {
    deep: "#3f5a28",
    bright: "#c8d96e",
    stem: "#5d5326",
    // Ripe olives really do go dark plum-brown, and that is the only thing in
    // this palette that isn't a green — which is exactly why the fruit reads.
    // A green-on-green olive (tried first) vanishes into the foliage.
    fruit: ["#3b2b33", "#a8bb4e"], // ripe, and still-green
  },
};

export const TIER_LABEL: Record<WreathTier, string> = {
  gold: "Gold laurel",
  silver: "Silver laurel",
  bronze: "Bronze laurel",
  founder: "Founder's olive",
};

/**
 * Worn as a crown across the forehead, not a ring around the avatar — the
 * arrangement from the Cube Arena Figma file ("Laurel wreat example", node
 * 14:3), whose fronds start just outside the disc at the sides and finish
 * short of top-centre *inside* it. That inside finish is why this paints over
 * the avatar rather than behind it.
 *
 * Each frond is described in polar terms rather than as a chord, because the
 * radius is what makes it look worn: it stays out at the disc's edge through
 * the first part of the sweep, so the end follows the head's curve as though
 * continuing around the back, and only draws inward as it climbs toward the
 * tip. `R_EASE` above 1 is what delays that — a straight radial interpolation
 * cuts the corner immediately and reads as a stick laid across the top.
 *
 * Over the last third the radius then flares back OUT (`flare`). Without it
 * the tips keep curling inward and run into each other at top-centre; with it
 * each frond lifts away from the head at the end and the two finish short of
 * one another, open crown between them.
 *
 * Angles are SVG-convention degrees from +x with y pointing down: 172 is just
 * below the left ear, 252 is up and inward but well short of top-centre (270).
 */
const R_EASE = 1.9;
const FLARE_FROM = 0.62; // fraction of the sweep after which the tip lifts

/**
 * The two fronds are deliberately NOT mirror images. Same construction, small
 * differences in where they start and finish, how far they reach, and how
 * their leaves are jittered — a real wreath is two cuttings, not a reflection.
 * `flip` mirrors the polar sweep onto the other side of the head; `salt` seeds
 * that frond's jitter.
 */
const FRONDS = [
  { flip: false, a0: 172, a1: 252, r0: 1.0, r1: 0.58, flare: 0.13, pad: 0, salt: 3.1 },
  { flip: true, a0: 167, a1: 247, r0: 1.03, r1: 0.62, flare: 0.11, pad: 1, salt: 8.7 },
];

/** Leaves rake harder toward the tip, shallower down at the base. */
const LEAF_TILT = [54, 32];

/**
 * Taper, jitter and leaf width all have to ease off as the avatar shrinks. At
 * 128px a 0.12r tip leaf knocked down by a 0.77x length draw is a fine detail;
 * at 56px it is under three pixels and reads as a speck, and the whole wreath
 * turns into a spray of needles. `detail` is 0 on the smallest avatars and 1
 * from 128px up, and everything irregular is interpolated against it — so
 * small sizes get fatter, more uniform leaves that still read as a wreath.
 */
function metrics(size: number) {
  const d = clamp((size - 48) / 80, 0, 1);
  const lerp = (a: number, b: number) => a + (b - a) * d;
  return {
    len: [0.3, lerp(0.21, 0.12)], // × radius, base of frond -> tip
    aspect: lerp(0.42, 0.32), // half-width control offset, fraction of length
    jitLen: lerp(0.16, 0.46), // spread of the length multiplier
    jitTilt: lerp(9, 22), // degrees
    jitRoot: lerp(0.02, 0.05), // × radius, off the stem centreline
    jitT: lerp(0.02, 0.04), // along the stem
    bend: lerp(0.45, 0.9),
  };
}

/**
 * Base intensities, cycled along the frond and then jittered. The cycle keeps
 * bright and deep leaves from clumping the way a pure random draw does; the
 * jitter keeps the cycle from being visible as a repeat.
 */
const LEAF_SHADE = [0.92, 0.34, 0.72, 0.16, 1.0, 0.46, 0.82, 0.28, 0.64, 0.4];

/**
 * Reference is drawn at 500px; twelve leaf pairs of detail turns to mush on a
 * 56px row avatar, so the count follows the space available.
 */
function leafPairs(size: number) {
  return Math.max(7, Math.min(12, Math.round(size / 9)));
}

/** Stable hash — jitter has to survive a re-render, so no Math.random(). */
function rnd(i: number, salt: number) {
  const x = Math.sin(i * 12.9898 + salt * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

/** Straight-line blend between two #rrggbb strings. */
function mixHex(a: string, b: string, t: number) {
  const ch = (s: string, i: number) => parseInt(s.slice(1 + i * 2, 3 + i * 2), 16);
  const v = [0, 1, 2].map((i) => Math.round(ch(a, i) + (ch(b, i) - ch(a, i)) * t));
  return `#${v.map((n) => n.toString(16).padStart(2, "0")).join("")}`;
}

const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);
const f = (n: number) => n.toFixed(2);

type Frond = (typeof FRONDS)[number];
type Pt = { x: number; y: number };

/** Point on a frond's stem at 0 (base, at the ear) to 1 (tip, near the top). */
function stemAt(fr: Frond, t: number, radius: number): Pt {
  const ang = ((fr.a0 + (fr.a1 - fr.a0) * t) * Math.PI) / 180;
  const lift = Math.pow(Math.max(0, (t - FLARE_FROM) / (1 - FLARE_FROM)), 2) * fr.flare;
  const rad = (fr.r0 + (fr.r1 - fr.r0) * Math.pow(t, R_EASE) + lift) * radius;
  return { x: Math.cos(ang) * rad * (fr.flip ? -1 : 1), y: Math.sin(ang) * rad };
}

/** Unit tangent, base -> tip, by central difference. */
function tangentAt(fr: Frond, t: number, radius: number) {
  const h = 0.01;
  const a = stemAt(fr, Math.max(0, t - h), radius);
  const b = stemAt(fr, Math.min(1, t + h), radius);
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const l = Math.hypot(dx, dy) || 1;
  return { x: dx / l, y: dy / l };
}

function stemPath(fr: Frond, radius: number) {
  const pts = Array.from({ length: 26 }, (_, i) => {
    const p = stemAt(fr, i / 25, radius);
    return `${f(p.x)},${f(p.y)}`;
  });
  return `M${pts.join("L")}`;
}

type Leaf = { root: Pt; dx: number; dy: number; len: number; bend: number; shade: number };

function fronLeaves(fr: Frond, size: number): Leaf[] {
  const radius = size / 2;
  const m = metrics(size);
  const pairs = leafPairs(size) + fr.pad;
  const step = 1 / Math.max(1, pairs - 1);
  const out: Leaf[] = [];

  for (let i = 0; i < pairs; i++) {
    for (const side of [1, -1]) {
      const k = i * 2 + (side > 0 ? 0 : 1);
      // The two sides are offset half a step apart rather than paired, so the
      // frond alternates like a real one instead of reading as a ladder.
      const stagger = side > 0 ? 0 : step * 0.5;
      const nominal = 0.07 + 0.93 * (i * step) + stagger;
      // Positional jitter is scaled down toward the base. Leaves are longest
      // there, so the same nudge that reads as pleasant scatter among the small
      // tip leaves leaves a big base leaf visibly stranded off the mass.
      const pos = 0.35 + 0.65 * nominal;
      const t = clamp(nominal + (rnd(k, fr.salt) - 0.5) * m.jitT * pos, 0, 1);

      const p = stemAt(fr, t, radius);
      const u = tangentAt(fr, t, radius);
      const perp = { x: -u.y * side, y: u.x * side };

      const tilt =
        ((LEAF_TILT[0] +
          (LEAF_TILT[1] - LEAF_TILT[0]) * t +
          (rnd(k, fr.salt + 1) - 0.5) * m.jitTilt) *
          side *
          Math.PI) /
        180;
      const len =
        radius *
        (m.len[0] + (m.len[1] - m.len[0]) * t) *
        (1 - m.jitLen / 2 + rnd(k, fr.salt + 2) * m.jitLen);
      const off = (rnd(k, fr.salt + 3) - 0.5) * m.jitRoot * radius * pos;

      out.push({
        root: { x: p.x + perp.x * off, y: p.y + perp.y * off },
        dx: u.x * Math.cos(tilt) - u.y * Math.sin(tilt),
        dy: u.x * Math.sin(tilt) + u.y * Math.cos(tilt),
        len,
        bend: (rnd(k, fr.salt + 4) - 0.5) * m.bend,
        shade: clamp(
          LEAF_SHADE[k % LEAF_SHADE.length] + (rnd(k, fr.salt + 5) - 0.5) * 0.24,
          0.08,
          1,
        ),
      });
    }
  }

  // terminal leaf, straight along the stem, so the tip looks finished
  const p = stemAt(fr, 1, radius);
  const u = tangentAt(fr, 1, radius);
  out.push({
    root: p,
    dx: u.x,
    dy: u.y,
    len: radius * m.len[1] * 1.15,
    bend: 0.2,
    shade: 0.86,
  });
  return out;
}

type Olive = { x: number; y: number; r: number; stalk: Pt; ripe: boolean };

/**
 * Olives nestle in among the leaves on short stalks, a few per frond, sitting
 * alternately either side of the stem. They are drawn last so they read as
 * fruit resting on the foliage rather than buried under it.
 */
function frondOlives(fr: Frond, size: number): Olive[] {
  const radius = size / 2;
  // Below ~64px an olive is a two-pixel dot; a fourth one is just noise.
  const count = size >= 64 ? 4 : 3;
  return Array.from({ length: count }, (_, i) => {
    const t = clamp(
      0.2 + 0.52 * (i / Math.max(1, count - 1)) + (rnd(i, fr.salt + 7) - 0.5) * 0.09,
      0,
      1,
    );
    const p = stemAt(fr, t, radius);
    const u = tangentAt(fr, t, radius);
    const side = i % 2 === 0 ? 1 : -1;
    const reach = radius * (0.07 + rnd(i, fr.salt + 8) * 0.05);
    return {
      x: p.x - u.y * side * reach,
      y: p.y + u.x * side * reach,
      r: radius * (0.075 + rnd(i, fr.salt + 9) * 0.03),
      stalk: p,
      // a roughly even mix — uneven ripening is the tell that these are fruit
      ripe: rnd(i, fr.salt + 10) > 0.5,
    };
  });
}

/**
 * Lanceolate leaf, pointed at both ends and widest at 42%. `bend` swings the
 * tip off-axis so leaves curve rather than sitting as straight spikes.
 */
function leafPath({ root, dx, dy, len, bend }: Leaf, aspect: number) {
  const px = -dy;
  const py = dx;
  const tx = root.x + dx * len + px * len * bend * 0.2;
  const ty = root.y + dy * len + py * len * bend * 0.2;
  const mx = root.x + dx * len * 0.42 + px * len * bend * 0.12;
  const my = root.y + dy * len * 0.42 + py * len * bend * 0.12;
  const w = len * aspect;
  return (
    `M${f(root.x)},${f(root.y)}` +
    `Q${f(mx + px * w)},${f(my + py * w)} ${f(tx)},${f(ty)}` +
    `Q${f(mx - px * w)},${f(my - py * w)} ${f(root.x)},${f(root.y)}Z`
  );
}

/**
 * Earned cosmetic for top-3 rankings — gold/silver/bronze. Sized to the avatar
 * it crowns and painted over it; leaf tips overhang the box by roughly 13% of
 * `size` at the sides, which `overflow-visible` lets through.
 */
export function LaurelWreath({
  tier,
  size = 100,
  className,
}: {
  tier: WreathTier;
  size?: number;
  className?: string;
}) {
  const tone = TIER_TONES[tier];
  const radius = size / 2;
  const { aspect } = metrics(size);

  return (
    <svg
      viewBox={`${-size / 2} ${-size / 2} ${size} ${size}`}
      width={size}
      height={size}
      className={cn("overflow-visible", className)}
      aria-hidden
    >
      {FRONDS.map((fr, fi) => (
        <g key={fi}>
          <path
            d={stemPath(fr, radius)}
            fill="none"
            stroke={tone.stem}
            strokeWidth={Math.max(1, size * 0.013)}
            strokeLinecap="round"
          />
          {fronLeaves(fr, size).map((l, i) => (
            <path key={i} d={leafPath(l, aspect)} fill={mixHex(tone.deep, tone.bright, l.shade)} />
          ))}
          {tone.fruit &&
            frondOlives(fr, size).map((o, i) => (
              <g key={`o${i}`}>
                <line
                  x1={f(o.stalk.x)}
                  y1={f(o.stalk.y)}
                  x2={f(o.x)}
                  y2={f(o.y)}
                  stroke={tone.stem}
                  strokeWidth={Math.max(0.6, size * 0.008)}
                  strokeLinecap="round"
                />
                {/* thin rim in the stem tone: without it a green olive melts
                    into the leaf it overlaps, on either page theme */}
                <circle
                  cx={f(o.x)}
                  cy={f(o.y)}
                  r={f(o.r)}
                  fill={o.ripe ? tone.fruit![0] : tone.fruit![1]}
                  stroke={tone.stem}
                  strokeWidth={Math.max(0.5, size * 0.006)}
                />
              </g>
            ))}
        </g>
      ))}
    </svg>
  );
}
