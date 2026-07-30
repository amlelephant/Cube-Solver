export type SolveRecord = {
  date: string;
  opponent: string;
  time: string;
  result: "win" | "loss" | "solo";
};

export const solveHistory: SolveRecord[] = [
  { date: "Jul 21, 2026", opponent: "Solo trial", time: "15.61s", result: "solo" },
  { date: "Jul 20, 2026", opponent: "vs. mira_cubes", time: "12.98s", result: "win" },
  { date: "Jul 19, 2026", opponent: "Solo trial", time: "16.40s", result: "solo" },
  { date: "Jul 18, 2026", opponent: "vs. speedy_dan", time: "14.22s", result: "loss" },
  { date: "Jul 17, 2026", opponent: "Solo trial", time: "13.87s", result: "solo" },
  { date: "Jul 16, 2026", opponent: "vs. jperm_fan42", time: "11.75s", result: "win" },
  { date: "Jul 15, 2026", opponent: "Solo trial", time: "17.03s", result: "solo" },
  { date: "Jul 14, 2026", opponent: "vs. cubewizard", time: "13.02s", result: "win" },
  { date: "Jul 13, 2026", opponent: "Solo trial", time: "15.98s", result: "solo" },
];

export type LeaderboardEntry = {
  rank: number;
  name: string;
  rating: number;
  best: string;
  country: string;
  isYou?: boolean;
};

// Rating and best-time are independent metrics here on purpose — rating rank
// (feliks/mira/jperm) and time rank (mira/speedy/cubewizard) genuinely disagree,
// so the leaderboard's time/rating toggle shows a different podium, not a relabel.
export const leaderboard: LeaderboardEntry[] = [
  { rank: 1, name: "feliks_zx", rating: 2340, best: "5.51s", country: "AU" },
  { rank: 2, name: "mira_cubes", rating: 2198, best: "4.89s", country: "JP" },
  { rank: 3, name: "jperm_fan42", rating: 2107, best: "6.20s", country: "US" },
  { rank: 4, name: "speedy_dan", rating: 1972, best: "5.12s", country: "GB" },
  { rank: 5, name: "cubewizard", rating: 1888, best: "5.44s", country: "KR" },
  { rank: 6, name: "torch_bearer", rating: 1804, best: "7.61s", country: "DE" },
  { rank: 7, name: "algcrunch", rating: 1755, best: "6.01s", country: "CA" },
  { rank: 8, name: "olltwist", rating: 1690, best: "6.71s", country: "FR" },
  { rank: 9, name: "f2l_finn", rating: 1622, best: "6.33s", country: "SE" },
  { rank: 10, name: "crossmaster", rating: 1560, best: "7.02s", country: "BR" },
];

export const you: LeaderboardEntry & { bestRank: number; isFounder: boolean } = {
  rank: 47,
  name: "you",
  rating: 1240,
  best: "15.61s",
  country: "US",
  isYou: true,
  // Best global rank ever reached — cosmetics (laurel wreaths) are earned by
  // this, not by current standing, so a past top-2 finish keeps the silver
  // and bronze wreaths unlocked even after slipping to #47.
  bestRank: 2,
  // Hand-granted, never rank-earned: the founder's olive is only ever on the
  // two founder accounts. Real deployments must set this server-side — a flag
  // the client can edit is a cosmetic anyone could mint for themselves.
  isFounder: true,
};

export const profileMetrics = [
  { label: "3x3 best solve", value: "11.75s" },
  { label: "3x3 best ao 10", value: "13.90s" },
  { label: "Average cross time", value: "2.10s" },
  { label: "Total solves", value: "1,204" },
  { label: "Average turns per second", value: "6.8 TPS" },
];

// Deterministic pseudo-random generator so server and client render the same
// values (Math.random()/Date.now() would mismatch during hydration).
function seededRandom(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

// Average of every solve in the history table, plus a daily streak — the
// "little metrics" surfaced on the home page next to the profile deep link.
export const homeStats = (() => {
  const seconds = solveHistory.map((r) => parseFloat(r.time));
  const average = seconds.reduce((sum, s) => sum + s, 0) / seconds.length;
  return {
    averageSolve: `${average.toFixed(2)}s`,
    dailyStreak: solveHistory.length,
  };
})();

function generateEloSeries(length: number, seed: number, start: number): number[] {
  const gen = seededRandom(seed);
  let rating = start;
  const points: number[] = [];
  for (let i = 0; i < length; i++) {
    rating += Math.round((gen() - 0.35) * 40);
    rating = Math.max(600, rating);
    points.push(rating);
  }
  points[points.length - 1] = 1240;
  return points;
}

// Each window is its own series (not a slice of one long history) so a wider
// window shows real earlier volatility instead of a flat lead-in.
export const eloHistoryWindows = {
  "24 solves": generateEloSeries(24, 42, 950),
  "3 months": generateEloSeries(48, 108, 1040),
  "1 year": generateEloSeries(80, 205, 780),
  "All time": generateEloSeries(140, 311, 620),
} satisfies Record<string, number[]>;

export type EloWindow = keyof typeof eloHistoryWindows;
export const eloWindowOptions = Object.keys(eloHistoryWindows) as EloWindow[];

const heatmapRand = seededRandom(77);

// 26 weeks x 7 days (~6 months) of solve-count intensity, 0 (none) to 4
// (busiest) — sized to read as a recognizable GitHub-style contribution grid.
export const solveHeatmap: number[][] = Array.from({ length: 26 }, () =>
  Array.from({ length: 7 }, () => Math.floor(heatmapRand() * 5)),
);

export type SolvePhase = { label: string; seconds: number; pct: number };

export type SolveDetail = {
  scramble: string;
  phases: SolvePhase[];
  moveCount: number;
  tps: string;
  ratingDelta: number | null;
};

const SCRAMBLE_FACES = ["U", "D", "L", "R", "F", "B"] as const;
const SCRAMBLE_SUFFIXES = ["", "'", "2"];

function generateScramble(seed: number, length = 20): string {
  const gen = seededRandom(seed);
  const moves: string[] = [];
  let lastFace: string | null = null;
  for (let i = 0; i < length; i++) {
    let face: string;
    do {
      face = SCRAMBLE_FACES[Math.floor(gen() * SCRAMBLE_FACES.length)];
    } while (face === lastFace);
    lastFace = face;
    const suffix = SCRAMBLE_SUFFIXES[Math.floor(gen() * SCRAMBLE_SUFFIXES.length)];
    moves.push(face + suffix);
  }
  return moves.join(" ");
}

const PHASE_WEIGHTS: [string, number][] = [
  ["Inspection", 0.05],
  ["Cross", 0.14],
  ["F2L", 0.46],
  ["OLL", 0.18],
  ["PLL", 0.17],
];

function phaseBreakdown(seed: number, totalSeconds: number): SolvePhase[] {
  const gen = seededRandom(seed);
  const jittered = PHASE_WEIGHTS.map(([label, weight]) => ({
    label,
    weight: Math.max(0.03, weight + (gen() - 0.5) * 0.06),
  }));
  const totalWeight = jittered.reduce((sum, p) => sum + p.weight, 0);
  return jittered.map(({ label, weight }) => {
    const pct = weight / totalWeight;
    return { label, seconds: Number((pct * totalSeconds).toFixed(2)), pct };
  });
}

/** Fabricated but deterministic per-solve breakdown, keyed by row index in solveHistory. */
export function getSolveDetail(index: number): SolveDetail | null {
  const record = solveHistory[index];
  if (!record) return null;

  const totalSeconds = parseFloat(record.time);
  const seed = 900 + index * 37;
  const phases = phaseBreakdown(seed, totalSeconds);
  const moveCount = Math.round(totalSeconds * (6.2 + seededRandom(seed + 1)() * 1.4));
  const tps = (moveCount / totalSeconds).toFixed(2);
  const ratingDelta = record.result === "win" ? 18 : record.result === "loss" ? -12 : null;

  return {
    scramble: generateScramble(seed + 2),
    phases,
    moveCount,
    tps,
    ratingDelta,
  };
}
