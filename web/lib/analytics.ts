/**
 * analytics.ts — the coach's L1 output, typed for the web app.
 *
 * PROVENANCE, BECAUSE IT MATTERS HERE
 * -----------------------------------
 * Solve values are fetched from `GET /api/solves/analysis/`, which returns
 * `coach/report.py`'s `solve_report()` payload as it was stored on each
 * solve. They are the signed-in account's own solves — nothing on this
 * page is shared between accounts, and an account with no analysed solves
 * gets an empty state rather than someone else's numbers.
 *
 * `METRICS` below stays static, and deliberately. It carries only what the
 * server has no business deciding — the display label, the one-line blurb,
 * the ordering — plus a mirror of `coach/report.py`'s `MEASURED` registry
 * for the accuracy footnote. That registry IS the gate on what may be
 * shown, so a metric absent there must be absent here. The gate is applied
 * server-side: `suppressed` and each value's `confidence` arrive already
 * decided, and this file never re-derives them.
 *
 * The accuracy figures are real — median and worst-session relative error
 * against BLE ground truth on held-out solves, from `metric_robustness.py`,
 * both seeds. They stay held-out figures even when the solves on screen
 * were decoded by a model that trained on them; a number measured on
 * training data is not an accuracy claim and must not be published as one.
 */

import { api } from "@/lib/api";

export type FaceKey = "U" | "D" | "L" | "R" | "F" | "B";
export type Confidence = "high" | "caution" | "suppressed";
/**
 * `unknown` is what the server reports when `lighting_check.py` has not run
 * for a solve. It is treated as `evening` everywhere below, never as
 * daytime: the point of the gate is that an unmeasured regime must not
 * borrow daytime's accuracy claim. Same rule as `coach/report.py`.
 */
export type Regime = "daytime" | "evening" | "unknown";

export interface MetricSpec {
  key: string;
  label: string;
  /**
   * One short line. Deliberately terse: explaining what a metric MEANS is
   * the coach's job, and long captions here duplicated it badly.
   */
  blurb: string;
  unit: "s" | "TPS" | "QTM" | "frac" | "cv" | "0-1" | "moves" | "frac/face";
  /** mean | local — see metric_robustness.STAT_KIND. */
  kind: "mean" | "local";
  dayErrPct: number;
  dayWorstPct: number;
  eveErrPct: number;
  eveWorstPct: number;
}

/**
 * The shipped registry, mirroring `coach/report.py`'s `MEASURED` tuple.
 * A metric absent there must be absent here — that registry is a gate, and
 * duplicating it loosely would defeat the point of having one.
 */
export const METRICS: MetricSpec[] = [
  {
    key: "span_seconds",
    label: "Solve time",
    blurb: "First move to last.",
    unit: "s",
    kind: "mean",
    dayErrPct: 0.1,
    dayWorstPct: 0.2,
    eveErrPct: 1.0,
    eveWorstPct: 2.0,
  },
  {
    key: "n_moves_qtm",
    label: "Moves",
    blurb: "Quarter turns.",
    unit: "QTM",
    kind: "mean",
    dayErrPct: 5.3,
    dayWorstPct: 12.1,
    eveErrPct: 2.8,
    eveWorstPct: 14.3,
  },
  {
    key: "span_tps",
    label: "Turns per second",
    blurb: "Moves over solve time.",
    unit: "TPS",
    kind: "mean",
    dayErrPct: 5.4,
    dayWorstPct: 12.3,
    eveErrPct: 2.8,
    eveWorstPct: 16.2,
  },
  {
    key: "execution_tps",
    label: "Execution speed",
    blurb: "Turn rate with hesitation removed.",
    unit: "TPS",
    kind: "mean",
    dayErrPct: 2.7,
    dayWorstPct: 13.8,
    eveErrPct: 8.6,
    eveWorstPct: 17.9,
  },
  {
    key: "mean_move_duration_s",
    label: "Average move",
    blurb: "Time per turn while executing.",
    unit: "s",
    kind: "mean",
    dayErrPct: 2.8,
    dayWorstPct: 16.0,
    eveErrPct: 7.9,
    eveWorstPct: 15.2,
  },
  {
    key: "median_move_duration_s",
    label: "Typical move",
    blurb: "Median turn.",
    unit: "s",
    kind: "mean",
    dayErrPct: 6.0,
    dayWorstPct: 13.0,
    eveErrPct: 11.2,
    eveWorstPct: 23.1,
  },
  {
    key: "move_duration_cv",
    label: "Turn consistency",
    blurb: "Spread of your turn timings. Lower is steadier.",
    unit: "cv",
    kind: "mean",
    dayErrPct: 5.1,
    dayWorstPct: 9.1,
    eveErrPct: 16.2,
    eveWorstPct: 19.6,
  },
  {
    key: "hesitation_seconds",
    label: "Time spent thinking",
    blurb: "Time between moves that was thinking.",
    unit: "s",
    kind: "mean",
    dayErrPct: 4.8,
    dayWorstPct: 6.5,
    eveErrPct: 20.8,
    eveWorstPct: 24.2,
  },
  {
    key: "hesitation_fraction",
    label: "Share of solve thinking",
    blurb: "Share of the solve spent thinking.",
    unit: "frac",
    kind: "mean",
    dayErrPct: 4.6,
    dayWorstPct: 6.5,
    eveErrPct: 21.7,
    eveWorstPct: 22.0,
  },
  {
    key: "ccw_fraction",
    label: "Counter-clockwise share",
    blurb: "Share of turns counter-clockwise.",
    unit: "frac",
    kind: "mean",
    dayErrPct: 5.6,
    dayWorstPct: 9.0,
    eveErrPct: 12.9,
    eveWorstPct: 15.2,
  },
  {
    key: "face_share",
    label: "Face usage",
    blurb: "Which faces you turn.",
    unit: "frac/face",
    kind: "mean",
    dayErrPct: 2.5,
    dayWorstPct: 6.4,
    eveErrPct: 4.5,
    eveWorstPct: 7.9,
  },
  {
    key: "top_face_share",
    label: "Most-used face",
    blurb: "Share on your most-used face.",
    unit: "frac",
    kind: "mean",
    dayErrPct: 3.5,
    dayWorstPct: 13.8,
    eveErrPct: 10.9,
    eveWorstPct: 16.7,
  },
  {
    key: "easy_face_fraction",
    label: "R/U share",
    blurb: "Share on R and U — no regrip needed.",
    unit: "frac",
    kind: "mean",
    dayErrPct: 1.5,
    dayWorstPct: 5.9,
    eveErrPct: 5.6,
    eveWorstPct: 7.1,
  },
  {
    key: "face_entropy",
    label: "Face-usage spread",
    blurb: "How evenly the six faces are used.",
    unit: "0-1",
    kind: "mean",
    dayErrPct: 0.5,
    dayWorstPct: 4.5,
    eveErrPct: 4.9,
    eveWorstPct: 6.6,
  },
  {
    key: "distinct_face_runs",
    label: "Face changes per move",
    blurb: "How often consecutive moves switch face.",
    unit: "frac",
    kind: "local",
    dayErrPct: 1.9,
    dayWorstPct: 2.7,
    eveErrPct: 7.9,
    eveWorstPct: 8.5,
  },
  {
    key: "moves_per_face_run",
    label: "Moves per face run",
    blurb: "Turns on one face before moving on.",
    unit: "moves",
    kind: "local",
    dayErrPct: 1.9,
    dayWorstPct: 2.7,
    eveErrPct: 8.6,
    eveWorstPct: 9.3,
  },
];

export const METRIC_BY_KEY = Object.fromEntries(
  METRICS.map((m) => [m.key, m]),
) as Record<string, MetricSpec>;

/** URFDLB order, matching the server's `FACES`. */
export const FACE_KEYS: FaceKey[] = ["U", "R", "F", "D", "L", "B"];

/** The three fields every view renders from, single solve or average. */
export interface MetricView {
  values: Record<string, number>;
  faceShare: Record<FaceKey, number>;
  /** Keys not reportable — withheld by the server, or with no contributors. */
  suppressed: string[];
}

export interface SolveAnalysis extends MetricView {
  id: string;
  /** ISO date, for the trend chart's x axis. */
  date: string;
  /**
   * Lighting the solve was recorded in. NOT shown to the viewer — time of
   * day is irrelevant to cubing and surfacing it invited the wrong reading.
   * It survives because it is what decides `suppressed`.
   */
  regime: Regime;
}

export interface Aggregate extends MetricView {
  /** Solves in the window. */
  n: number;
  /** How many solves actually contributed to each metric. */
  counts: Record<string, number>;
  /**
   * Mean solve time over exactly the solves that reported hesitation.
   *
   * Exists because `span_seconds - hesitation_seconds` is a trap when
   * averaging. Evening solves withhold hesitation but still report a span,
   * so the two means are built on different sets of solves and subtracting
   * them yields a "turning time" belonging to no solve at all. The
   * turning/thinking split must use this, not `values.span_seconds`.
   * Undefined when no solve in the window reported hesitation.
   */
  hesitationSpanSeconds?: number;
  /**
   * Per metric, the worst regime among its contributors — which is the
   * regime its accuracy claim must be read in. Per-metric rather than one
   * value for the whole aggregate, because a metric that evening suppresses
   * is averaged over daytime solves ONLY and would be defamed by inheriting
   * an evening error bar it never touched.
   */
  regimes: Record<string, Regime>;
}

export interface AnalysisResponse {
  solves: SolveAnalysis[];
  is_premium: boolean;
  /** True when the server withheld solves behind the paid tier. */
  truncated: boolean;
  /** How many analysed solves the account actually has. */
  total: number;
}

/**
 * Mean of each metric across a window of solves.
 *
 * THE MEAN IS OVER SOLVES, NOT OVER MOVES. `span_tps` here is the average
 * of each solve's own turns-per-second, not total turns over total time.
 * They are different numbers and the first is the one a cuber means by "my
 * average TPS" — the second silently weights long solves more heavily,
 * which is the opposite of how an ao12 reads.
 *
 * A metric is skipped for any solve that withheld it, so an evening solve
 * drops out of the hesitation average without dragging the other metrics'
 * sample size down with it. `counts` records what each average was actually
 * built from; a metric no solve in the window reported comes back
 * suppressed rather than as a spurious zero.
 *
 * Face share is averaged per face and renormalised, so the six still sum to
 * one after solves that never turned a given face are folded in.
 */
export function averageOf(solves: SolveAnalysis[]): Aggregate {
  const sums: Record<string, number> = {};
  const counts: Record<string, number> = {};
  // Per metric: did anything other than a daytime solve contribute? That is
  // the only distinction the error bars draw — `errorOf` reads `unknown` and
  // `evening` identically, on purpose — so collapsing to a flag rather than
  // ranking three regimes keeps this honest and readable at once.
  const dim: Record<string, boolean> = {};
  // Span accumulated over hesitation-reporting solves only — see
  // `hesitationSpanSeconds`.
  let hSpan = 0;
  let hSpanN = 0;

  for (const s of solves) {
    for (const m of METRICS) {
      if (m.key === "face_share") continue;
      const v = s.values[m.key];
      if (v === undefined || s.suppressed.includes(m.key)) continue;
      sums[m.key] = (sums[m.key] ?? 0) + v;
      counts[m.key] = (counts[m.key] ?? 0) + 1;
      if (s.regime !== "daytime") dim[m.key] = true;
    }
    const hasHes =
      s.values.hesitation_seconds !== undefined &&
      !s.suppressed.includes("hesitation_seconds") &&
      s.values.span_seconds !== undefined;
    if (hasHes) {
      hSpan += s.values.span_seconds;
      hSpanN += 1;
    }
  }

  const values: Record<string, number> = {};
  const suppressed: string[] = [];
  const regimes: Record<string, Regime> = {};
  for (const m of METRICS) {
    if (m.key === "face_share") continue;
    if (counts[m.key]) {
      values[m.key] = sums[m.key] / counts[m.key];
      regimes[m.key] = dim[m.key] ? "evening" : "daytime";
    } else {
      suppressed.push(m.key);
    }
  }

  const faceTotals = Object.fromEntries(
    FACE_KEYS.map((f) => [
      f,
      solves.reduce((a, s) => a + (s.faceShare?.[f] ?? 0), 0),
    ]),
  ) as Record<FaceKey, number>;
  const faceSum = FACE_KEYS.reduce((a, f) => a + faceTotals[f], 0) || 1;
  const faceShare = Object.fromEntries(
    FACE_KEYS.map((f) => [f, faceTotals[f] / faceSum]),
  ) as Record<FaceKey, number>;

  return {
    values,
    faceShare,
    suppressed,
    counts,
    regimes,
    n: solves.length,
    hesitationSpanSeconds: hSpanN ? hSpan / hSpanN : undefined,
  };
}

/**
 * Windows the average can be taken over. `0` means every analysed solve.
 * Paid accounts choose freely — it is their data, and a fixed window would
 * be us deciding which of their own numbers they are allowed to see.
 */
export const WINDOWS = [
  { key: "5", label: "Last 5", n: 5 },
  { key: "12", label: "Last 12", n: 12 },
  { key: "all", label: "All time", n: 0 },
] as const;

export type WindowKey = (typeof WINDOWS)[number]["key"];

/** The last `n` solves, or all of them when `n` is 0. */
export function windowOf(solves: SolveAnalysis[], n: number): SolveAnalysis[] {
  return n > 0 ? solves.slice(-n) : solves;
}

/**
 * The signed-in account's analysed solves, oldest first.
 *
 * Chronological rather than the API's usual newest-first: this list IS the
 * trend chart's x axis, and a caller that had to know to reverse it is a
 * caller that will one day forget.
 */
export async function fetchSolves(): Promise<AnalysisResponse> {
  const data = await api<AnalysisResponse>("/api/solves/analysis/");
  return {
    solves: data?.solves ?? [],
    is_premium: !!data?.is_premium,
    truncated: !!data?.truncated,
    total: data?.total ?? 0,
  };
}

export interface SolveDetail extends SolveAnalysis {
  is_premium: boolean;
  solve: {
    id: number;
    verdict: string;
    seconds: number;
    observed_moves: number;
    tps: number | null;
    reject_reasons: string[];
    review_reasons: string[];
    reverified: boolean;
    received_at: string;
  };
  result: "solo" | "win" | "loss";
  opponent: string | null;
  rating_delta: number | null;
  scramble: string;
}

/**
 * One solve's analysis. Throws `ApiError` with status 402 when the solve is
 * behind the paid tier and 404 when it is not the caller's — the caller
 * should render those differently, so they are not collapsed here.
 */
export function fetchSolveDetail(id: string | number) {
  return api<SolveDetail>(`/api/solves/${id}/analysis/`);
}

/** Chronological series for one metric, skipping solves where it was withheld. */
export function seriesFor(solves: SolveAnalysis[], key: string) {
  return solves
    .filter((s) => !s.suppressed.includes(key) && s.values[key] !== undefined)
    .map((s) => ({ x: s.date, y: s.values[key], id: s.id }));
}

export function confidenceOf(m: MetricSpec, regime: Regime): Confidence {
  const err = errorOf(m, regime).median;
  if (err > 15) return "suppressed";
  return err > 8 ? "caution" : "high";
}

export function errorOf(m: MetricSpec, regime: Regime) {
  return regime === "daytime"
    ? { median: m.dayErrPct, worst: m.dayWorstPct }
    : { median: m.eveErrPct, worst: m.eveWorstPct };
}

export function formatValue(unit: MetricSpec["unit"], v: number): string {
  switch (unit) {
    case "frac":
      return `${(v * 100).toFixed(1)}%`;
    case "s":
      return v < 1 ? `${v.toFixed(3)}s` : `${v.toFixed(2)}s`;
    case "TPS":
      return v.toFixed(2);
    case "QTM":
      return v.toFixed(0);
    case "moves":
      return v.toFixed(2);
    default:
      return v.toFixed(3);
  }
}
