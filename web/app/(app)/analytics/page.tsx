"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Card } from "@/components/ui/Card";
import { CubeViz } from "@/components/CubeViz";
import { MetricTrend } from "@/components/charts/MetricTrend";
import { cn } from "@/lib/cn";
import {
  METRIC_BY_KEY,
  METRICS,
  WINDOWS,
  averageOf,
  errorOf,
  fetchSolves,
  formatValue,
  seriesFor,
  windowOf,
  type Aggregate,
  type AnalysisResponse,
  type FaceKey,
  type MetricView,
  type WindowKey,
} from "@/lib/analytics";

/**
 * /analytics — the numbers, shown on the cube.
 *
 * Reads the signed-in account's own solves from `/api/solves/analysis/`.
 *
 * Copy is deliberately thin. Explaining what a metric MEANS is the coach's
 * job; this page displays what was measured and lets the cube carry the
 * intuition. An earlier draft narrated every section and read like filler.
 *
 * Two things are deliberately absent. There is no time-of-day selector:
 * lighting decides which metrics the server will report, but it is not
 * something a cuber should be asked to think about, so it stays internal.
 * And per-figure error bars are gone from the tiles — they are one footnote
 * at the bottom, because a number the product chose to display is a number
 * it stands behind.
 */

/** Loops back to solved every 6 repetitions, so the demo never drifts into
 *  a state the viewer cannot interpret. */
const DEMO_ALG = "R U R' U'";

const FACE_LABEL: Record<FaceKey, string> = {
  U: "Up",
  D: "Down",
  L: "Left",
  R: "Right",
  F: "Front",
  B: "Back",
};
const FACE_SWATCH: Record<FaceKey, string> = {
  R: "#D42A3D",
  L: "#FF5F0F",
  U: "#F4F6F9",
  D: "#FFD41F",
  F: "#00A067",
  B: "#0A5AC4",
};

function Stat({
  label,
  value,
  sub,
  active,
  onClick,
  size = "md",
}: {
  label: string;
  value: string;
  sub?: string;
  active?: boolean;
  onClick?: () => void;
  size?: "md" | "lg";
}) {
  const Tag = onClick ? "button" : "div";
  return (
    <Tag
      type={onClick ? "button" : undefined}
      onClick={onClick}
      aria-pressed={onClick ? !!active : undefined}
      className={cn(
        "rounded-xl border p-4 text-left transition-colors",
        onClick && "hover:border-ink-faint cursor-pointer",
        active ? "border-ink bg-cloud" : "border-mist",
      )}
    >
      <span className="block text-xs font-medium uppercase tracking-wide text-ink-faint">
        {label}
      </span>
      <span
        className={cn(
          "mt-1 block font-semibold tracking-tight",
          size === "lg" ? "text-3xl" : "text-2xl",
        )}
      >
        {value}
      </span>
      {sub && <span className="mt-0.5 block text-xs text-ink-faint">{sub}</span>}
    </Tag>
  );
}

function Shell({
  subtitle,
  children,
}: {
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mx-auto max-w-6xl px-6 py-10 md:px-10">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">Analytics</h1>
        <p className="mt-1 text-sm text-ink-soft">{subtitle}</p>
      </header>
      {children}
    </div>
  );
}

export default function AnalyticsPage() {
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    fetchSolves()
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e?.message ?? "Could not load analytics."));
    return () => {
      live = false;
    };
  }, []);

  if (error) {
    return (
      <Shell subtitle="Your solves, move by move.">
        <Card className="mt-8 p-6 md:p-8">
          <p className="text-sm text-ink-soft">{error}</p>
        </Card>
      </Shell>
    );
  }

  if (data === null) {
    return (
      <Shell subtitle="Your solves, move by move.">
        <div className="mt-8 flex min-h-[40vh] items-center justify-center">
          <span className="sr-only">Loading your solves…</span>
          <span className="size-5 animate-pulse rounded-[4px] cube-gradient" aria-hidden />
        </div>
      </Shell>
    );
  }

  if (data.solves.length === 0) {
    return (
      <Shell subtitle="Your solves, move by move.">
        <Card className="mt-8 p-6 md:p-8">
          <p className="text-sm text-ink-soft">
            Nothing to show yet. Analytics appear once you have a solve that has
            been analysed — record one from the Play tab.
          </p>
        </Card>
      </Shell>
    );
  }

  return <AnalyticsView data={data} />;
}

/** Segmented control. Used for both the speed toggle and the window picker. */
function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: readonly { key: T; label: string; disabled?: boolean }[];
  value: T;
  onChange: (k: T) => void;
  label: string;
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className="inline-flex rounded-lg border border-mist p-0.5"
    >
      {options.map((o) => (
        <button
          key={o.key}
          type="button"
          disabled={o.disabled}
          aria-pressed={value === o.key}
          onClick={() => onChange(o.key)}
          className={cn(
            "rounded-[6px] px-3 py-1.5 text-xs font-medium transition-colors",
            o.disabled && "cursor-not-allowed opacity-40",
            value === o.key
              ? "bg-ink text-paper"
              : !o.disabled && "text-ink-soft hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function AnalyticsView({ data }: { data: AnalysisResponse }) {
  const { solves, is_premium: premium } = data;
  const [speedMode, setSpeedMode] = useState<"execution_tps" | "span_tps">(
    "execution_tps",
  );
  const [trendKey, setTrendKey] = useState("span_seconds");
  // Paid accounts default to the last 12 rather than all-time: on a
  // improving cuber an all-time mean sits above current form and moves
  // slower the longer you play, which reads as "I am not getting better".
  // All-time stays one click away — see WINDOWS.
  const [windowKey, setWindowKey] = useState<WindowKey | "last">(
    premium ? "12" : "last",
  );

  // A free account is SERVED one solve, so there is nothing here to average
  // even if this state were forced. The gate is the response, not this line.
  const windowed = useMemo(() => {
    if (windowKey === "last") return solves.slice(-1);
    const spec = WINDOWS.find((w) => w.key === windowKey) ?? WINDOWS[1];
    return windowOf(solves, spec.n);
  }, [solves, windowKey]);

  const view: MetricView = useMemo(
    () =>
      windowKey === "last" ? windowed[windowed.length - 1] : averageOf(windowed),
    [windowed, windowKey],
  );
  const solve = view;
  const averaging = windowKey !== "last";
  const shownCount = windowed.length;

  // For a single solve the two are the same number by construction; for an
  // average they are not — see `Aggregate.hesitationSpanSeconds`.
  const agg = averaging ? (view as Aggregate) : null;
  const hesSpan = agg?.hesitationSpanSeconds ?? solve.values.span_seconds;
  const hesCount = agg?.counts.hesitation_seconds ?? 1;

  const tps = solve.values[speedMode];

  // Usage share -> brightness, normalised to the busiest face so the
  // contrast describes THIS solve rather than an absolute scale no solve
  // reaches.
  //
  // The exponent is doing real work. A LINEAR ramp did not read at all:
  // hue dominates perceived brightness, so a green face at 27% albedo
  // still just looks green, and the viewer has no reference for how bright
  // that green could have been. Raising the ratio to a power pushes the
  // rarely-used faces far enough down that they read as switched off,
  // which is the only comparison that works across different hues. The
  // floor keeps them present as faces rather than holes.
  const faceGain = useMemo(() => {
    const top = Math.max(...Object.values(solve.faceShare), 0.001);
    return Object.fromEntries(
      (Object.keys(solve.faceShare) as FaceKey[]).map((f) => [
        f,
        0.08 + 0.92 * Math.pow(solve.faceShare[f] / top, 1.8),
      ]),
    ) as Record<FaceKey, number>;
  }, [solve]);

  const facesSorted = useMemo(
    () =>
      (Object.keys(solve.faceShare) as FaceKey[]).sort(
        (a, b) => solve.faceShare[b] - solve.faceShare[a],
      ),
    [solve],
  );
  const topShare = Math.max(...Object.values(solve.faceShare));

  const trendSpec = METRIC_BY_KEY[trendKey];
  const trendPoints = useMemo(
    () => seriesFor(solves, trendKey),
    [solves, trendKey],
  );
  const shown = METRICS.filter((m) => m.key !== "face_share");

  return (
    <Shell
      subtitle={
        averaging
          ? `Your average over ${shownCount} ${shownCount === 1 ? "solve" : "solves"}.`
          : "Your last solve, move by move."
      }
    >
      {/* ---------------- window ---------------- */}
      <section className="mt-6">
        {premium ? (
          <div className="flex flex-wrap items-center gap-3">
            <Segmented
              label="Averaging window"
              value={windowKey}
              onChange={setWindowKey}
              options={[
                { key: "last" as const, label: "Last solve" },
                ...WINDOWS.map((w) => ({
                  key: w.key as WindowKey,
                  label: w.label,
                  // Offering "Last 12" on an account with 6 solves would
                  // silently show a 6-solve mean under a 12-solve label.
                  disabled: w.n > 0 && solves.length < w.n,
                })),
              ]}
            />
            <span className="text-xs text-ink-faint">
              {averaging
                ? `Averaged over ${shownCount} of your ${solves.length} analysed solves.`
                : `Showing 1 of your ${solves.length} analysed solves.`}
            </span>
          </div>
        ) : (
          <Card className="flex flex-wrap items-center justify-between gap-4 p-5">
            <div>
              <p className="text-sm font-medium">Your last solve</p>
              <p className="mt-0.5 text-sm text-ink-soft">
                {data.total > 1
                  ? `Coach averages these over your last 5, 12 or all ${data.total} solves, and opens any one of them.`
                  : "Coach averages your solves over time, and opens any one of them."}
              </p>
            </div>
            <Link
              href="/settings"
              className="shrink-0 rounded-lg bg-ink px-4 py-2 text-sm font-medium text-paper transition-opacity hover:opacity-90"
            >
              Get Coach
            </Link>
          </Card>
        )}
      </section>

      {/* ---------------- speed ---------------- */}
      <section className="mt-8">
        <Card className="overflow-hidden">
          <div className="grid gap-0 md:grid-cols-[1fr_1.1fr]">
            <div className="relative min-h-[340px] bg-cloud">
              <CubeViz
                className="absolute inset-0"
                alg={DEMO_ALG}
                tps={tps}
                spin={5}
                label={`A cube turning at ${tps.toFixed(2)} turns per second`}
              />
              <span className="absolute bottom-3 left-4 font-mono text-xs text-ink-faint">
                R U R&apos; U&apos; · {tps.toFixed(2)} TPS
              </span>
            </div>

            <div className="p-6 md:p-8">
              <h2 className="text-xl font-semibold tracking-tight">Turn speed</h2>
              <p className="mt-1 text-sm text-ink-soft">
                Pick one to drive the cube.
              </p>

              <div className="mt-5 grid grid-cols-2 gap-3">
                <Stat
                  size="lg"
                  label="Execution speed"
                  value={formatValue("TPS", solve.values.execution_tps)}
                  sub="hesitation removed"
                  active={speedMode === "execution_tps"}
                  onClick={() => setSpeedMode("execution_tps")}
                />
                <Stat
                  size="lg"
                  label="Overall"
                  value={formatValue("TPS", solve.values.span_tps)}
                  sub="whole solve"
                  active={speedMode === "span_tps"}
                  onClick={() => setSpeedMode("span_tps")}
                />
                <Stat
                  label="Average move"
                  value={formatValue("s", solve.values.mean_move_duration_s)}
                />
                <Stat
                  label="Turn consistency"
                  value={
                    solve.values.move_duration_cv === undefined
                      ? "—"
                      : formatValue("cv", solve.values.move_duration_cv)
                  }
                  sub="lower is steadier"
                />
              </div>
            </div>
          </div>
        </Card>
      </section>

      {/* ---------------- hesitation ---------------- */}
      <section className="mt-6">
        <Card className="p-6 md:p-8">
          <h2 className="text-xl font-semibold tracking-tight">Turning vs thinking</h2>
          {solve.values.hesitation_fraction === undefined ? (
            <p className="mt-4 text-sm text-ink-faint">Not measured for this solve.</p>
          ) : (
            <>
              <div className="mt-5 flex h-11 w-full overflow-hidden rounded-lg">
                <div
                  className="flex items-center justify-center bg-ink text-xs font-medium text-paper"
                  style={{ width: `${(1 - solve.values.hesitation_fraction) * 100}%` }}
                >
                  turning
                </div>
                <div
                  className="flex items-center justify-center bg-mist text-xs font-medium text-ink-soft"
                  style={{ width: `${solve.values.hesitation_fraction * 100}%` }}
                >
                  thinking
                </div>
              </div>
              <div className="mt-2 flex justify-between font-mono text-xs text-ink-faint">
                {/* NOT `values.span_seconds - values.hesitation_seconds`.
                    When averaging, evening solves contribute a span but
                    withhold hesitation, so those two means cover different
                    sets of solves and their difference is a turning time no
                    solve actually had. `hesitationSpanSeconds` is the span
                    over exactly the hesitation contributors. */}
                <span>{(hesSpan - solve.values.hesitation_seconds).toFixed(1)}s</span>
                <span>{solve.values.hesitation_seconds.toFixed(1)}s</span>
              </div>
              {averaging && hesCount < shownCount && (
                <p className="mt-2 text-xs text-ink-faint">
                  From {hesCount} of these {shownCount} solves — thinking time
                  is not reported in dim lighting, and those solves are left
                  out rather than counted as no hesitation.
                </p>
              )}
            </>
          )}

          <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="Solve time" value={formatValue("s", solve.values.span_seconds)} />
            <Stat label="Moves" value={formatValue("QTM", solve.values.n_moves_qtm)} sub="quarter turns" />
            <Stat
              label="Thinking"
              value={
                solve.values.hesitation_seconds === undefined
                  ? "—"
                  : formatValue("s", solve.values.hesitation_seconds)
              }
            />
            <Stat
              label="Share thinking"
              value={
                solve.values.hesitation_fraction === undefined
                  ? "—"
                  : formatValue("frac", solve.values.hesitation_fraction)
              }
            />
          </div>
        </Card>
      </section>

      {/* ---------------- face usage ---------------- */}
      <section className="mt-6">
        <Card className="overflow-hidden">
          <div className="grid gap-0 md:grid-cols-[1.1fr_1fr]">
            <div className="p-6 md:p-8">
              <h2 className="text-xl font-semibold tracking-tight">Face usage</h2>
              <p className="mt-1 text-sm text-ink-soft">
                Brighter means more turns. Drag the cube to see the back.
              </p>

              <ul className="mt-5 space-y-2">
                {facesSorted.map((f) => (
                  <li key={f} className="flex items-center gap-3">
                    <span
                      className="size-3 shrink-0 rounded-[3px] border border-mist"
                      style={{ background: FACE_SWATCH[f] }}
                      aria-hidden
                    />
                    <span className="w-14 shrink-0 text-sm text-ink-soft">
                      {FACE_LABEL[f]}
                    </span>
                    <span className="h-2 flex-1 overflow-hidden rounded-full bg-cloud">
                      <span
                        className="block h-full rounded-full bg-ink"
                        style={{ width: `${(solve.faceShare[f] / topShare) * 100}%` }}
                      />
                    </span>
                    <span className="w-12 shrink-0 text-right font-mono text-xs text-ink-faint">
                      {(solve.faceShare[f] * 100).toFixed(0)}%
                    </span>
                  </li>
                ))}
              </ul>

              <div className="mt-6 grid grid-cols-2 gap-3">
                <Stat
                  label="R/U share"
                  value={formatValue("frac", solve.values.easy_face_fraction)}
                  sub="no regrip needed"
                />
                <Stat
                  label="Counter-clockwise"
                  value={formatValue("frac", solve.values.ccw_fraction)}
                />
              </div>
            </div>

            <div className="relative min-h-[380px] touch-none bg-cloud">
              {/* Still, not spinning: face identity here is position-relative
                  — "up" is whichever face is up — so a cube rotating on its
                  own detaches the labels from what is on screen. The viewer
                  drags to look, and it eases back to the same front. */}
              <CubeViz
                className="absolute inset-0"
                faceGain={faceGain}
                tps={0}
                spin={0}
                interactive
                label={
                  "A still cube, each face lit in proportion to how often it was turned: " +
                  facesSorted
                    .map((f) => `${FACE_LABEL[f]} ${(solve.faceShare[f] * 100).toFixed(0)} percent`)
                    .join(", ")
                }
              />
            </div>
          </div>
        </Card>
      </section>

      {/* ---------------- everything ---------------- */}
      <section className="mt-6">
        <Card className="p-6 md:p-8">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-xl font-semibold tracking-tight">
              {premium ? "Trends" : "Everything measured"}
            </h2>
            <p className="text-sm text-ink-soft">
              {premium
                ? "Tap any measure to chart it."
                : "This solve, in full."}
            </p>
          </div>

          {premium ? (
            <>
              <div className="mt-5 flex flex-wrap items-baseline justify-between gap-2 border-t border-mist pt-4">
                <span className="text-sm font-medium">{trendSpec?.label}</span>
                <span className="text-xs text-ink-faint">{trendSpec?.blurb}</span>
              </div>

              <div className="mt-2">
                {/* The chart is every analysed solve, NOT the averaging
                    window: a trend drawn only over the window you are
                    averaging cannot show you the window moving. */}
                <MetricTrend
                  points={trendPoints}
                  label={trendSpec?.label ?? ""}
                  format={(v) => formatValue(trendSpec?.unit ?? "0-1", v)}
                />
              </div>
            </>
          ) : (
            <p className="mt-5 border-t border-mist pt-4 text-sm text-ink-soft">
              Charting a measure over time is part of Coach.
            </p>
          )}

          <div className="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
            {shown.map((m) => {
              const v = solve.values[m.key];
              const missing = v === undefined || solve.suppressed.includes(m.key);
              return (
                <Stat
                  key={m.key}
                  label={m.label}
                  value={missing ? "—" : formatValue(m.unit, v)}
                  active={premium && trendKey === m.key}
                  onClick={premium ? () => setTrendKey(m.key) : undefined}
                />
              );
            })}
          </div>
        </Card>
      </section>

      <footer className="mt-8 space-y-1.5 text-xs leading-relaxed text-ink-faint">
        <p>
          Rates are QTM — a half turn is two moves, so these read higher than a
          timer that counts it as one.
        </p>
        <p>
          Figures are decoded from video and checked against a smart cube on
          solves the model never saw: typically within{" "}
          {Math.max(...shown.map((m) => errorOf(m, "daytime").median)).toFixed(0)}% ,
          and metrics we cannot measure well enough in the recording conditions
          are left blank rather than guessed.
          {averaging && (
            <>
              {" "}
              That figure is <em>per solve</em>. Averaging several will in
              practice do better, but by how much is not something we have
              measured, so the number above is not adjusted for it.
            </>
          )}
        </p>
        <p>
          {averaging
            ? `Averaged over ${shownCount} ${shownCount === 1 ? "solve" : "solves"}; a measure withheld for a given solve is left out of its average rather than counted as zero.`
            : `1 solve shown, of ${data.total} analysed.`}
        </p>
      </footer>
    </Shell>
  );
}
