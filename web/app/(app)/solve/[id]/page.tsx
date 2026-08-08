"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/cn";
import {
  FACE_KEYS,
  METRICS,
  errorOf,
  fetchSolveDetail,
  formatValue,
  type FaceKey,
  type SolveDetail,
} from "@/lib/analytics";

/**
 * /solve/<id> — one solve, in full.
 *
 * Reached by clicking a row in your own profile history. Opening any solve
 * but your most recent is part of Coach, and the SERVER enforces that: this
 * page renders the 402 as an upgrade prompt, it does not decide the rule.
 *
 * NO PHASE BREAKDOWN. An earlier draft of this page rendered cross / F2L /
 * OLL / PLL splits from `lib/mockData`. Phase splits are not built (TODO
 * §7E) and inventing them here would have been the one thing this project
 * cannot afford — a number on screen that no measurement stands behind.
 * What is shown instead is the coach payload, which is measured, and which
 * carries its own accuracy.
 */

const FACE_LABEL: Record<FaceKey, string> = {
  U: "Up",
  D: "Down",
  L: "Left",
  R: "Right",
  F: "Front",
  B: "Back",
};

const resultColor = {
  win: "text-cube-green",
  loss: "text-cube-red",
  solo: "text-ink-faint",
} as const;

const resultLabel = {
  win: "Win",
  loss: "Loss",
  solo: "Solo trial",
} as const;

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-3xl px-6 pb-24 pt-16 md:px-10">
      <Link
        href="/profile"
        className="inline-flex items-center gap-1 text-sm font-medium text-ink-faint transition-colors hover:text-ink"
      >
        <ArrowLeft size={16} /> Back to profile
      </Link>
      {children}
    </div>
  );
}

export default function SolveDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [detail, setDetail] = useState<SolveDetail | null>(null);
  const [status, setStatus] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    fetchSolveDetail(id)
      .then((d) => live && setDetail(d))
      .catch((e) => {
        if (!live) return;
        setStatus(e instanceof ApiError ? e.status : null);
        setError(e?.message ?? "Could not load this solve.");
      });
    return () => {
      live = false;
    };
  }, [id]);

  if (status === 402) {
    return (
      <Shell>
        <Card className="mt-8 p-6 md:p-8">
          <p className="text-lg font-medium">This solve is part of Coach</p>
          <p className="mt-2 text-sm text-ink-soft">
            Free accounts can open their most recent analysed solve. Coach opens
            every solve you have recorded, and averages them.
          </p>
          <Link
            href="/settings"
            className="mt-5 inline-block rounded-lg bg-ink px-4 py-2 text-sm font-medium text-paper transition-opacity hover:opacity-90"
          >
            Get Coach
          </Link>
        </Card>
      </Shell>
    );
  }

  if (error) {
    return (
      <Shell>
        <Card className="mt-8 p-6 md:p-8">
          <p className="text-sm text-ink-soft">
            {status === 404 ? "No analysis for that solve." : error}
          </p>
        </Card>
      </Shell>
    );
  }

  if (!detail) {
    return (
      <Shell>
        <div className="mt-8 flex min-h-[40vh] items-center justify-center">
          <span className="sr-only">Loading this solve…</span>
          <span className="size-5 animate-pulse rounded-[4px] cube-gradient" aria-hidden />
        </div>
      </Shell>
    );
  }

  const { solve, values, faceShare, suppressed } = detail;
  const facesSorted = [...FACE_KEYS].sort((a, b) => faceShare[b] - faceShare[a]);
  const topShare = Math.max(...FACE_KEYS.map((f) => faceShare[f]), 0.001);
  const worstErr = Math.max(
    ...METRICS.filter((m) => !suppressed.includes(m.key)).map(
      (m) => errorOf(m, detail.regime).worst,
    ),
  );

  return (
    <Shell>
      <div className="mt-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-ink-faint">
            {new Date(solve.received_at).toLocaleString(undefined, {
              dateStyle: "medium",
              timeStyle: "short",
            })}
          </p>
          <p className="mt-1 text-2xl font-medium">
            {detail.opponent ? `vs. ${detail.opponent}` : "Solo trial"}
          </p>
        </div>
        <span
          className={cn(
            "rounded-full bg-cloud px-3 py-1 text-sm font-medium",
            resultColor[detail.result],
          )}
        >
          {resultLabel[detail.result]}
        </span>
      </div>

      {/* The SERVER-derived duration, not the coach's move span — this is
          the number the verdict rests on. */}
      <p className="mt-6 text-7xl font-semibold tracking-tight cube-gradient-text">
        {solve.seconds.toFixed(2)}s
      </p>

      <div className="mt-2 flex flex-wrap items-center gap-3 text-sm font-medium text-ink-faint">
        <span className="capitalize">{solve.verdict}</span>
        <span aria-hidden>·</span>
        <span>{solve.observed_moves} moves</span>
        {detail.rating_delta !== null && (
          <>
            <span aria-hidden>·</span>
            <span>
              Rating {detail.rating_delta > 0 ? "+" : ""}
              {detail.rating_delta}
            </span>
          </>
        )}
      </div>

      <Card className="mt-10 p-6 md:p-8">
        <p className="text-lg font-medium">Measured</p>
        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3">
          {METRICS.filter((m) => m.key !== "face_share").map((m) => {
            const v = values[m.key];
            const missing = v === undefined || suppressed.includes(m.key);
            return (
              <div key={m.key} className="rounded-xl border border-mist p-4">
                <span className="block text-xs font-medium uppercase tracking-wide text-ink-faint">
                  {m.label}
                </span>
                <span className="mt-1 block text-xl font-semibold tracking-tight">
                  {missing ? "—" : formatValue(m.unit, v)}
                </span>
              </div>
            );
          })}
        </div>
        {suppressed.length > 0 && (
          <p className="mt-4 text-xs text-ink-faint">
            {suppressed.length} {suppressed.length === 1 ? "measure was" : "measures were"}{" "}
            withheld — we could not measure {suppressed.length === 1 ? "it" : "them"}{" "}
            accurately enough in this recording&apos;s lighting, so{" "}
            {suppressed.length === 1 ? "it is" : "they are"} left blank rather
            than guessed.
          </p>
        )}
      </Card>

      <Card className="mt-6 p-6 md:p-8">
        <p className="text-lg font-medium">Face usage</p>
        <ul className="mt-5 space-y-2">
          {facesSorted.map((f) => (
            <li key={f} className="flex items-center gap-3">
              <span className="w-14 shrink-0 text-sm text-ink-soft">
                {FACE_LABEL[f]}
              </span>
              <span className="h-2 flex-1 overflow-hidden rounded-full bg-cloud">
                <span
                  className="block h-full rounded-full bg-ink"
                  style={{ width: `${(faceShare[f] / topShare) * 100}%` }}
                />
              </span>
              <span className="w-12 shrink-0 text-right font-mono text-xs text-ink-faint">
                {(faceShare[f] * 100).toFixed(0)}%
              </span>
            </li>
          ))}
        </ul>
      </Card>

      <Card className="mt-6 p-6 md:p-8">
        <p className="text-lg font-medium">Scramble</p>
        <p className="mt-3 break-words font-mono text-sm leading-relaxed text-ink-soft">
          {detail.scramble}
        </p>
      </Card>

      <p className="mt-6 text-xs leading-relaxed text-ink-faint">
        Rates are QTM — a half turn is two moves. These are one solve&apos;s
        figures, so read them at the worst case rather than the average: on a
        bad solve the least reliable measure above has been seen{" "}
        {worstErr.toFixed(0)}% out against a smart cube.{" "}
        {detail.is_premium && (
          <Link href="/analytics" className="underline underline-offset-2">
            Averaging several is steadier.
          </Link>
        )}
      </p>
    </Shell>
  );
}
