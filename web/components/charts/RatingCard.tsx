"use client";

import Link from "next/link";
import { Swords } from "lucide-react";
import { EloChart } from "@/components/charts/EloChart";
import { Card } from "@/components/ui/Card";
import type { SolveRow } from "@/lib/account";

/**
 * Rating over time — and, when there is none, an invitation to make some.
 *
 * The empty state is the point of this card existing at launch. Solo trials
 * are unrated, so a new account has a flat 1000 and nothing to plot; hiding
 * the card until a first match would mean the one surface that explains
 * *why* you would play a match only appears after you already have. So it
 * stays, and says what it is missing.
 */

/** How many rated solves before a line is worth drawing rather than a dot. */
const MIN_POINTS = 2;

/**
 * Rating after each rated solve, oldest first.
 *
 * Reconstructed by walking `rating_delta` BACKWARDS from the account's
 * current rating, because that is the only rating the server actually
 * stores — there is no rating-history table. Walking forwards from a
 * presumed starting value would drift the moment any adjustment happened
 * outside a solve.
 *
 * `rows` arrives newest-first (the API's order); the returned series is
 * chronological, which is what a chart's x axis means.
 */
export function ratingSeries(rows: SolveRow[], currentRating: number): number[] {
  const rated = rows.filter((r) => r.rating_delta != null);
  if (!rated.length) return [];

  // rated[0] is the most recent, and its delta is already inside
  // `currentRating`. Peel deltas off to recover each earlier value.
  const series: number[] = [currentRating];
  let running = currentRating;
  for (const row of rated) {
    running -= row.rating_delta ?? 0;
    series.push(running);
  }
  series.reverse();
  return series;
}

export function RatingCard({
  rows,
  rating,
  isYou,
}: {
  rows: SolveRow[];
  rating: number;
  isYou: boolean;
}) {
  const series = ratingSeries(rows, rating);
  const enough = series.length >= MIN_POINTS;
  const change = enough ? series[series.length - 1] - series[0] : 0;

  return (
    <Card className="flex h-full flex-col p-8">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-xl font-medium">Rating</p>
        <p className="text-sm text-ink-faint">
          {enough ? (
            <>
              {change >= 0 ? "+" : ""}
              {change} over {series.length - 1}{" "}
              {series.length - 1 === 1 ? "match" : "matches"}
            </>
          ) : (
            "Unrated so far"
          )}
        </p>
      </div>

      {enough ? (
        <div className="mt-6 flex-1">
          <EloChart points={series} />
        </div>
      ) : (
        <div className="mt-6 flex flex-1 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-mist px-6 py-10 text-center">
          <Swords size={28} strokeWidth={1.25} className="text-ink-faint" aria-hidden />
          <p className="text-3xl font-semibold tracking-tight">{rating}</p>
          <p className="max-w-xs text-sm text-ink-soft">
            {isYou
              ? "Solo trials are unrated. Play a match and your rating starts moving."
              : "This player has not played a rated match yet."}
          </p>
          {isYou && (
            <Link
              href="/compete"
              className="mt-1 rounded-lg bg-ink px-4 py-2 text-sm font-medium text-paper transition-opacity hover:opacity-90"
            >
              Find a match
            </Link>
          )}
        </div>
      )}
    </Card>
  );
}
