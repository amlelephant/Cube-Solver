"use client";

import Link from "next/link";
import { ChevronRight } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { PlayerLink } from "@/components/PlayerLink";
import { cn } from "@/lib/cn";
import type { SolveRow } from "@/lib/account";

/**
 * The solve history table. ONE component, used by both `/home` and a
 * profile — they showed the same data in two implementations that had
 * already drifted apart on row height, date format and whether a row was
 * clickable.
 *
 * It replaced `HistoryTable`, which took a `SolveRecord` shaped around mock
 * data — a pre-formatted `time` string, a display-ready `opponent` label like
 * "vs. mira_cubes", and a row index for its link. None of those survive
 * contact with a database: an opponent has to be a username you can link to,
 * and a solve has to be identified by its own id rather than its position in
 * an array.
 */

const resultColor: Record<SolveRow["result"], string> = {
  win: "text-cube-green",
  loss: "text-cube-red",
  solo: "text-ink-faint",
};

export function RecentSolves({
  title = "Current Best",
  bestSeconds,
  bestSolveId,
  rows,
  limit,
  /**
   * Whether rows open the solve. Off for someone else's profile: their solve
   * analysis is 404 by design, so a link there is an invitation to a dead
   * end.
   */
  linkRows = true,
  emptyLabel = "No solves yet. Your first verified solve shows up here.",
}: {
  title?: string;
  bestSeconds: number | null;
  bestSolveId?: number | null;
  rows: SolveRow[];
  limit?: number;
  linkRows?: boolean;
  emptyLabel?: string;
}) {
  const shown = limit ? rows.slice(0, limit) : rows;
  // The header is a link only when there is somewhere to go: an account with
  // no verified solve has a best of "—", and a button leading nowhere is
  // worse than plain text.
  const bestHref =
    linkRows && bestSolveId != null ? `/solve/${bestSolveId}` : null;

  const best = (
    <span className="cube-gradient-text text-lg font-semibold">
      {bestSeconds == null ? "—" : `${bestSeconds.toFixed(2)}s`}
    </span>
  );

  return (
    <Card className="overflow-hidden p-0">
      <div className="flex items-center justify-between gap-4 bg-cloud px-6 py-4">
        <p className="text-lg font-medium">{title}</p>
        {bestHref ? (
          <Link
            href={bestHref}
            title="Open your personal best"
            className={cn(
              "-my-1 -mr-2 inline-flex items-center gap-1 rounded-lg px-2 py-1",
              "transition-colors hover:bg-paper/70",
            )}
          >
            {best}
            <ChevronRight size={16} className="text-ink-faint" aria-hidden />
          </Link>
        ) : (
          best
        )}
      </div>

      {shown.length === 0 ? (
        <p className="px-6 py-10 text-center text-sm text-ink-faint">{emptyLabel}</p>
      ) : (
        shown.map((row, i) => {
          const className = cn(
            "flex items-center justify-between gap-4 px-6 py-4 text-sm",
            i !== shown.length - 1 && "border-b border-mist",
            linkRows && "transition-colors hover:bg-cloud",
          );
          const cells = (
            <>
              <span className="w-32 shrink-0 text-ink-faint">
                {new Date(row.received_at).toLocaleDateString(undefined, {
                  month: "short",
                  day: "numeric",
                  year: "numeric",
                })}
              </span>
              <span className="flex-1 text-center font-medium">
                {row.opponent ? (
                  <>
                    vs. <PlayerLink username={row.opponent} className="text-ink" />
                  </>
                ) : (
                  <span className="text-ink-faint">Solo trial</span>
                )}
              </span>
              <span
                className={cn(
                  "w-24 shrink-0 text-right font-medium",
                  resultColor[row.result],
                )}
              >
                {row.seconds.toFixed(2)}s
              </span>
            </>
          );

          return linkRows ? (
            <Link key={row.id} href={`/solve/${row.id}`} className={className}>
              {cells}
            </Link>
          ) : (
            <div key={row.id} className={className}>
              {cells}
            </div>
          );
        })
      )}
    </Card>
  );
}
