"use client";

import Link from "next/link";
import { ArrowUpRight, Swords, Timer } from "lucide-react";
import { Avatar } from "@/components/Avatar";
import { Card } from "@/components/ui/Card";
import { RecentSolves } from "@/components/RecentSolves";
import { useCosmetics } from "@/lib/cosmetics";
import { useMe } from "@/lib/account";
import { Flag } from "@/components/Flag";
import { profileHref } from "@/components/PlayerLink";

/** Rows the home summary shows before pointing at the full profile history. */
const HOME_HISTORY_ROWS = 9;

export default function HomePage() {
  const { equippedWreath } = useCosmetics();
  const { me } = useMe();

  const lastSolve = me?.solves?.[0];
  const average = averageSeconds(me?.solves ?? []);

  return (
    <div className="mx-auto max-w-6xl px-6 pb-24 md:px-10">
      <section className="cube-gradient relative -mx-6 mt-6 flex h-72 items-center justify-center overflow-hidden rounded-3xl text-center md:-mx-10 md:h-96">
        <div className="absolute inset-0 bg-black/45" />
        <div className="relative px-6">
          <h1 className="text-4xl font-bold tracking-tight text-white md:text-6xl">
            Welcome home{me ? `, ${me.username}` : ""}
          </h1>
          <p className="mx-auto mt-3 max-w-lg text-balance text-white/80">
            {lastSolve
              ? `Your last verified solve was ${lastSolve.seconds.toFixed(2)}s. Ready to beat it?`
              : "No solves yet — record your first one and it lands here."}
          </p>
        </div>
      </section>

      <section className="mt-8 grid grid-cols-1 gap-6 md:grid-cols-[225px_1fr]">
        <Card className="flex flex-col items-center justify-center gap-3 p-8">
          <Avatar size={64} wreath={equippedWreath} preset={me?.avatar} />
          <p className="flex items-center gap-1.5 text-sm font-medium text-ink-faint">
            <Flag code={me?.country} decorative />
            {me?.username ?? "Your avatar"}
          </p>
        </Card>

        <Card className="flex flex-col gap-6 p-8">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <p className="text-2xl font-medium">Analytics</p>
            <Link
              href={me ? profileHref(me.username) : "/profile"}
              className="flex shrink-0 items-center gap-1 whitespace-nowrap text-sm font-medium text-ink transition-opacity hover:opacity-60"
            >
              More data <ArrowUpRight size={16} />
            </Link>
          </div>

          <div className="grid grid-cols-2 gap-4 border-t border-mist pt-6 sm:max-w-xs">
            <div>
              <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">
                Average solve
              </p>
              <p className="mt-1 text-2xl font-semibold">
                {average == null ? "—" : `${average.toFixed(2)}s`}
              </p>
            </div>
            <div>
              <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">
                Daily streak
              </p>
              <p className="mt-1 text-2xl font-semibold">
                {dailyStreak(me?.solves ?? [])} days
              </p>
            </div>
          </div>
        </Card>
      </section>

      <section className="mt-8 grid grid-cols-1 gap-6 md:grid-cols-2">
        <Link href="/compete/play?mode=solo" className="group">
          <Card className="flex h-96 flex-col overflow-hidden p-0 transition-shadow group-hover:shadow-lg">
            <div className="flex flex-1 items-center justify-center bg-cloud">
              <Timer size={64} strokeWidth={1.25} className="text-ink-faint" />
            </div>
            <div className="border-t border-mist py-6 text-center">
              <p className="text-2xl font-medium">Solo Practice</p>
            </div>
          </Card>
        </Link>

        <Link href="/compete/play?mode=live" className="group">
          <Card className="flex h-96 flex-col overflow-hidden p-0 transition-shadow group-hover:shadow-lg">
            <div className="flex flex-1 items-center justify-center bg-cloud">
              <Swords size={64} strokeWidth={1.25} className="text-ink-faint" />
            </div>
            <div className="border-t border-mist py-6 text-center">
              <p className="text-2xl font-medium">Live Match</p>
            </div>
          </Card>
        </Link>
      </section>

      {/* Same component and same props as the profile's History, so the two
          cannot drift apart again — only the row cap differs, because this
          is a summary and that is the record. */}
      <section className="mt-8">
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
          <p className="text-2xl text-ink-faint">History</p>
          {me && (me.solves?.length ?? 0) > HOME_HISTORY_ROWS && (
            <Link
              href={profileHref(me.username)}
              className="flex items-center gap-1 text-sm font-medium text-ink transition-opacity hover:opacity-60"
            >
              All {me.total_solves} solves <ArrowUpRight size={16} />
            </Link>
          )}
        </div>
        <RecentSolves
          rows={me?.solves ?? []}
          bestSeconds={me?.best_seconds ?? null}
          bestSolveId={me?.best_solve_id}
          limit={HOME_HISTORY_ROWS}
        />
      </section>
    </div>
  );
}

/** Mean of every verified solve. Null when there are none to average. */
function averageSeconds(rows: { seconds: number; verdict: string }[]): number | null {
  const ok = rows.filter((r) => r.verdict === "verified");
  if (!ok.length) return null;
  return ok.reduce((sum, r) => sum + r.seconds, 0) / ok.length;
}

/**
 * Consecutive days ending today (or yesterday) with at least one solve.
 *
 * Counted from real timestamps rather than "how many rows are there", which
 * is what the mock version did — it reported nine days because there were
 * nine rows in an array.
 */
function dailyStreak(rows: { received_at: string }[]): number {
  if (!rows.length) return 0;
  const days = new Set(rows.map((r) => r.received_at.slice(0, 10)));
  const cursor = new Date();
  // A streak is still alive if you solved yesterday but not yet today.
  if (!days.has(cursor.toISOString().slice(0, 10))) {
    cursor.setDate(cursor.getDate() - 1);
    if (!days.has(cursor.toISOString().slice(0, 10))) return 0;
  }
  let streak = 0;
  while (days.has(cursor.toISOString().slice(0, 10))) {
    streak++;
    cursor.setDate(cursor.getDate() - 1);
  }
  return streak;
}
