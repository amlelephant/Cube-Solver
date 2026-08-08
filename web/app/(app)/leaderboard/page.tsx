"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Trophy } from "lucide-react";
import { Avatar } from "@/components/Avatar";
import { Card } from "@/components/ui/Card";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { PlayerLink, profileHref } from "@/components/PlayerLink";
import type { WreathTier } from "@/components/LaurelWreath";
import { cn } from "@/lib/cn";
import { Flag } from "@/components/Flag";
import { getLeaderboard, useMe, type PublicProfile } from "@/lib/account";

type Metric = "rating" | "time";

const metricOptions = [
  { value: "rating" as Metric, label: "Elo rating" },
  { value: "time" as Metric, label: "Solve time" },
];

const podiumHeight = ["h-[250px]", "h-[326px]", "h-[250px]"];
// No medal icons: the wreath's metal already says which place this is, and
// showing both said it twice.
const podiumWreath: WreathTier[] = ["silver", "gold", "bronze"];

const fmt = (s: number | null) => (s == null ? "—" : `${s.toFixed(2)}s`);

export default function LeaderboardPage() {
  const [metric, setMetric] = useState<Metric>("rating");
  const [rows, setRows] = useState<PublicProfile[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { me } = useMe();

  useEffect(() => {
    getLeaderboard(25)
      .then((r) => setRows(r.results))
      .catch((e) => setError(e?.message ?? "Could not load the leaderboard."));
  }, []);

  // Rating and solve time are independent rankings — sorting by each on
  // demand (rather than trusting the server's rating order) is what makes the
  // toggle show a genuinely different podium instead of relabeling one.
  // Players with no verified solve sort last under "time" rather than first.
  const sorted = useMemo(() => {
    if (!rows) return [];
    return [...rows].sort((a, b) =>
      metric === "rating"
        ? b.rating - a.rating
        : (a.best_seconds ?? Infinity) - (b.best_seconds ?? Infinity),
    );
  }, [rows, metric]);

  if (error) {
    return (
      <div className="mx-auto max-w-2xl px-6 pt-24 text-center md:px-10">
        <h1 className="text-3xl font-bold tracking-tight">Leaderboard unavailable</h1>
        <p className="mt-3 text-ink-faint">{error}</p>
      </div>
    );
  }

  if (!rows) {
    return (
      <div className="mx-auto max-w-6xl px-6 pb-24 pt-16 md:px-10">
        <div className="h-[420px] animate-pulse rounded-2xl bg-cloud" />
      </div>
    );
  }

  const podium = [sorted[1], sorted[0], sorted[2]].filter(Boolean);

  return (
    <div className="mx-auto max-w-6xl px-6 pb-24 pt-16 text-center md:px-10">
      <h1 className="text-5xl font-bold tracking-tight md:text-6xl">Your Champions</h1>

      <div className="mt-8 flex justify-center">
        <SegmentedControl options={metricOptions} value={metric} onChange={setMetric} />
      </div>

      <div className="mx-auto mt-12 flex max-w-3xl items-end justify-center gap-6">
        {podium.map((entry, i) => (
          <div key={entry.username} className="flex flex-1 flex-col items-center gap-3">
            <Link
              href={profileHref(entry.username)}
              aria-label={`${entry.username}'s profile`}
              className="w-full transition-transform hover:-translate-y-1"
            >
              <Card
                className={cn(
                  "flex w-full flex-col items-center justify-center p-6",
                  podiumHeight[i],
                )}
              >
                <Avatar size={72} wreath={podiumWreath[i]} preset={entry.avatar} />
              </Card>
            </Link>
            <PlayerLink
              username={entry.username}
              country={entry.country}
              showFlag
              className="font-medium"
            />
            <p className="text-sm text-ink-faint">
              {metric === "rating" ? (
                <>
                  <span className="font-semibold text-ink">{entry.rating}</span> ·{" "}
                  {fmt(entry.best_seconds)}
                </>
              ) : (
                <>
                  <span className="font-semibold text-ink">{fmt(entry.best_seconds)}</span>{" "}
                  · {entry.rating}
                </>
              )}
            </p>
          </div>
        ))}
      </div>

      <div className="mt-16 text-left">
        {me && (
          <Link
            href={profileHref(me.username)}
            className="flex items-center justify-between rounded-xl bg-cloud px-6 py-4 font-medium transition-colors hover:bg-mist"
          >
            <span className="flex items-center gap-3">
              <Trophy size={18} className="text-cube-blue" />
              {metric === "rating" ? me.rating.toLocaleString() : fmt(me.best_seconds)}
              <span className="text-ink-faint">You · #{me.rank}</span>
            </span>
            <span>
              {metric === "rating" ? fmt(me.best_seconds) : me.rating.toLocaleString()}
            </span>
          </Link>
        )}

        <Card className="mt-4 overflow-hidden p-0">
          {sorted.map((entry, i) => (
            <Link
              key={entry.username}
              href={profileHref(entry.username)}
              className={cn(
                "flex items-center justify-between px-6 py-4 text-sm transition-colors hover:bg-cloud",
                i !== sorted.length - 1 && "border-b border-mist",
                me?.id === entry.id && "bg-cloud/60",
              )}
            >
              <span className="flex w-14 shrink-0 items-center gap-2 text-ink-faint">
                #{i + 1}
              </span>
              {/* Flag beside the name, not in its own column. A separate
                  country column cost 80px that narrow screens do not have,
                  and read as a data field rather than as part of who the
                  player is. */}
              <span className="flex min-w-0 flex-1 items-center gap-2 font-medium">
                <span className="flex w-5 shrink-0 justify-center">
                  <Flag code={entry.country} />
                </span>
                <span className="truncate">{entry.username}</span>
              </span>
              {metric === "rating" ? (
                <>
                  <span className="w-20 shrink-0 text-right font-medium">{entry.rating}</span>
                  <span className="w-20 shrink-0 text-right text-ink-faint">
                    {fmt(entry.best_seconds)}
                  </span>
                </>
              ) : (
                <>
                  <span className="w-20 shrink-0 text-right font-medium">
                    {fmt(entry.best_seconds)}
                  </span>
                  <span className="w-20 shrink-0 text-right text-ink-faint">
                    {entry.rating}
                  </span>
                </>
              )}
            </Link>
          ))}
        </Card>
      </div>
    </div>
  );
}
