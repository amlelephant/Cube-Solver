"use client";

import { useMemo, useState } from "react";
import { Trophy } from "lucide-react";
import { Avatar } from "@/components/Avatar";
import { Card } from "@/components/ui/Card";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import type { WreathTier } from "@/components/LaurelWreath";
import { cn } from "@/lib/cn";
import { leaderboard, you } from "@/lib/mockData";

type Metric = "rating" | "time";

const metricOptions = [
  { value: "rating" as Metric, label: "Elo rating" },
  { value: "time" as Metric, label: "Solve time" },
];

const podiumHeight = ["h-[250px]", "h-[326px]", "h-[250px]"];
// No medal icons: the wreath's metal already says which place this is, and
// showing both said it twice.
const podiumWreath: WreathTier[] = ["silver", "gold", "bronze"];

export default function LeaderboardPage() {
  const [metric, setMetric] = useState<Metric>("rating");

  // Rating and solve time are independent rankings — sorting by each on
  // demand (rather than trusting the static `rank` field) is what makes the
  // toggle show a genuinely different podium instead of just relabeling one.
  const sorted = useMemo(() => {
    return [...leaderboard].sort((a, b) =>
      metric === "rating" ? b.rating - a.rating : parseFloat(a.best) - parseFloat(b.best),
    );
  }, [metric]);

  const podium = [sorted[1], sorted[0], sorted[2]];

  return (
    <div className="mx-auto max-w-6xl px-6 pb-24 pt-16 text-center md:px-10">
      <h1 className="text-5xl font-bold tracking-tight md:text-6xl">Your Champions</h1>

      <div className="mt-8 flex justify-center">
        <SegmentedControl options={metricOptions} value={metric} onChange={setMetric} />
      </div>

      <div className="mx-auto mt-12 flex max-w-3xl items-end justify-center gap-6">
        {podium.map((entry, i) => (
          <div key={entry.name} className="flex flex-1 flex-col items-center gap-3">
            <Card
              className={cn(
                "flex w-full flex-col items-center justify-center p-6",
                podiumHeight[i],
              )}
            >
              <Avatar size={72} wreath={podiumWreath[i]} />
            </Card>
            <p className="font-medium">{entry.name}</p>
            <p className="text-sm text-ink-faint">
              {metric === "rating" ? (
                <>
                  <span className="font-semibold text-ink">{entry.rating}</span> · {entry.best}
                </>
              ) : (
                <>
                  <span className="font-semibold text-ink">{entry.best}</span> · {entry.rating}
                </>
              )}
            </p>
          </div>
        ))}
      </div>

      <div className="mt-16 text-left">
        <div className="flex items-center justify-between rounded-xl bg-cloud px-6 py-4 font-medium">
          <span className="flex items-center gap-3">
            <Trophy size={18} className="text-cube-blue" />
            {metric === "rating" ? you.rating.toLocaleString() : you.best}
            <span className="text-ink-faint">You</span>
          </span>
          <span>{metric === "rating" ? you.best : you.rating.toLocaleString()}</span>
        </div>

        <Card className="mt-4 overflow-hidden p-0">
          {sorted.map((entry, i) => (
            <div
              key={entry.name}
              className={cn(
                "flex items-center justify-between px-6 py-4 text-sm",
                i !== sorted.length - 1 && "border-b border-mist",
              )}
            >
              <span className="flex w-14 shrink-0 items-center gap-2 text-ink-faint">
                #{i + 1}
              </span>
              <span className="flex-1 font-medium">{entry.name}</span>
              <span className="w-20 shrink-0 text-ink-faint">{entry.country}</span>
              {metric === "rating" ? (
                <>
                  <span className="w-20 shrink-0 text-right font-medium">{entry.rating}</span>
                  <span className="w-20 shrink-0 text-right text-ink-faint">{entry.best}</span>
                </>
              ) : (
                <>
                  <span className="w-20 shrink-0 text-right font-medium">{entry.best}</span>
                  <span className="w-20 shrink-0 text-right text-ink-faint">{entry.rating}</span>
                </>
              )}
            </div>
          ))}
        </Card>
      </div>
    </div>
  );
}
