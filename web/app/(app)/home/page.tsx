"use client";

import Link from "next/link";
import { ArrowUpRight, Swords, Timer } from "lucide-react";
import { Avatar } from "@/components/Avatar";
import { Card } from "@/components/ui/Card";
import { HistoryTable } from "@/components/HistoryTable";
import { useCosmetics } from "@/lib/cosmetics";
import { solveHistory, homeStats } from "@/lib/mockData";

const USERNAME = "Aiden";

export default function HomePage() {
  const { equippedWreath } = useCosmetics();

  return (
    <div className="mx-auto max-w-6xl px-6 pb-24 md:px-10">
      <section className="cube-gradient relative -mx-6 mt-6 flex h-72 items-center justify-center overflow-hidden rounded-3xl text-center md:-mx-10 md:h-96">
        <div className="absolute inset-0 bg-black/45" />
        <div className="relative px-6">
          <h1 className="text-4xl font-bold tracking-tight text-white md:text-6xl">
            Welcome home, {USERNAME}
          </h1>
          <p className="mx-auto mt-3 max-w-lg text-balance text-white/80">
            Your last verified solve was 15.61s. Ready to beat it?
          </p>
        </div>
      </section>

      <section className="mt-8 grid grid-cols-1 gap-6 md:grid-cols-[225px_1fr]">
        <Card className="flex flex-col items-center justify-center gap-3 p-8">
          <Avatar size={64} wreath={equippedWreath} />
          <p className="text-sm font-medium text-ink-faint">Your avatar</p>
        </Card>

        <Card className="flex flex-col gap-6 p-8">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <p className="text-2xl font-medium">Analytics</p>
            <Link
              href="/profile"
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
              <p className="mt-1 text-2xl font-semibold">{homeStats.averageSolve}</p>
            </div>
            <div>
              <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">
                Daily streak
              </p>
              <p className="mt-1 text-2xl font-semibold">{homeStats.dailyStreak} days</p>
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

      <section className="mt-8">
        <HistoryTable bestTime="11.75s" rows={solveHistory} />
      </section>
    </div>
  );
}
