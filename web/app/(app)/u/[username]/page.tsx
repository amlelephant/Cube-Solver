"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { Avatar } from "@/components/Avatar";
import { Card } from "@/components/ui/Card";
import { Modal } from "@/components/ui/Modal";
import { CosmeticsPicker } from "@/components/CosmeticsPicker";
import { RecentSolves } from "@/components/RecentSolves";
import { SolveHeatmap } from "@/components/charts/SolveHeatmap";
import { RatingCard } from "@/components/charts/RatingCard";
import { useCosmetics } from "@/lib/cosmetics";
import { useAuth } from "@/lib/auth";
import { ApiError } from "@/lib/api";
import { getProfile, type PublicProfile, type SolveRow } from "@/lib/account";
import { countryName } from "@/lib/countries";
import { Flag } from "@/components/Flag";
import type { WreathTier } from "@/components/LaurelWreath";

/**
 * Anyone's profile. EVERY profile is public, so this route has no guard of
 * its own beyond the `(app)` group's — there is nothing here that depends on
 * who is looking, except the cosmetics editor, which only appears on your
 * own.
 *
 * Everything on the page comes from `/api/users/<username>/`: the metrics are
 * aggregates over that account's Solve rows, and the history is those rows.
 * Nothing is generated in the browser.
 */

function wreathFor(p: PublicProfile): WreathTier | null {
  if (p.is_founder) return "founder";
  if (p.best_rank === 1) return "gold";
  if (p.best_rank === 2) return "silver";
  if (p.best_rank === 3) return "bronze";
  return null;
}

function fmt(seconds: number | null): string {
  return seconds == null ? "—" : `${seconds.toFixed(2)}s`;
}

/** Mean of the best N consecutive solves, the way cubers actually quote it. */
function averageOf(rows: SolveRow[], n: number): number | null {
  const verified = rows.filter((r) => r.verdict === "verified");
  if (verified.length < n) return null;
  let best = Infinity;
  for (let i = 0; i + n <= verified.length; i++) {
    const window = verified.slice(i, i + n);
    best = Math.min(best, window.reduce((s, r) => s + r.seconds, 0) / n);
  }
  return best === Infinity ? null : best;
}

/** 26 weeks x 7 days of solve counts, bucketed 0-4, from real timestamps. */
function heatmapFrom(rows: SolveRow[]): number[][] {
  const counts = new Map<string, number>();
  for (const r of rows) {
    const key = r.received_at.slice(0, 10);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const weeks: number[][] = [];
  const today = new Date();
  for (let w = 25; w >= 0; w--) {
    const week: number[] = [];
    for (let d = 6; d >= 0; d--) {
      const day = new Date(today);
      day.setDate(today.getDate() - (w * 7 + d));
      const n = counts.get(day.toISOString().slice(0, 10)) ?? 0;
      week.push(Math.min(4, n));
    }
    weeks.push(week);
  }
  return weeks;
}

export default function ProfilePage({
  params,
}: {
  params: Promise<{ username: string }>;
}) {
  const { username } = use(params);
  const { user } = useAuth();
  const { equippedWreath } = useCosmetics();

  const [profile, setProfile] = useState<PublicProfile | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "missing" | "error">("loading");
  const [cosmeticsOpen, setCosmeticsOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    getProfile(username)
      .then((p) => {
        if (cancelled) return;
        setProfile(p);
        setState("ready");
      })
      .catch((e) => {
        if (cancelled) return;
        setState(e instanceof ApiError && e.status === 404 ? "missing" : "error");
      });
    return () => {
      cancelled = true;
    };
  }, [username]);

  if (state === "loading") {
    return (
      <div className="mx-auto max-w-6xl px-6 pb-24 pt-16 md:px-10">
        <div className="h-40 animate-pulse rounded-2xl bg-cloud" />
      </div>
    );
  }

  if (state === "missing" || !profile) {
    return (
      <div className="mx-auto max-w-2xl px-6 pb-24 pt-24 text-center md:px-10">
        <h1 className="text-4xl font-bold tracking-tight">No such player</h1>
        <p className="mt-3 text-ink-faint">
          There's no CubeArena account called “{username}”.
        </p>
        <Link
          href="/leaderboard"
          className="mt-6 inline-block text-sm underline underline-offset-2 hover:text-ink"
        >
          Back to the leaderboard
        </Link>
      </div>
    );
  }

  const isYou = user?.username === profile.username;
  const rows = profile.solves ?? [];
  const wreath = isYou ? equippedWreath : wreathFor(profile);

  const metrics = [
    { label: "3x3 best solve", value: fmt(profile.best_seconds) },
    { label: "3x3 best ao5", value: fmt(averageOf(rows, 5)) },
    { label: "3x3 best ao12", value: fmt(averageOf(rows, 12)) },
    { label: "Total verified solves", value: profile.total_solves.toLocaleString() },
    { label: "Elo rating", value: profile.rating.toLocaleString() },
  ];

  return (
    <div className="mx-auto max-w-6xl px-6 pb-24 pt-16 md:px-10">
      <div className="grid grid-cols-1 gap-10 md:grid-cols-[1fr_391px]">
        <div>
          {profile.country && (
            <span className="inline-flex items-center gap-2 rounded-full bg-cloud px-3 py-1 text-xs font-medium tracking-wide text-ink-faint">
              <Flag code={profile.country} decorative className="h-3 w-4" />
              {countryName(profile.country) ?? profile.country}
            </span>
          )}
          <h1 className="mt-3 text-6xl font-bold tracking-tight">{profile.username}</h1>
          <p className="mt-2 text-2xl text-ink-faint">
            Global Ranking #{profile.rank ?? "—"}
            {isYou && <span className="ml-3 text-base">(that's you)</span>}
          </p>

          <div className="mt-8 flex flex-col gap-3">
            <p className="text-sm font-medium text-ink-faint">List of metrics</p>
            {metrics.map((m) => (
              <div key={m.label} className="flex items-center justify-between gap-6">
                <span className="text-lg">{m.label}</span>
                <span className="rounded-lg bg-cloud px-4 py-2 text-sm font-semibold">
                  {m.value}
                </span>
              </div>
            ))}
          </div>
        </div>

        <Card className="flex flex-col items-center justify-center gap-4 p-8">
          {isYou ? (
            <>
              <button
                onClick={() => setCosmeticsOpen(true)}
                aria-label="Change avatar and cosmetics"
                className="rounded-full transition-opacity hover:opacity-80"
              >
                <Avatar size={128} wreath={wreath} preset={profile.avatar} />
              </button>
              <p className="text-lg font-medium text-ink-faint">Player Avatar</p>
              <button
                onClick={() => setCosmeticsOpen(true)}
                className="text-xs font-medium text-ink-faint underline-offset-2 hover:text-ink hover:underline"
              >
                Change cosmetics
              </button>
            </>
          ) : (
            <>
              <Avatar size={128} wreath={wreath} preset={profile.avatar} />
              <p className="text-lg font-medium text-ink-faint">Player Avatar</p>
              <p className="text-xs text-ink-faint">
                Member #{profile.id} · joined{" "}
                {new Date(profile.joined).toLocaleDateString(undefined, {
                  month: "short",
                  year: "numeric",
                })}
              </p>
            </>
          )}
        </Card>
      </div>

      {isYou && (
        <Modal
          open={cosmeticsOpen}
          onClose={() => {
            setCosmeticsOpen(false);
            // The picker writes the avatar through `/api/me/`, which is a
            // different fetch from this page's `/api/users/<name>/`. Without
            // this the card behind the modal keeps the old avatar until a
            // reload, which reads as the save having failed.
            void getProfile(username).then(setProfile).catch(() => {});
          }}
          title="Avatar & cosmetics"
        >
          <CosmeticsPicker />
        </Modal>
      )}

      {/* Heat map and rating side by side: one is how OFTEN you play, the
          other is how it is going. They answer the same question from two
          directions and reading them together is the point — stacked, the
          rating card fell below the fold and nobody saw the reason to
          compete. Single column below lg, where 26 weeks of heat map has no
          room to share. */}
      <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="flex h-full flex-col p-8">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <p className="text-xl font-medium">Solve heat map</p>
            <p className="text-sm text-ink-faint">Last 6 months</p>
          </div>
          {/* Centred vertically in the leftover space — the row gives both
              cards the rating card's height and the heat map is shorter, so
              top-aligning dumped ~150px of void under it.

              flex-COL, not flex-row. A row-direction container makes the
              heat map a flex item sized by its content, and its columns are
              `flex-1 min-w-0` with no intrinsic width, so the whole grid
              collapsed to 152px with 2px cells. In a column container the
              cross axis still stretches, so it keeps its full width. */}
          <div className="mt-6 flex flex-1 flex-col justify-center">
            <SolveHeatmap weeks={heatmapFrom(rows)} />
          </div>
        </Card>

        <RatingCard rows={rows} rating={profile.rating} isYou={isYou} />
      </div>

      <div className="mt-10">
        <p className="mb-4 text-2xl text-ink-faint">History</p>
        <RecentSolves
          bestSeconds={profile.best_seconds}
          bestSolveId={profile.best_solve_id}
          rows={rows}
          linkRows={isYou}
          emptyLabel="No solves recorded yet."
        />
      </div>
    </div>
  );
}
