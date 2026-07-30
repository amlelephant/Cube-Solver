"use client";

import { Avatar } from "@/components/Avatar";
import { Card } from "@/components/ui/Card";
import { Modal } from "@/components/ui/Modal";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { HistoryTable } from "@/components/HistoryTable";
import { CosmeticsPicker } from "@/components/CosmeticsPicker";
import { EloChart } from "@/components/charts/EloChart";
import { SolveHeatmap } from "@/components/charts/SolveHeatmap";
import { useCosmetics } from "@/lib/cosmetics";
import { useState } from "react";
import {
  solveHistory,
  profileMetrics,
  eloHistoryWindows,
  eloWindowOptions,
  solveHeatmap,
  you,
  type EloWindow,
} from "@/lib/mockData";

export default function ProfilePage() {
  const { equippedWreath } = useCosmetics();
  const [eloWindow, setEloWindow] = useState<EloWindow>("24 solves");
  const [cosmeticsOpen, setCosmeticsOpen] = useState(false);

  return (
    <div className="mx-auto max-w-6xl px-6 pb-24 pt-16 md:px-10">
      <div className="grid grid-cols-1 gap-10 md:grid-cols-[1fr_391px]">
        <div>
          <span className="inline-flex items-center rounded-full bg-cloud px-3 py-1 text-xs font-medium tracking-wide text-ink-faint">
            US
          </span>
          <h1 className="mt-3 text-6xl font-bold tracking-tight">Aiden</h1>
          <p className="mt-2 text-2xl text-ink-faint">Global Ranking #{you.rank}</p>

          <div className="mt-8 flex flex-col gap-3">
            <p className="text-sm font-medium text-ink-faint">List of metrics</p>
            {profileMetrics.map((metric) => (
              <div key={metric.label} className="flex items-center justify-between gap-6">
                <span className="text-lg">{metric.label}</span>
                <span className="rounded-lg bg-cloud px-4 py-2 text-sm font-semibold">
                  {metric.value}
                </span>
              </div>
            ))}
          </div>
        </div>

        <Card className="flex flex-col items-center justify-center gap-4 p-8">
          <button
            onClick={() => setCosmeticsOpen(true)}
            aria-label="Change avatar cosmetics"
            className="rounded-full transition-opacity hover:opacity-80"
          >
            <Avatar size={128} wreath={equippedWreath} />
          </button>
          <p className="text-lg font-medium text-ink-faint">Player Avatar</p>
          <button
            onClick={() => setCosmeticsOpen(true)}
            className="text-xs font-medium text-ink-faint underline-offset-2 hover:text-ink hover:underline"
          >
            Change cosmetics
          </button>
        </Card>
      </div>

      <Modal open={cosmeticsOpen} onClose={() => setCosmeticsOpen(false)} title="Avatar cosmetics">
        <CosmeticsPicker />
      </Modal>

      <div className="mt-10 grid grid-cols-1 gap-6 md:grid-cols-2">
        <Card className="p-8">
          <p className="text-xl font-medium">Solve heat map</p>
          <p className="mt-1 text-sm text-ink-faint">Last 6 months</p>
          <div className="mt-6">
            <SolveHeatmap weeks={solveHeatmap} />
          </div>
        </Card>

        <Card className="p-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xl font-medium">ELO chart</p>
              <p className="mt-1 text-sm text-ink-faint">Rating over {eloWindow}</p>
            </div>
            <SegmentedControl
              options={eloWindowOptions.map((w) => ({ value: w, label: w }))}
              value={eloWindow}
              onChange={setEloWindow}
            />
          </div>
          <div className="mt-4">
            <EloChart points={eloHistoryWindows[eloWindow]} />
          </div>
        </Card>
      </div>

      <div className="mt-10">
        <p className="mb-4 text-2xl text-ink-faint">History</p>
        <HistoryTable bestTime="11.75s" rows={solveHistory} />
      </div>
    </div>
  );
}
