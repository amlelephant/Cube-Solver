import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/cn";
import { solveHistory, getSolveDetail } from "@/lib/mockData";

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

export default async function SolveDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const index = Number(id);
  const record = solveHistory[index];
  const detail = getSolveDetail(index);

  if (!record || !detail) notFound();

  return (
    <div className="mx-auto max-w-3xl px-6 pb-24 pt-16 md:px-10">
      <Link
        href="/profile"
        className="inline-flex items-center gap-1 text-sm font-medium text-ink-faint transition-colors hover:text-ink"
      >
        <ArrowLeft size={16} /> Back to profile
      </Link>

      <div className="mt-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-ink-faint">{record.date}</p>
          <p className="mt-1 text-2xl font-medium">{record.opponent}</p>
        </div>
        <span className={cn("rounded-full bg-cloud px-3 py-1 text-sm font-medium", resultColor[record.result])}>
          {resultLabel[record.result]}
        </span>
      </div>

      <p className="mt-6 text-7xl font-semibold tracking-tight cube-gradient-text">{record.time}</p>

      {detail.ratingDelta !== null && (
        <p className="mt-2 text-sm font-medium text-ink-faint">
          Rating {detail.ratingDelta > 0 ? "+" : ""}
          {detail.ratingDelta}
        </p>
      )}

      <div className="mt-10 grid grid-cols-1 gap-6 md:grid-cols-[1fr_260px]">
        <Card className="p-8">
          <p className="text-lg font-medium">Phase breakdown</p>
          <div className="mt-6 flex flex-col gap-4">
            {detail.phases.map((phase) => (
              <div key={phase.label}>
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">{phase.label}</span>
                  <span className="text-ink-faint">{phase.seconds.toFixed(2)}s</span>
                </div>
                <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-cloud">
                  <div
                    className="h-full rounded-full bg-cube-blue"
                    style={{ width: `${phase.pct * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>

        <div className="flex flex-col gap-6">
          <Card className="p-6">
            <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">Move count</p>
            <p className="mt-1 text-2xl font-semibold">{detail.moveCount}</p>
          </Card>
          <Card className="p-6">
            <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">
              Turns per second
            </p>
            <p className="mt-1 text-2xl font-semibold">{detail.tps} TPS</p>
          </Card>
        </div>
      </div>

      <Card className="mt-6 p-8">
        <p className="text-lg font-medium">Scramble</p>
        <p className="mt-3 break-words font-mono text-sm leading-relaxed text-ink-soft">
          {detail.scramble}
        </p>
      </Card>
    </div>
  );
}
