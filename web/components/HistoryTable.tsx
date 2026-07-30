import Link from "next/link";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/cn";
import type { SolveRecord } from "@/lib/mockData";

const resultColor: Record<SolveRecord["result"], string> = {
  win: "text-cube-green",
  loss: "text-cube-red",
  solo: "text-ink-faint",
};

export function HistoryTable({
  title = "Current Best",
  bestTime,
  rows,
}: {
  title?: string;
  bestTime: string;
  rows: SolveRecord[];
}) {
  return (
    <Card className="overflow-hidden p-0">
      <div className="flex items-center justify-between bg-cloud px-6 py-4">
        <p className="text-lg font-medium">{title}</p>
        <p className="text-lg font-semibold cube-gradient-text">{bestTime}</p>
      </div>
      <div>
        {rows.map((row, i) => (
          <Link
            key={row.date + i}
            href={`/solve/${i}`}
            className={cn(
              "flex items-center justify-between px-6 py-4 text-sm transition-colors hover:bg-cloud",
              i !== rows.length - 1 && "border-b border-mist",
            )}
          >
            <span className="w-36 shrink-0 text-ink-faint">{row.date}</span>
            <span className="flex-1 text-center font-medium">{row.opponent}</span>
            <span className={cn("w-20 shrink-0 text-right font-medium", resultColor[row.result])}>
              {row.time}
            </span>
          </Link>
        ))}
      </div>
    </Card>
  );
}
