import { User } from "lucide-react";
import { cn } from "@/lib/cn";
import { LaurelWreath, type WreathTier } from "./LaurelWreath";

export function Avatar({
  size = 64,
  wreath,
  className,
}: {
  size?: number;
  wreath?: WreathTier | null;
  className?: string;
}) {
  return (
    <div
      className={cn("relative inline-flex shrink-0 items-center justify-center", className)}
      style={{ width: size, height: size }}
    >
      <div
        className="flex items-center justify-center rounded-full bg-cloud"
        style={{ width: size, height: size }}
      >
        <User size={Math.round(size * 0.42)} className="text-ink-faint" />
      </div>
      {/* after the disc, so the crown sits on the head — the frond tips land
          inside the avatar's circle and have to stay visible there */}
      {wreath && (
        <LaurelWreath
          tier={wreath}
          size={size}
          className="pointer-events-none absolute inset-0"
        />
      )}
    </div>
  );
}
