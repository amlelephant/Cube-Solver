import { User } from "lucide-react";
import { cn } from "@/lib/cn";
import { AVATAR_BY_KEY, PresetAvatar } from "@/lib/avatars";
import { LaurelWreath, type WreathTier } from "./LaurelWreath";

/**
 * A player's avatar: their chosen preset, or a neutral silhouette.
 *
 * `preset` is a key the SERVER validated (`core/avatars.py`) and handed back
 * — never a URL and never anything the account authored, which is what lets
 * this render on someone else's profile with no moderation behind it.
 * Unknown keys fall through to the silhouette rather than rendering blank,
 * so a preset retired server-side degrades instead of leaving a hole.
 */
export function Avatar({
  size = 64,
  wreath,
  preset,
  className,
}: {
  size?: number;
  wreath?: WreathTier | null;
  preset?: string | null;
  className?: string;
}) {
  const known = preset ? AVATAR_BY_KEY[preset] : undefined;
  return (
    <div
      className={cn("relative inline-flex shrink-0 items-center justify-center", className)}
      style={{ width: size, height: size }}
    >
      <div
        className="flex items-center justify-center overflow-hidden rounded-full bg-cloud"
        style={{ width: size, height: size }}
      >
        {known ? (
          <PresetAvatar preset={known.key} size={size} />
        ) : (
          <User size={Math.round(size * 0.42)} className="text-ink-faint" />
        )}
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
