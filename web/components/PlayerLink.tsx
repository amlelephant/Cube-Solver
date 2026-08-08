"use client";

import Link from "next/link";
import { Flag } from "@/components/Flag";
import { cn } from "@/lib/cn";

/**
 * A player's name, anywhere it appears, always going to their profile.
 *
 * One component rather than an ad-hoc <Link> per surface, because "names are
 * clickable wherever they are encountered" only stays true if there is a
 * single thing to reach for. The leaderboard, the podium, a solve's opponent
 * and the history table all use this.
 *
 * `profileHref` is exported separately for the cases that need the URL but
 * not the markup (a whole table row that is itself a link, for instance).
 */
export function profileHref(username: string): string {
  return `/u/${encodeURIComponent(username)}`;
}

export function PlayerLink({
  username,
  country,
  showFlag = false,
  className,
}: {
  username: string;
  country?: string | null;
  showFlag?: boolean;
  className?: string;
}) {
  return (
    <Link
      href={profileHref(username)}
      className={cn(
        "inline-flex items-center gap-1.5 rounded transition-colors",
        "hover:text-ink hover:underline underline-offset-2",
        className,
      )}
    >
      {showFlag && <Flag code={country} decorative />}
      {username}
    </Link>
  );
}
