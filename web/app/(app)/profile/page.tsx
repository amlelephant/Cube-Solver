"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { profileHref } from "@/components/PlayerLink";

/**
 * `/profile` is a shortcut to your own profile at `/u/<username>`.
 *
 * A redirect rather than a second implementation: your profile and someone
 * else's differ only in whether the cosmetics editor is offered, and two
 * copies of a page that renders the same data is how the two quietly stop
 * agreeing. `/u/[username]` handles both and decides by comparing the
 * signed-in user to the profile it loaded.
 */
export default function MyProfileRedirect() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (loading) return;
    // RequireAuth already guarantees a session here; the fallback only
    // matters for the beat between logout and its redirect firing.
    router.replace(user?.username ? profileHref(user.username) : "/auth/login");
  }, [loading, user, router]);

  return (
    <div className="flex min-h-[50vh] items-center justify-center">
      <span className="sr-only">Opening your profile…</span>
      <span className="size-5 animate-pulse rounded-[4px] cube-gradient" aria-hidden />
    </div>
  );
}
