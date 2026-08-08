"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";

/**
 * Gate for every page in the `(app)` group. The landing page (`/`) and the
 * auth screens sit outside it and stay public.
 *
 * THIS IS A UX GUARD, NOT A SECURITY BOUNDARY. It runs in the browser, and
 * anyone can bypass it with devtools. That is fine, because it protects
 * nothing on its own: the pages behind it are shells, and every piece of real
 * data comes from the API, which enforces permissions server-side. The rule
 * is the same one the anticheat follows — client checks are UX, server checks
 * are authority.
 *
 * The redirect carries `?next=` so a deep link survives the detour through
 * the login page instead of dumping everyone on /home.
 */
export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const [slow, setSlow] = useState(false);

  const target = `/auth/login?next=${encodeURIComponent(pathname || "/home")}`;

  useEffect(() => {
    if (loading || user) return;
    router.replace(target);
  }, [loading, user, target, router]);

  // A redirect is not instant — in dev the login route compiles on demand the
  // first time, which can take seconds. Say so rather than sitting on a mute
  // spinner, and offer a real link, so a slow or failed client-side
  // navigation degrades into something the visitor can act on instead of a
  // page that looks hung.
  useEffect(() => {
    if (loading || user) {
      setSlow(false);
      return;
    }
    const id = setTimeout(() => setSlow(true), 600);
    return () => clearTimeout(id);
  }, [loading, user]);

  // Render nothing decisive until the session check resolves. Flashing the
  // app and then yanking it away is worse than a beat of blank space, and
  // flashing the login page at someone who IS signed in is worse still.
  if (loading || !user) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4">
        <span
          className="size-5 animate-pulse rounded-[4px] cube-gradient"
          aria-hidden
        />
        {loading ? (
          <span className="sr-only">Checking your session…</span>
        ) : (
          <p
            className={`text-sm text-ink-soft transition-opacity duration-300 ${
              slow ? "opacity-100" : "opacity-0"
            }`}
            role="status"
          >
            Taking you to sign in…{" "}
            <Link href={target} className="underline underline-offset-2 hover:text-ink">
              Go now
            </Link>
          </p>
        )}
      </div>
    );
  }

  return <>{children}</>;
}
