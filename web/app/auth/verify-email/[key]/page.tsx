"use client";

import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { useAuth } from "@/lib/auth";
import { ApiError } from "@/lib/api";
import { AuthCard, AuthPage } from "@/components/auth/AuthCard";

/**
 * The landing page for the link in a confirmation email. The path shape is
 * fixed by settings.py:
 *
 *     HEADLESS_FRONTEND_URLS["account_confirm_email"] = "/auth/verify-email/{key}"
 *
 * so this folder name is load-bearing — rename it and every verification mail
 * already in someone's inbox breaks.
 *
 * Verification is confirmed with a POST, never on render alone. Mail clients
 * and corporate scanners prefetch links, and a GET that consumed the key
 * would burn it before the human ever clicked — the classic symptom being
 * "the link was already used" on the first attempt. So we confirm on mount
 * via an explicit request, and if anything goes wrong we hand back a button
 * rather than a dead end.
 */

type State =
  | { s: "working" }
  | { s: "done" }
  | { s: "expired" }
  | { s: "error"; message: string };

export default function VerifyEmailPage({
  params,
}: {
  params: Promise<{ key: string }>;
}) {
  const { key } = use(params);
  const { verifyEmail, user } = useAuth();
  const router = useRouter();
  const [state, setState] = useState<State>({ s: "working" });

  // StrictMode double-invokes effects in dev. The second call would hit
  // allauth with an already-spent key and report "expired" over a
  // verification that in fact just succeeded.
  const fired = useRef(false);

  useEffect(() => {
    if (fired.current) return;
    fired.current = true;

    (async () => {
      try {
        await verifyEmail(decodeURIComponent(key));
        setState({ s: "done" });
      } catch (e) {
        if (e instanceof ApiError && (e.status === 400 || e.status === 409)) {
          setState({ s: "expired" });
        } else if (e instanceof ApiError && e.status === 401) {
          // Verified, but allauth wants another step (or simply did not open a
          // session). The account is usable — sign in normally.
          setState({ s: "done" });
        } else {
          setState({
            s: "error",
            message: (e as Error)?.message ?? "Something went wrong.",
          });
        }
      }
    })();
  }, [key, verifyEmail]);

  if (state.s === "working") {
    return (
      <AuthPage>
        <AuthCard title="Confirming your email" subtitle="One moment…">
          <div className="flex justify-center py-2">
            <span
              className="size-5 animate-pulse rounded-[4px] cube-gradient"
              aria-hidden
            />
          </div>
        </AuthCard>
      </AuthPage>
    );
  }

  if (state.s === "done") {
    // Whether confirming also SIGNS YOU IN depends on how allauth is
    // configured and on whether the link was opened in the same browser that
    // started the signup. Read the session rather than assuming: offering
    // "go to your dashboard" to someone with no session just bounces them off
    // RequireAuth and back to a login page they were not expecting.
    const signedIn = !!user;
    return (
      <AuthPage>
        <AuthCard
          title="You're verified"
          subtitle={
            signedIn
              ? "Your account is ready. Welcome to CubeArena."
              : "Your account is ready. Sign in and you're off."
          }
        >
          <Button
            className="w-full"
            onClick={() => router.replace(signedIn ? "/home" : "/auth/login")}
          >
            {signedIn ? "Go to your dashboard" : "Sign in"}
          </Button>
        </AuthCard>
      </AuthPage>
    );
  }

  if (state.s === "expired") {
    return (
      <AuthPage>
        <AuthCard
          title="This link has expired"
          subtitle="Confirmation links are single-use and time-limited. Sign in and we'll send a fresh one."
        >
          <Button className="w-full" onClick={() => router.replace("/auth/login")}>
            Back to sign in
          </Button>
        </AuthCard>
      </AuthPage>
    );
  }

  return (
    <AuthPage>
      <AuthCard title="We couldn't confirm that" subtitle={state.message}>
        <p className="text-center text-xs text-ink-faint">
          <Link
            href="/auth/login"
            className="underline underline-offset-2 hover:text-ink"
          >
            Back to sign in
          </Link>
        </p>
      </AuthCard>
    </AuthPage>
  );
}
