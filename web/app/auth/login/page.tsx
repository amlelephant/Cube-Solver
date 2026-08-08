"use client";

import { useEffect, useState, Suspense } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { PasswordInput } from "@/components/ui/PasswordInput";
import { useAuth } from "@/lib/auth";
import {
  AuthCard,
  AuthError,
  AuthPage,
  authInputClass,
} from "@/components/auth/AuthCard";

/**
 * Sign in / create account.
 *
 * Lives OUTSIDE the `(app)` route group on purpose: that group is wrapped in
 * `RequireAuth`, so putting the login page inside it would redirect an
 * unauthenticated visitor to the login page from the login page, forever.
 *
 * Sign-in and sign-up are one component with a `mode` toggle rather than two
 * pages, because they are the same two fields and the same request shape and
 * people switch between them constantly. `/auth/signup` is a real route that
 * simply lands here with `?mode=signup` — allauth's own emails link to it
 * (HEADLESS_FRONTEND_URLS in settings.py), so it has to resolve.
 */

type Mode = "login" | "signup";

function AuthForm() {
  const router = useRouter();
  const params = useSearchParams();
  const { user, loading, login, signup, resendVerification } = useAuth();

  const [mode, setMode] = useState<Mode>(
    params.get("mode") === "signup" ? "signup" : "login",
  );
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sentTo, setSentTo] = useState<string | null>(null);
  const [resent, setResent] = useState<string | null>(null);

  // Where to land after signing in. Defaults to /home, but RequireAuth passes
  // the page they were originally trying to reach so a deep link survives the
  // detour through here.
  //
  // Only ever a same-site PATH. `next` arrives from the query string, so
  // without this check `/auth/login?next=https://evil.example` sends someone
  // off-site the instant they authenticate — an open redirect, and a
  // convincing one precisely because the visitor really did just sign in to
  // the real site. `//host` is rejected too: it is protocol-relative and
  // browsers read it as another origin.
  const rawNext = params.get("next");
  const next =
    rawNext && rawNext.startsWith("/") && !rawNext.startsWith("//")
      ? rawNext
      : "/home";

  useEffect(() => {
    if (!loading && user) router.replace(next);
  }, [loading, user, next, router]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "login") {
        await login(email.trim(), password);
        router.replace(next);
      } else {
        const { verificationRequired } = await signup(email.trim(), password);
        if (verificationRequired) setSentTo(email.trim());
        else router.replace(next);
      }
    } catch (err: any) {
      setError(err?.message ?? "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  if (sentTo) {
    return (
      <AuthCard
        title="Check your email"
        subtitle={
          <>
            We sent a confirmation link to{" "}
            <span className="text-ink">{sentTo}</span>. Open it to finish
            setting up your account.
          </>
        }
      >
        <p className="text-center text-xs text-ink-faint">
          Nothing arrived? Check spam,{" "}
          <button
            className="underline underline-offset-2 hover:text-ink"
            onClick={async () => {
              setResent(null);
              try {
                await resendVerification(sentTo);
                setResent("Sent again — give it a minute.");
              } catch {
                setResent("Could not resend just now. Try again shortly.");
              }
            }}
          >
            send it again
          </button>
          , or{" "}
          <button
            className="underline underline-offset-2 hover:text-ink"
            onClick={() => {
              setSentTo(null);
              setMode("login");
            }}
          >
            try signing in
          </button>
          .
        </p>
        {resent && (
          <p className="mt-3 text-center text-xs text-ink-soft" role="status">
            {resent}
          </p>
        )}
      </AuthCard>
    );
  }

  return (
    <AuthCard
      title={mode === "login" ? "Sign in to CubeArena" : "Create your account"}
      subtitle={
        mode === "login"
          ? "Verified solves, weekly competitions, and your own history."
          : "Free forever. Verification is never paywalled."
      }
      footer={
        <p className="mt-6 text-center text-sm text-ink-soft">
          {mode === "login" ? "No account yet?" : "Already have one?"}{" "}
          <button
            type="button"
            className="font-medium text-ink underline underline-offset-2"
            onClick={() => {
              setMode(mode === "login" ? "signup" : "login");
              setError(null);
            }}
          >
            {mode === "login" ? "Create one" : "Sign in"}
          </button>
        </p>
      }
    >
      <form onSubmit={onSubmit} className="space-y-4" noValidate>
        <div>
          <label htmlFor="email" className="mb-1.5 block text-sm font-medium">
            Email
          </label>
          <input
            id="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            className={authInputClass}
          />
        </div>

        <div>
          <div className="mb-1.5 flex items-baseline justify-between">
            <label htmlFor="password" className="block text-sm font-medium">
              Password
            </label>
            {mode === "login" && (
              <Link
                href="/auth/reset-password"
                className="text-xs text-ink-faint underline underline-offset-2 hover:text-ink"
              >
                Forgot?
              </Link>
            )}
          </div>
          <PasswordInput
            id="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder={mode === "signup" ? "At least 8 characters" : "••••••••"}
          />
        </div>

        <AuthError>{error}</AuthError>

        <Button type="submit" disabled={busy} className="w-full">
          {busy
            ? "One moment…"
            : mode === "login"
              ? "Sign in"
              : "Create account"}
        </Button>
      </form>
    </AuthCard>
  );
}

export default function LoginPage() {
  return (
    <AuthPage>
      {/* useSearchParams needs a Suspense boundary to prerender. */}
      <Suspense fallback={<Card className="w-full max-w-md p-8" />}>
        <AuthForm />
      </Suspense>
    </AuthPage>
  );
}
