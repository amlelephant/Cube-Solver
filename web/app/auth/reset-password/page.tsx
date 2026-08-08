"use client";

import { useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { useAuth } from "@/lib/auth";
import {
  AuthCard,
  AuthError,
  AuthPage,
  authInputClass,
} from "@/components/auth/AuthCard";

/**
 * "Forgot password", step one: ask for the mail.
 *
 * Path fixed by HEADLESS_FRONTEND_URLS["account_reset_password"].
 *
 * NOTE THE WORDING of the confirmation. It says we sent mail *if that address
 * has an account* — it never confirms the account exists. That is not
 * hedging: a form that says "no such user" is an account-enumeration oracle,
 * and on a competitive ladder that hands someone the roster of real
 * competitors to target. allauth answers identically either way on purpose,
 * and the copy has to match or it leaks what the API declined to.
 */
export default function RequestResetPage() {
  const { requestPasswordReset } = useAuth();
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await requestPasswordReset(email.trim());
      setSent(true);
    } catch (err: any) {
      setError(err?.message ?? "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  if (sent) {
    return (
      <AuthPage>
        <AuthCard
          title="Check your email"
          subtitle={
            <>
              If <span className="text-ink">{email.trim()}</span> has an
              account, a reset link is on its way. The link expires shortly.
            </>
          }
        >
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

  return (
    <AuthPage>
      <AuthCard
        title="Reset your password"
        subtitle="We'll email you a link to set a new one."
        footer={
          <p className="mt-6 text-center text-sm text-ink-soft">
            Remembered it?{" "}
            <Link
              href="/auth/login"
              className="font-medium text-ink underline underline-offset-2"
            >
              Sign in
            </Link>
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

          <AuthError>{error}</AuthError>

          <Button type="submit" disabled={busy} className="w-full">
            {busy ? "Sending…" : "Send reset link"}
          </Button>
        </form>
      </AuthCard>
    </AuthPage>
  );
}
