"use client";

import { use, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { PasswordInput } from "@/components/ui/PasswordInput";
import { useAuth } from "@/lib/auth";
import { ApiError } from "@/lib/api";
import { AuthCard, AuthError, AuthPage } from "@/components/auth/AuthCard";

/**
 * "Forgot password", step two: set the new one.
 *
 * Path fixed by HEADLESS_FRONTEND_URLS["account_reset_password_from_key"].
 *
 * The key is NOT validated on mount, deliberately. allauth exposes a GET that
 * would check it, but mail scanners prefetch links, and on this flow a
 * prefetch that consumes the key locks the real person out of their own
 * account recovery. So the key is spent only when the form is submitted —
 * which is also the first moment we have anything to spend it on.
 */
export default function ResetPasswordPage({
  params,
}: {
  params: Promise<{ key: string }>;
}) {
  const { key } = use(params);
  const { resetPassword } = useAuth();
  const router = useRouter();

  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expired, setExpired] = useState(false);
  const [done, setDone] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (password !== confirm) {
      setError("Those two passwords don't match.");
      return;
    }

    setBusy(true);
    try {
      const { signedIn } = await resetPassword(decodeURIComponent(key), password);
      // A reset does not sign you in by default (see resetPassword) — say the
      // password changed and hand over a sign-in form, rather than bouncing
      // off RequireAuth into a login page that looks like the reset failed.
      if (signedIn) router.replace("/home");
      else setDone(true);
    } catch (err: any) {
      if (err instanceof ApiError && (err.status === 400 || err.status === 409)) {
        // 400 covers BOTH a spent/expired key and a rejected password, so the
        // message has to carry both readings rather than guess wrong.
        const detail = err?.message;
        if (detail && !/token|key|invalid|expired/i.test(detail)) {
          setError(detail);
        } else {
          setExpired(true);
        }
      } else {
        setError(err?.message ?? "Something went wrong.");
      }
    } finally {
      setBusy(false);
    }
  }

  if (done) {
    return (
      <AuthPage>
        <AuthCard
          title="Password changed"
          subtitle="Sign in with your new password and you're back in."
        >
          <Button className="w-full" onClick={() => router.replace("/auth/login")}>
            Sign in
          </Button>
        </AuthCard>
      </AuthPage>
    );
  }

  if (expired) {
    return (
      <AuthPage>
        <AuthCard
          title="This link has expired"
          subtitle="Reset links are single-use and time-limited. Request a fresh one and it'll arrive in a moment."
        >
          <Button
            className="w-full"
            onClick={() => router.replace("/auth/reset-password")}
          >
            Request a new link
          </Button>
        </AuthCard>
      </AuthPage>
    );
  }

  return (
    <AuthPage>
      <AuthCard
        title="Set a new password"
        subtitle="Pick something you don't use anywhere else."
        footer={
          <p className="mt-6 text-center text-sm text-ink-soft">
            <Link
              href="/auth/login"
              className="font-medium text-ink underline underline-offset-2"
            >
              Back to sign in
            </Link>
          </p>
        }
      >
        <form onSubmit={onSubmit} className="space-y-4" noValidate>
          <div>
            <label
              htmlFor="password"
              className="mb-1.5 block text-sm font-medium"
            >
              New password
            </label>
            <PasswordInput
              id="password"
              autoComplete="new-password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 8 characters"
            />
          </div>

          <div>
            <label
              htmlFor="confirm"
              className="mb-1.5 block text-sm font-medium"
            >
              Confirm password
            </label>
            <PasswordInput
              id="confirm"
              autoComplete="new-password"
              required
              minLength={8}
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              placeholder="••••••••"
            />
          </div>

          <AuthError>{error}</AuthError>

          <Button type="submit" disabled={busy} className="w-full">
            {busy ? "Saving…" : "Set new password"}
          </Button>
        </form>
      </AuthCard>
    </AuthPage>
  );
}
