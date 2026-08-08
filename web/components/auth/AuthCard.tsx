"use client";

import Link from "next/link";
import { type ReactNode } from "react";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/cn";

/**
 * The shell every `/auth/*` screen sits in.
 *
 * Extracted rather than copy-pasted per page: there are five of these screens
 * (sign in, sign up, verify email, request a reset, set a new password) and
 * they are the first thing anyone sees of the product. Five hand-maintained
 * copies of the same card drift, and the drift shows up as the login flow
 * looking subtly cheaper on the screens people hit when something has already
 * gone wrong.
 */
export function AuthCard({
  title,
  subtitle,
  children,
  footer,
  className,
}: {
  title: string;
  subtitle?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  className?: string;
}) {
  return (
    <Card className={cn("w-full max-w-md p-8", className)}>
      <div className="mb-7 text-center">
        <span
          className="mx-auto mb-4 inline-block size-5 rounded-[4px] cube-gradient"
          aria-hidden
        />
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-2 text-sm text-ink-soft">{subtitle}</p>}
      </div>
      {children}
      {footer}
      <p className="mt-8 text-center text-xs text-ink-faint">
        <Link href="/" className="hover:text-ink">
          &larr; Back to cubearena
        </Link>
      </p>
    </Card>
  );
}

/** The page-level wrapper: centres the card on its own full-height screen. */
export function AuthPage({ children }: { children: ReactNode }) {
  return (
    <main className="flex min-h-screen items-center justify-center bg-cloud px-6 py-16">
      {children}
    </main>
  );
}

export const authInputClass = cn(
  "w-full rounded-lg border border-mist bg-cloud px-3.5 py-2.5 text-sm",
  "text-ink placeholder:text-ink-faint",
  "outline-none transition-colors focus:border-ink/30 focus:bg-paper",
);

/** Inline error, styled once so every screen reports failure identically. */
export function AuthError({ children }: { children: ReactNode }) {
  if (!children) return null;
  return (
    <p
      role="alert"
      className="rounded-lg bg-cube-red/10 px-3 py-2 text-sm text-cube-red"
    >
      {children}
    </p>
  );
}
