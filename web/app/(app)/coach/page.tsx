"use client";

import Link from "next/link";
import { useState } from "react";
import { ArrowUpRight, Check, Lock } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/cn";
import { useMe } from "@/lib/account";
import { METRICS } from "@/lib/analytics";

/**
 * /coach — what Coach is, and how to get it.
 *
 * Shown to everyone, deliberately: this is the page the locked nav entry
 * points at, so gating it would leave the lock leading nowhere. What changes
 * is the content — a subscriber gets an owned state and links into their
 * data, everyone else gets the pitch and the plans.
 *
 * WHAT THE COPY MAY CLAIM. Every number on this page comes from
 * `lib/analytics.ts`'s registry, which mirrors `coach/report.py`'s measured
 * accuracy. Nothing here rounds those figures up, and the metric count is
 * derived from the registry rather than typed in, so it cannot drift into an
 * overstatement when a metric is added or suppressed.
 *
 * NO CHECKOUT EXISTS. The plan buttons say so rather than opening a payment
 * flow that would fail — see TODO §7D "Billing".
 */

const MONTHLY = 4;
const YEARLY = 36;

type PlanKey = "monthly" | "yearly";

const PLANS: Record<
  PlanKey,
  { label: string; price: string; per: string; note: string }
> = {
  monthly: {
    label: "Monthly",
    price: `$${MONTHLY}`,
    per: "per month",
    note: "Cancel any time.",
  },
  yearly: {
    label: "Yearly",
    price: `$${YEARLY}`,
    per: "per year",
    note: `Works out at $${(YEARLY / 12).toFixed(2)} a month.`,
  },
};

const INCLUDED = [
  {
    title: "Averages, not just your last solve",
    body: "Every measure over your last 5, 12, or all of your solves. One solve moves with a single decode edit; an average does not.",
  },
  {
    title: "Every solve, openable",
    body: "Your whole history, each solve with its own full breakdown. Free accounts see their most recent one.",
  },
  {
    title: "Trends over time",
    body: "Chart any measure across your history and watch it move.",
  },
  {
    title: "Accuracy attached to every number",
    body: "Each figure carries its measured error, and anything we cannot measure well enough in your lighting is left blank rather than guessed.",
  },
];

function Feature({ title, body }: { title: string; body: string }) {
  return (
    <li className="flex gap-3">
      <span
        className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-cloud"
        aria-hidden
      >
        <Check size={13} className="text-ink" />
      </span>
      <span>
        <span className="block text-sm font-medium">{title}</span>
        <span className="mt-0.5 block text-sm text-ink-soft">{body}</span>
      </span>
    </li>
  );
}

export default function CoachPage() {
  const { me, loading } = useMe();
  const [plan, setPlan] = useState<PlanKey>("yearly");
  const premium = !!me?.is_premium;

  // Derived, never typed in — see the header note.
  const metricCount = METRICS.length;
  // The MEDIAN metric's daytime error, not the best one's. `span_seconds`
  // lands at 0.1%, and quoting that as "within 0.1%" would be true of one
  // measure and wildly flattering of the set — the kind of number a reader
  // is right to distrust once they meet the others.
  const typicalErr = (() => {
    const sorted = [...METRICS].map((m) => m.dayErrPct).sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  })();

  if (loading) {
    return (
      <div className="mx-auto max-w-5xl px-6 py-16 md:px-10">
        <div className="h-40 animate-pulse rounded-2xl bg-cloud" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl px-6 pb-24 pt-12 md:px-10">
      <header className="text-center">
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium tracking-wide",
            premium ? "bg-cloud text-ink" : "bg-cloud text-ink-faint",
          )}
        >
          {premium ? <Check size={12} /> : <Lock size={12} />}
          {premium ? "Coach is active" : "Coach"}
        </span>
        <h1 className="mt-4 text-4xl font-bold tracking-tight md:text-5xl">
          {premium ? (
            <>Your solves, <span className="cube-gradient-text">measured</span></>
          ) : (
            <>Know <span className="cube-gradient-text">why</span> you are slow</>
          )}
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-balance text-ink-soft">
          {premium
            ? `All ${metricCount} measures, across every solve you have recorded.`
            : `A timer tells you 14.2 seconds. Coach tells you that six of them were
               spent not turning — from ${metricCount} measures decoded out of your
               webcam and checked against a smart cube.`}
        </p>

        {premium && (
          <Link
            href="/analytics"
            className="mt-6 inline-flex items-center gap-1.5 rounded-lg bg-ink px-5 py-2.5 text-sm font-medium text-paper transition-opacity hover:opacity-90"
          >
            Open your analytics <ArrowUpRight size={16} />
          </Link>
        )}
      </header>

      <div
        className={cn(
          // `items-start` so the pricing card sizes to its content. Stretched
          // to match the taller feature list it grew a ~150px void above the
          // buy button, which read as a section that had failed to load.
          "mt-12 grid items-start gap-6",
          premium ? "grid-cols-1" : "lg:grid-cols-[1.15fr_1fr]",
        )}
      >
        <Card className="p-8">
          <h2 className="text-xl font-semibold tracking-tight">
            {premium ? "What you have" : "What you get"}
          </h2>
          <ul className="mt-6 space-y-5">
            {INCLUDED.map((f) => (
              <Feature key={f.title} {...f} />
            ))}
          </ul>
          <p className="mt-7 border-t border-mist pt-5 text-xs leading-relaxed text-ink-faint">
            Figures are decoded from video and checked against a smart cube on
            solves the model never saw — the typical measure lands within{" "}
            {typicalErr.toFixed(0)}% in good light. Accuracy varies by measure
            and by your lighting, and each number carries its own on the page.
          </p>
        </Card>

        {!premium && (
          <Card className="p-8">
            <h2 className="text-xl font-semibold tracking-tight">Pricing</h2>

            <div
              role="group"
              aria-label="Billing period"
              className="mt-5 inline-flex rounded-lg border border-mist p-0.5"
            >
              {(Object.keys(PLANS) as PlanKey[]).map((k) => (
                <button
                  key={k}
                  type="button"
                  aria-pressed={plan === k}
                  onClick={() => setPlan(k)}
                  className={cn(
                    "flex-1 rounded-[6px] px-4 py-1.5 text-xs font-medium transition-colors",
                    plan === k
                      ? "bg-ink text-paper"
                      : "text-ink-soft hover:text-ink",
                  )}
                >
                  {PLANS[k].label}
                </button>
              ))}
            </div>

            <div className="mt-6 flex items-baseline gap-2">
              <span className="text-5xl font-semibold tracking-tight">
                {PLANS[plan].price}
              </span>
              <span className="text-sm text-ink-faint">{PLANS[plan].per}</span>
            </div>
            <p className="mt-1 text-sm text-ink-soft">{PLANS[plan].note}</p>

            <div className="mt-8">
              <button
                type="button"
                disabled
                className="w-full cursor-not-allowed rounded-lg bg-ink px-5 py-3 text-sm font-medium text-paper opacity-40"
              >
                Not on sale yet
              </button>
              <p className="mt-3 text-center text-xs leading-relaxed text-ink-faint">
                Coach is not purchasable yet — we would rather show you the
                price than a checkout that does not work. Founding members get
                this rate for good.
              </p>
              <Link
                href="/#waitlist"
                className="mt-3 block text-center text-xs font-medium text-ink underline underline-offset-2 hover:opacity-70"
              >
                Tell me when it opens
              </Link>
            </div>
          </Card>
        )}
      </div>

      {!premium && (
        <p className="mt-8 text-center text-sm text-ink-soft">
          Verification stays free, always.{" "}
          <Link href="/analytics" className="underline underline-offset-2 hover:text-ink">
            Your last solve is analysed already
          </Link>
          .
        </p>
      )}
    </div>
  );
}
