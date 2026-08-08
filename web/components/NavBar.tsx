"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Lock, LogOut, Menu, Settings, X } from "lucide-react";
import { Avatar } from "@/components/Avatar";
import { primaryNav } from "@/lib/nav";
import { cn } from "@/lib/cn";
import { useAuth } from "@/lib/auth";
import { useMe } from "@/lib/account";

/**
 * The app header.
 *
 * RESPONSIVE BEHAVIOUR. The links collapse into a sheet below `md`. They
 * used to be `hidden md:flex` with no replacement, so at half-width — a
 * perfectly ordinary desktop window, not just phones — the entire primary
 * navigation simply vanished and the only way to reach another page was the
 * footer.
 *
 * The Coach entry carries a lock for accounts without it. That is a
 * MARKER, not a guard: the page stays reachable, because it is where you go
 * to buy it. Every real gate is server-side.
 */
export function NavBar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout } = useAuth();
  const { me } = useMe();
  const [open, setOpen] = useState(false);

  // Route changes have to close the sheet, or tapping a link leaves the
  // overlay covering the page you just navigated to.
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  // Escape closes it, matching the Modal's behaviour elsewhere.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Real address, not a hardcoded "A". Falls back rather than rendering an
  // empty circle while the session is still resolving.
  const initial = (user?.email ?? "?").trim().charAt(0).toUpperCase();
  const locked = (item: { premium?: boolean }) => !!item.premium && !me?.is_premium;

  async function onSignOut() {
    await logout();
    router.replace("/auth/login");
  }

  return (
    <header className="sticky top-0 z-40 border-b border-mist/0 bg-paper/80 backdrop-blur">
      <div className="mx-auto flex h-20 max-w-6xl items-center justify-between gap-3 px-6 md:px-10">
        <Link
          href="/home"
          className="flex shrink-0 items-center gap-2 text-lg font-semibold tracking-tight"
        >
          <span className="inline-block size-4 rounded-[3px] cube-gradient" aria-hidden />
          Cube Arena
        </Link>

        <nav className="hidden items-center gap-8 text-sm font-medium text-ink-soft md:flex">
          {primaryNav.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "inline-flex items-center gap-1.5",
                  active ? "text-ink" : "transition-colors hover:text-ink",
                )}
              >
                {item.label}
                {locked(item) && (
                  <Lock size={12} className="text-ink-faint" aria-label="Requires Coach" />
                )}
              </Link>
            );
          })}
        </nav>

        <div className="flex shrink-0 items-center gap-1 sm:gap-2">
          <Link
            href="/settings"
            aria-label="Settings"
            className={cn(
              "hidden size-10 items-center justify-center rounded-full transition-colors hover:bg-cloud sm:flex",
              pathname === "/settings" ? "text-ink" : "text-ink-faint",
            )}
          >
            <Settings size={19} />
          </Link>
          <button
            onClick={onSignOut}
            aria-label="Sign out"
            title="Sign out"
            className="hidden size-10 items-center justify-center rounded-full text-ink-faint transition-colors hover:bg-cloud hover:text-ink sm:flex"
          >
            <LogOut size={18} />
          </button>
          <Link
            href="/profile"
            aria-label="Your profile"
            title={user?.email ?? "Your profile"}
            className="flex items-center justify-center rounded-full transition-transform hover:scale-105"
          >
            {me?.avatar ? (
              <Avatar size={40} preset={me.avatar} />
            ) : (
              <span className="flex size-10 items-center justify-center rounded-full bg-ink text-sm font-semibold text-paper">
                {initial}
              </span>
            )}
          </Link>

          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            aria-controls="primary-menu"
            className="flex size-10 items-center justify-center rounded-full text-ink-faint transition-colors hover:bg-cloud hover:text-ink md:hidden"
          >
            {open ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {open && (
        <>
          {/* Scrim below the sheet but above the page. Clicking it closes,
              which is the gesture people try first. */}
          <button
            type="button"
            aria-hidden
            tabIndex={-1}
            onClick={() => setOpen(false)}
            className="fixed inset-0 top-20 z-30 cursor-default bg-ink/20 md:hidden"
          />
          <nav
            id="primary-menu"
            className="relative z-40 border-t border-mist bg-paper px-6 pb-4 pt-2 md:hidden"
          >
            {primaryNav.map((item) => {
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 rounded-lg px-2 py-3 text-base font-medium",
                    active ? "bg-cloud text-ink" : "text-ink-soft hover:bg-cloud/60",
                  )}
                >
                  {item.label}
                  {locked(item) && (
                    <Lock size={13} className="text-ink-faint" aria-label="Requires Coach" />
                  )}
                </Link>
              );
            })}

            <div className="mt-2 flex items-center gap-2 border-t border-mist pt-3 sm:hidden">
              <Link
                href="/settings"
                className="flex flex-1 items-center gap-2 rounded-lg px-2 py-3 text-base font-medium text-ink-soft hover:bg-cloud/60"
              >
                <Settings size={17} /> Settings
              </Link>
              <button
                type="button"
                onClick={onSignOut}
                className="flex items-center gap-2 rounded-lg px-3 py-3 text-base font-medium text-ink-soft hover:bg-cloud/60"
              >
                <LogOut size={17} /> Sign out
              </button>
            </div>
          </nav>
        </>
      )}
    </header>
  );
}
