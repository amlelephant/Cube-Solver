"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Settings } from "lucide-react";
import { primaryNav } from "@/lib/nav";
import { cn } from "@/lib/cn";

export function NavBar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-mist/0 bg-paper/80 backdrop-blur">
      <div className="mx-auto flex h-20 max-w-6xl items-center justify-between px-6 md:px-10">
        <Link href="/home" className="flex items-center gap-2 text-lg font-semibold tracking-tight">
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
                className={active ? "text-ink" : "transition-colors hover:text-ink"}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-2">
          <Link
            href="/settings"
            aria-label="Settings"
            className={cn(
              "flex size-10 items-center justify-center rounded-full transition-colors hover:bg-cloud",
              pathname === "/settings" ? "text-ink" : "text-ink-faint",
            )}
          >
            <Settings size={19} />
          </Link>
          <Link
            href="/profile"
            aria-label="Your profile"
            className="flex size-10 items-center justify-center rounded-full bg-ink text-sm font-semibold text-paper transition-transform hover:scale-105"
          >
            A
          </Link>
        </div>
      </div>
    </header>
  );
}
