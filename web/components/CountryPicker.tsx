"use client";

import { COUNTRIES } from "@/lib/countries";
import { Flag } from "@/components/Flag";
import { cn } from "@/lib/cn";

/**
 * Pick the country you rep.
 *
 * A NATIVE `<select>` on purpose. 250 options is exactly the case where a
 * custom dropdown gets worse than the platform one: the native control gets
 * type-ahead, keyboard paging, screen-reader support and a scroll position
 * that survives, all for free, and on mobile it becomes the OS picker.
 *
 * Nothing here reads locale or IP. The list is a menu, not a detection —
 * you rep whoever you want, wherever you happen to be.
 */
export function CountryPicker({
  value,
  onChange,
  disabled,
  id = "country",
  className,
}: {
  value: string | null;
  onChange: (code: string) => void;
  disabled?: boolean;
  id?: string;
  className?: string;
}) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <span className="flex w-6 justify-center text-base">
        {value ? (
          <Flag code={value} decorative />
        ) : (
          // A dashed placeholder rather than a white flag: an empty box that
          // is clearly a slot reads as "nothing chosen", where a blank flag
          // reads as a country whose artwork failed to load.
          <span
            aria-hidden
            className="h-[0.85em] w-[1.13em] rounded-[2px] border border-dashed border-mist"
          />
        )}
      </span>
      <select
        id={id}
        value={value ?? ""}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className={cn(
          "w-52 rounded-lg border border-mist bg-cloud px-3 py-2 text-sm text-ink",
          "outline-none transition-colors focus:border-ink/30 focus:bg-paper",
          "disabled:opacity-50",
        )}
      >
        <option value="">No country</option>
        {COUNTRIES.map((c) => (
          <option key={c.code} value={c.code}>
            {c.name}
          </option>
        ))}
      </select>
    </div>
  );
}
