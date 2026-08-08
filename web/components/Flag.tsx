import { cn } from "@/lib/cn";
import { countryName } from "@/lib/countries";

/**
 * A country flag, as an image.
 *
 * WHY NOT THE EMOJI. `countryFlag()` returns a regional-indicator pair —
 * 🇺🇸 — which renders as a flag on macOS, iOS and Android, and as the bare
 * letters "US" on Windows, which ships no country-flag glyphs in any system
 * font. That is not a rounding error: Windows is most of the desktop
 * audience, and "the flag is just two grey letters" was the actual state of
 * the leaderboard before this component existed. There is no font or CSS
 * workaround; the only fix is real artwork.
 *
 * The SVGs are vendored in `public/flags/` from `flag-icons` v7.5.0 (MIT —
 * see `public/flags/LICENSE.txt`), rather than kept as a dependency: they
 * are static assets that never need a build step, and the package's CSS
 * would have shipped 271 rules to use two of them. Each file is well under
 * 1KB and only the flags actually on screen are ever fetched.
 *
 * ACCESSIBILITY. The flag is decorative wherever the country name or code is
 * already adjacent — pass `decorative` there so a screen reader does not
 * announce "United States" twice. Standing alone it keeps its label.
 */
export function Flag({
  code,
  className,
  decorative = false,
}: {
  /** ISO 3166-1 alpha-2. Anything falsy renders nothing. */
  code?: string | null;
  className?: string;
  decorative?: boolean;
}) {
  if (!code) return null;
  const lower = code.toLowerCase();
  const name = countryName(code) ?? code;

  return (
    // eslint-disable-next-line @next/next/no-img-element -- a 600-byte static
    // SVG needs no optimisation pass, and next/image would add a loader
    // round-trip per flag on a page that shows twenty-five of them.
    <img
      src={`/flags/${lower}.svg`}
      alt={decorative ? "" : name}
      title={decorative ? undefined : name}
      aria-hidden={decorative || undefined}
      width={20}
      height={15}
      loading="lazy"
      decoding="async"
      className={cn(
        // A 4:3 flag in a fixed box, with a hairline so white flags (JP, PL)
        // do not dissolve into a light background.
        "inline-block h-[0.85em] w-[1.13em] shrink-0 rounded-[2px] object-cover",
        "ring-1 ring-inset ring-ink/10",
        className,
      )}
    />
  );
}
