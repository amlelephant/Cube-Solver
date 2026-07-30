const MAX_INTENSITY = 4;
const MIN_OPACITY = 0.08;

/** Natural width, but the grid shrinks below this rather than scrolling. */
const MAX_WIDTH = 430; // px

const GRADIENT =
  "conic-gradient(from 200deg at 50% 50%, var(--color-cube-blue), var(--color-cube-green), var(--color-cube-yellow), var(--color-cube-orange), var(--color-cube-red), var(--color-cube-blue))";

/**
 * The backdrop is solid black. Every cell is a window onto the SAME gradient
 * (sized to the full grid, offset per cell) rather than its own flat color —
 * so busier days don't just get "more blue," they reveal more of one
 * continuous field, and quiet days sink back into the black.
 *
 * Both the size and the per-cell offset are percentages of the cell, not
 * pixels, so the whole grid scales with its container: columns flex and cells
 * stay square instead of overflowing into a horizontal scrollbar on narrow
 * screens. `max-w` + `mx-auto` keeps it centered in a wide card.
 */
export function SolveHeatmap({ weeks }: { weeks: number[][] }) {
  const cols = weeks.length;
  const rows = 7;

  return (
    <div>
      <div className="mx-auto rounded-xl bg-black p-2.5" style={{ maxWidth: MAX_WIDTH }}>
        <div className="flex gap-[3px]">
          {weeks.map((week, wi) => (
            <div key={wi} className="flex min-w-0 flex-1 flex-col gap-[3px]">
              {week.map((intensity, di) => {
                const t = intensity / MAX_INTENSITY;
                const opacity = MIN_OPACITY + t * (1 - MIN_OPACITY);
                return (
                  <div
                    key={di}
                    className="aspect-square rounded-[3px]"
                    style={{
                      backgroundImage: GRADIENT,
                      backgroundSize: `${cols * 100}% ${rows * 100}%`,
                      backgroundPosition: `${(wi / (cols - 1)) * 100}% ${(di / (rows - 1)) * 100}%`,
                      opacity,
                    }}
                    title={`${intensity} solve${intensity === 1 ? "" : "s"}`}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <div
        className="mx-auto mt-3 flex items-center justify-end gap-1.5 text-xs text-ink-faint"
        style={{ maxWidth: MAX_WIDTH }}
      >
        <span>Less</span>
        {Array.from({ length: MAX_INTENSITY + 1 }, (_, i) => {
          const t = i / MAX_INTENSITY;
          const opacity = MIN_OPACITY + t * (1 - MIN_OPACITY);
          return (
            <span
              key={i}
              className="size-3 rounded-[3px] bg-black"
              style={{ boxShadow: `inset 0 0 0 12px rgba(36, 85, 230, ${opacity})` }}
            />
          );
        })}
        <span>More</span>
      </div>
    </div>
  );
}
