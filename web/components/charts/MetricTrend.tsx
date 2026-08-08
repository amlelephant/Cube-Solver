"use client";

import { useMemo, useState } from "react";

/**
 * One metric across solves, in order.
 *
 * Single series by design — the metric is chosen by clicking a tile below
 * the chart, so only ever one line is drawn. That choice removes the two
 * things that make small dashboards unreadable: there is no categorical
 * palette to get wrong, and there is no second y-axis, because two
 * measures of different scale are never on screen together.
 *
 * The line therefore carries no identity information and wears an ink
 * token rather than a series colour; the heading names it.
 */

export interface TrendPoint {
  x: string; // ISO date
  y: number;
  id: string;
}

const W = 720;
const H = 190;
const PAD = { top: 16, right: 18, bottom: 26, left: 46 };

function niceTicks(lo: number, hi: number, count = 3): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

export function MetricTrend({
  points,
  label,
  format,
}: {
  points: TrendPoint[];
  label: string;
  format: (v: number) => string;
}) {
  const [hover, setHover] = useState<number | null>(null);

  const geom = useMemo(() => {
    if (points.length === 0) return null;
    const ys = points.map((p) => p.y);
    let lo = Math.min(...ys);
    let hi = Math.max(...ys);
    if (hi === lo) {
      // A flat series would otherwise divide by zero and collapse onto the
      // axis; give it a band so the line sits mid-plot and reads as flat.
      hi = lo + Math.abs(lo || 1) * 0.1;
      lo = lo - Math.abs(lo || 1) * 0.1;
    } else {
      const pad = (hi - lo) * 0.18;
      lo -= pad;
      hi += pad;
    }
    const iw = W - PAD.left - PAD.right;
    const ih = H - PAD.top - PAD.bottom;
    const xs = (i: number) =>
      PAD.left + (points.length === 1 ? iw / 2 : (i / (points.length - 1)) * iw);
    const yscale = (v: number) => PAD.top + ih - ((v - lo) / (hi - lo)) * ih;
    return {
      lo,
      hi,
      xs,
      yscale,
      ticks: niceTicks(lo, hi),
      d: points.map((p, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)} ${yscale(p.y).toFixed(1)}`).join(" "),
    };
  }, [points]);

  if (!geom || points.length === 0) {
    return (
      <div className="flex h-[190px] items-center justify-center text-xs text-ink-faint">
        Not enough solves to chart {label.toLowerCase()} yet.
      </div>
    );
  }

  const active = hover != null ? points[hover] : null;
  const last = points.length - 1;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full overflow-visible"
        role="img"
        aria-label={`${label} across ${points.length} solves, ${points
          .map((p) => format(p.y))
          .join(", ")}`}
        onMouseLeave={() => setHover(null)}
      >
        {/* recessive grid — present enough to read a value off, quiet
            enough that the line is the only thing with weight */}
        {geom.ticks.map((t) => (
          <g key={t}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={geom.yscale(t)}
              y2={geom.yscale(t)}
              className="stroke-mist"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 8}
              y={geom.yscale(t)}
              dominantBaseline="middle"
              textAnchor="end"
              className="fill-ink-faint text-[10px]"
            >
              {format(t)}
            </text>
          </g>
        ))}

        {active && (
          <line
            x1={geom.xs(hover!)}
            x2={geom.xs(hover!)}
            y1={PAD.top}
            y2={H - PAD.bottom}
            className="stroke-ink-faint"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
        )}

        <path d={geom.d} fill="none" className="stroke-ink" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />

        {points.map((p, i) => (
          <circle
            key={p.id}
            cx={geom.xs(i)}
            cy={geom.yscale(p.y)}
            r={hover === i ? 5.5 : 4}
            className="fill-ink stroke-paper"
            // A surface ring keeps the marker legible where it overlaps the
            // line it sits on.
            strokeWidth={2}
          />
        ))}

        {/* x labels: only the ends, so six dates never collide */}
        <text x={PAD.left} y={H - 8} textAnchor="start" className="fill-ink-faint text-[10px]">
          {points[0].x.slice(5)}
        </text>
        <text x={W - PAD.right} y={H - 8} textAnchor="end" className="fill-ink-faint text-[10px]">
          {points[last].x.slice(5)}
        </text>

        {/* generous invisible hit targets — the markers themselves are far
            too small to aim at */}
        {points.map((p, i) => (
          <rect
            key={`hit-${p.id}`}
            x={geom.xs(i) - 18}
            y={0}
            width={36}
            height={H}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
          />
        ))}
      </svg>

      <figcaption className="mt-1 h-4 text-center text-xs text-ink-faint">
        {active ? (
          <span>
            <span className="font-medium text-ink">{format(active.y)}</span> · {active.x}
          </span>
        ) : (
          <span>
            {points.length} solves · latest {format(points[last].y)}
          </span>
        )}
      </figcaption>
    </figure>
  );
}
