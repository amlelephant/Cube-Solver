"use client";

import { useRef, useState } from "react";

export function EloChart({ points }: { points: number[] }) {
  const width = 560;
  const height = 220;
  const pad = 24;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;

  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const coords = points.map((p, i) => {
    const x = pad + (i / (points.length - 1)) * (width - pad * 2);
    const y = height - pad - ((p - min) / range) * (height - pad * 2);
    return [x, y] as const;
  });

  const linePath = coords.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ");
  const areaPath = `${linePath} L${coords[coords.length - 1][0]},${height - pad} L${coords[0][0]},${height - pad} Z`;

  function handleMove(e: React.MouseEvent<SVGSVGElement>) {
    const svg = svgRef.current;
    if (!svg || coords.length === 0) return;
    const rect = svg.getBoundingClientRect();
    const localX = ((e.clientX - rect.left) / rect.width) * width;
    let nearest = 0;
    let nearestDist = Infinity;
    coords.forEach(([x], i) => {
      const d = Math.abs(x - localX);
      if (d < nearestDist) {
        nearestDist = d;
        nearest = i;
      }
    });
    setHoverIndex(nearest);
  }

  const hovered = hoverIndex !== null ? coords[hoverIndex] : null;
  const tooltipLeftPct = hovered ? (hovered[0] / width) * 100 : 0;
  const tooltipTopPct = hovered ? (hovered[1] / height) * 100 : 0;
  const nearLeftEdge = tooltipLeftPct < 12;
  const nearRightEdge = tooltipLeftPct > 88;

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full cursor-crosshair"
        role="img"
        aria-label="Rating history chart"
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <defs>
          <linearGradient id="elo-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-cube-blue)" stopOpacity="0.25" />
            <stop offset="100%" stopColor="var(--color-cube-blue)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill="url(#elo-fill)" />
        <path
          d={linePath}
          fill="none"
          stroke="var(--color-cube-blue)"
          strokeWidth={2.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {hovered && (
          <line
            x1={hovered[0]}
            x2={hovered[0]}
            y1={pad}
            y2={height - pad}
            stroke="var(--color-mist)"
            strokeWidth={1}
          />
        )}

        {coords.length > 0 && (
          <circle
            cx={coords[coords.length - 1][0]}
            cy={coords[coords.length - 1][1]}
            r={5}
            fill="var(--color-cube-blue)"
            className="pointer-events-none"
          />
        )}

        {hovered && (
          <circle
            cx={hovered[0]}
            cy={hovered[1]}
            r={5}
            fill="var(--color-cube-blue)"
            stroke="var(--color-paper)"
            strokeWidth={2}
            className="pointer-events-none"
          />
        )}
      </svg>

      {hovered && hoverIndex !== null && (
        <div
          className="pointer-events-none absolute rounded-lg border border-mist bg-paper px-2.5 py-1.5 text-xs shadow-card"
          style={{
            left: `${tooltipLeftPct}%`,
            top: `${tooltipTopPct}%`,
            transform: `translate(${nearLeftEdge ? "-10%" : nearRightEdge ? "-90%" : "-50%"}, -130%)`,
          }}
        >
          <p className="font-semibold text-ink">{points[hoverIndex]} rating</p>
          <p className="text-ink-faint">Solve {hoverIndex + 1}</p>
        </div>
      )}
    </div>
  );
}
