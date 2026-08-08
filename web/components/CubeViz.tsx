"use client";

import { useEffect, useRef, useState } from "react";
import {
  ContextLostError,
  createCubeRenderer,
  parseAlg,
  type CubeHandle,
  type FaceKey,
} from "@/lib/cube/renderer";

const ALL_LIT: Record<FaceKey, number> = { U: 1, D: 1, L: 1, R: 1, F: 1, B: 1 };

export interface CubeVizProps {
  /** Quarter turns per second. 0 holds the cube still. */
  tps?: number;
  /** 0..1 albedo per face; omitted faces render at full brightness. */
  faceGain?: Partial<Record<FaceKey, number>>;
  /** WCA algorithm, looped. Quarter turns only — the decoder emits QTM. */
  alg?: string;
  /** Degrees per second of idle yaw. */
  spin?: number;
  paused?: boolean;
  /** Allow drag-to-rotate; the cube eases back to its default on release. */
  interactive?: boolean;
  className?: string;
  /** Describes what the cube is showing, for anyone not seeing it. */
  label: string;
}

/**
 * The landing page's cube, driven by a metric.
 *
 * WebGL2 only, and it fails to a caption rather than to a blank box: the
 * canvas sits under a text fallback that the renderer covers when it
 * starts. Every number the cube visualises is also printed beside it on
 * the page, so losing the canvas costs the illustration and never the
 * information.
 */
export function CubeViz({
  tps = 0,
  faceGain,
  alg = "",
  spin = 8,
  paused = false,
  interactive = false,
  className,
  label,
}: CubeVizProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const handle = useRef<CubeHandle | null>(null);
  const [failed, setFailed] = useState(false);
  /**
   * Bumping this remounts the <canvas> element itself, which is what a retry
   * after context loss actually requires: `getContext` on a canvas whose
   * context was lost hands back that same dead context, so retrying on the
   * old element can never succeed. A new element gets a new allocation.
   */
  const [canvasKey, setCanvasKey] = useState(0);

  // Mount once. The renderer owns its own rAF loop and reads params from a
  // closure, so prop changes go through update() rather than remounting —
  // remounting would restart the algorithm on every slider tick.
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    let disposed = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;

    function build() {
      if (disposed || handle.current) return;
      try {
        handle.current = createCubeRenderer(canvas!, {
          tps,
          faceGain: { ...ALL_LIT, ...faceGain },
          moves: parseAlg(alg),
          spin,
          paused,
          interactive,
        });
        if (!handle.current) setFailed(true); // driver rejected the shader
      } catch (err) {
        if (err instanceof ContextLostError && attempts < 2) {
          // The browser caps live WebGL contexts (16 per tab in Chrome) and
          // frees them asynchronously, so a context can arrive dead through
          // no fault of ours. Retry on a FRESH canvas — see canvasKey; the
          // old element's context is gone for good.
          attempts += 1;
          retry = setTimeout(() => setCanvasKey((k) => k + 1), 200 * attempts);
          return;
        }
        console.error("[CubeViz] renderer failed to start —", err);
        setFailed(true);
      }
    }

    // A context can also be lost long after a successful start — a GPU reset
    // or driver update is enough. Rebuild rather than leaving a frozen cube
    // on screen with no explanation.
    function onLost(e: Event) {
      e.preventDefault();
      if (disposed) return; // our own destroy() fired this; ignore it
      handle.current = null;
      retry = setTimeout(() => setCanvasKey((k) => k + 1), 200);
    }

    canvas.addEventListener("webglcontextlost", onLost);
    build();

    return () => {
      disposed = true;
      clearTimeout(retry);
      canvas.removeEventListener("webglcontextlost", onLost);
      handle.current?.destroy();
      handle.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canvasKey]);

  useEffect(() => {
    handle.current?.update({ tps, spin, paused, interactive });
  }, [tps, spin, paused, interactive]);

  useEffect(() => {
    handle.current?.update({ faceGain: { ...ALL_LIT, ...faceGain } });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(faceGain)]);

  useEffect(() => {
    handle.current?.update({ moves: parseAlg(alg) });
  }, [alg]);

  return (
    <div className={className}>
      {failed ? (
        <div className="flex size-full items-center justify-center p-6">
          <p className="max-w-[22ch] text-center text-xs leading-relaxed text-ink-faint">
            This illustration needs WebGL, which your browser did not start.
            The numbers beside it are unaffected.
          </p>
        </div>
      ) : (
        <canvas
          key={canvasKey}
          ref={ref}
          className="size-full"
          role="img"
          aria-label={label}
        />
      )}
    </div>
  );
}
