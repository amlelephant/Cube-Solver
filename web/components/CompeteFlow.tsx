"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { User } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button, LinkButton } from "@/components/ui/Button";
import { WebcamFeed } from "@/components/WebcamFeed";

type Stage = "verify-scramble" | "solving" | "verify-solve" | "result";

const FACES = [
  { name: "Up", color: "White", swatch: "#e9e9e9" },
  { name: "Right", color: "Red", swatch: "var(--color-cube-red)" },
  { name: "Front", color: "Green", swatch: "var(--color-cube-green)" },
  { name: "Down", color: "Yellow", swatch: "var(--color-cube-yellow)" },
  { name: "Left", color: "Orange", swatch: "var(--color-cube-orange)" },
  { name: "Back", color: "Blue", swatch: "var(--color-cube-blue)" },
];

function formatTime(ms: number) {
  const totalCs = Math.floor(ms / 10);
  const minutes = Math.floor(totalCs / 6000);
  const seconds = Math.floor((totalCs % 6000) / 100);
  const centis = totalCs % 100;
  return [minutes, seconds, centis].map((n) => String(n).padStart(2, "0")).join(":");
}

function VerifyStage({
  heading,
  onDone,
}: {
  heading: string;
  onDone: () => void;
}) {
  const [faceIndex, setFaceIndex] = useState(0);
  const [flash, setFlash] = useState(false);
  const face = FACES[faceIndex];

  const capture = useCallback(() => {
    setFlash(true);
    setTimeout(() => setFlash(false), 120);
    if (faceIndex === FACES.length - 1) {
      onDone();
    } else {
      setFaceIndex((i) => i + 1);
    }
  }, [faceIndex, onDone]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        e.preventDefault();
        capture();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [capture]);

  return (
    <section className="mx-auto flex max-w-2xl flex-col items-center px-6 py-20 text-center">
      <p className="text-3xl font-medium text-ink/40">{heading}</p>

      <div
        onClick={capture}
        className={`relative mt-10 h-96 w-full cursor-pointer overflow-hidden rounded-2xl transition-shadow ${flash ? "ring-4 ring-cube-green" : ""}`}
      >
        <WebcamFeed className="size-full" />
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <span className="rounded-md bg-black/40 px-4 py-2 text-sm font-medium text-white backdrop-blur">
            Verify Cube State — webcam feed
          </span>
        </div>
      </div>

      <div className="mt-10 flex items-center gap-2 text-2xl font-medium">
        Current Face :
        <span className="flex items-center gap-2" style={{ color: face.swatch }}>
          <span className="inline-block size-4 rounded-sm" style={{ background: face.swatch }} />
          {face.color}
        </span>
      </div>

      <button
        onClick={capture}
        className="mt-8 animate-pulse-rec text-lg font-medium text-ink/40 transition-colors hover:text-ink"
      >
        Space To Capture Face
      </button>
    </section>
  );
}

export function CompeteFlow({ mode }: { mode: "solo" | "live" }) {
  const router = useRouter();
  const [stage, setStage] = useState<Stage>("verify-scramble");
  const [elapsed, setElapsed] = useState(0);
  const [opponentSolved, setOpponentSolved] = useState(false);
  const [finalTime, setFinalTime] = useState(0);
  const startRef = useRef<number | null>(null);
  const frameRef = useRef<number>(0);
  const opponentTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (stage !== "solving") return;
    startRef.current = performance.now();

    const tick = () => {
      setElapsed(performance.now() - (startRef.current ?? performance.now()));
      frameRef.current = requestAnimationFrame(tick);
    };
    frameRef.current = requestAnimationFrame(tick);

    if (mode === "live") {
      const opponentMs = 8000 + Math.random() * 9000;
      opponentTimeoutRef.current = setTimeout(() => setOpponentSolved(true), opponentMs);
    }

    return () => {
      cancelAnimationFrame(frameRef.current);
      if (opponentTimeoutRef.current) clearTimeout(opponentTimeoutRef.current);
    };
  }, [stage, mode]);

  const endSolve = useCallback(() => {
    if (stage !== "solving") return;
    cancelAnimationFrame(frameRef.current);
    setFinalTime(elapsed);
    setStage("verify-solve");
  }, [stage, elapsed]);

  useEffect(() => {
    if (stage !== "solving") return;
    const handler = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        e.preventDefault();
        endSolve();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [stage, endSolve]);

  const won = useMemo(() => finalTime / 1000 < 12.98, [finalTime]);
  const eloDelta = mode === "solo" ? null : won ? 18 : -12;

  if (stage === "verify-scramble") {
    return <VerifyStage heading="Verify Scramble" onDone={() => setStage("solving")} />;
  }

  if (stage === "verify-solve") {
    return <VerifyStage heading="Verify Solve" onDone={() => setStage("result")} />;
  }

  if (stage === "result") {
    const resultLabel =
      mode === "solo" ? "Solve Verified" : won ? "You Win" : "Opponent Wins";
    return (
      <section className="mx-auto flex max-w-2xl flex-col items-center px-6 py-24 text-center">
        <p className="text-4xl font-medium">{resultLabel}</p>

        <div className="mt-8 flex size-[250px] items-center justify-center rounded-2xl bg-cloud">
          <User size={72} strokeWidth={1.1} className="text-ink-faint" />
        </div>
        <p className="mt-4 text-lg text-ink-faint">
          {mode === "solo" ? "Aiden" : won ? "Aiden" : "mira_cubes"}
        </p>

        <p className="mt-10 text-2xl font-semibold cube-gradient-text">{formatTime(finalTime)}</p>

        <div className="mt-6 flex w-full max-w-md items-center justify-center rounded-lg bg-cloud px-6 py-3 text-sm">
          {eloDelta === null
            ? "No rating change — solo trial"
            : `Rating ${eloDelta > 0 ? "+" : ""}${eloDelta}`}
        </div>

        <div className="mt-8 flex gap-4">
          <Button variant="secondary" onClick={() => router.push("/home")}>
            Home
          </Button>
          <LinkButton href={`/compete/play?mode=${mode}`}>Play Again</LinkButton>
        </div>
      </section>
    );
  }

  // stage === "solving"
  return (
    <section className="mx-auto flex max-w-2xl flex-col items-center px-6 py-24 text-center">
      <p className="text-6xl font-semibold tabular-nums tracking-tight md:text-7xl">
        {formatTime(elapsed)}
      </p>

      {mode === "live" && (
        <p className="mt-6 text-3xl font-medium">
          Opponent :{" "}
          <span className={opponentSolved ? "text-cube-green" : "text-cube-red"}>
            {opponentSolved ? "Solved" : "Unsolved"}
          </span>
        </p>
      )}

      <button
        onClick={endSolve}
        className="mt-14 animate-pulse-rec text-lg font-medium text-ink/40 transition-colors hover:text-ink"
      >
        Space To End Solve
      </button>
    </section>
  );
}
