"use client";

import { useEffect, useRef, useState } from "react";
import { VideoOff } from "lucide-react";

export function WebcamFeed({ className }: { className?: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;

    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: "environment" }, audio: false })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = s;
      })
      .catch(() => setError(true));

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  if (error) {
    return (
      <div className={`flex flex-col items-center justify-center gap-3 bg-cloud text-ink-faint ${className}`}>
        <VideoOff size={40} strokeWidth={1.25} />
        <p className="text-sm">Camera unavailable — continuing in demo mode</p>
      </div>
    );
  }

  return (
    <video
      ref={videoRef}
      autoPlay
      muted
      playsInline
      className={`bg-ink object-cover [transform:scaleX(-1)] ${className}`}
    />
  );
}
