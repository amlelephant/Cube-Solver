"""
frame_time_audit.py — is `frame / nominal_fps` good enough to time a solve?

TODO.md §7A calls this a defect and it may well be one, but nobody had
measured it. Every L1 coach metric is a DIFFERENCE OF ONSET TIMESTAMPS, and
those timestamps are currently computed as `frame_index / fps` where `fps` is
a single scalar — the session mean, `n / (ts[-1] - ts[0])` (prepare_data.py).
Webcam frame intervals jitter, so the nominal clock and the real one disagree.
The question this file answers is BY HOW MUCH, in the units the product ships:

    * per-onset error         nominal time vs the frame's real capture time.
                              A constant offset here is harmless — it is the
                              same object as the model's onset bias, which
                              onset_timing.py already measured and found
                              cancels under differencing.
    * IOI error               the one that matters. Every metric (hesitation,
                              execution TPS, move duration, the pause rule) is
                              built on inter-onset intervals, and an error
                              that survives differencing lands directly on the
                              metric instead of averaging out.
    * pause-rule flips        an IOI straddling the pause threshold decided
                              one way by the nominal clock and the other by
                              the real one. This is the failure with teeth:
                              §7C measured that thresholded counts are the
                              least robust statistic kind in the whole table,
                              so a clock error that flips them is worse than
                              its size suggests.

WHAT IT COMPARES AGAINST
------------------------
The measured onset-timing jitter floor is 41-48 ms MAD (results/2026-08-06/
onset_timing_s*.json), which is ~1.3 frames at 30 fps and is quantisation,
not model error. A clock error well under that floor is real but invisible:
it cannot be the reason a metric is wrong, because a larger error from the
frame rate itself is already there. That is the bar used in the summary.

The real per-frame capture times live in each session's `frames.jsonl` (the
`ts` field, written at capture by record_training.py / verify_solve.py's
save_take). They are NOT carried into the prepared `.npz`, which stores only
the scalar `fps` — which is exactly why the nominal clock is what the decode
currently emits.

Run from inside ble/move_detector:
    python frame_time_audit.py
    python frame_time_audit.py --json results/frame_time_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SESSION_ROOT = Path("../training_data")

#: The 30fps quantisation floor on onset timing, measured both seeds
#: (results/2026-08-06/onset_timing_s*.json, IOI MAD 41-48 ms). A clock error
#: below this cannot be the thing making a metric wrong.
JITTER_FLOOR_MS = 41.0

#: coach.timing's pause rule, restated here rather than imported so this file
#: stays a measurement of the CLOCK and does not drag in the metric stack.
PAUSE_FACTOR = 3.0
PAUSE_FLOOR_S = 0.25


def session_times(d: Path) -> np.ndarray | None:
    """Real per-frame capture times, seconds from the first frame."""
    p = d / "frames.jsonl"
    if not p.is_file():
        return None
    ts = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                ts.append(json.loads(line)["ts"])
    if len(ts) < 2:
        return None
    t = np.asarray(ts, dtype=np.float64)
    return t - t[0]


def onset_frames(d: Path) -> np.ndarray | None:
    """True onset frame indices from the prepared stream, if it exists.

    True onsets rather than decoded ones deliberately: this file is measuring
    the CLOCK, and using decoded onsets would fold the model's own errors into
    a number that is supposed to be about frame timestamps alone.
    """
    p = d / "detector_stream_color.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    idx = z["onset_idx"].astype(int)
    return idx if idx.size >= 2 else None


def pause_threshold(ioi: np.ndarray) -> float:
    if ioi.size < 2:
        return PAUSE_FLOOR_S
    return max(PAUSE_FACTOR * float(np.median(ioi)), PAUSE_FLOOR_S)


def audit_session(d: Path) -> dict | None:
    real = session_times(d)
    if real is None:
        return None
    idx = onset_frames(d)
    n = real.size
    dur = float(real[-1])
    fps = n / dur if dur > 0 else 30.0

    # The whole-stream view: how far the nominal clock drifts from the real
    # one at every frame. Reported because it is what someone would look at
    # first, and because it is the number that most overstates the problem.
    nominal_all = np.arange(n) / fps
    frame_err_ms = (nominal_all - real) * 1000.0

    row = {
        "session": d.name,
        "n_frames": n,
        "duration_s": round(dur, 2),
        "nominal_fps": round(fps, 3),
        "frame_interval_ms": {
            "median": round(float(np.median(np.diff(real))) * 1000, 2),
            "p95": round(float(np.percentile(np.diff(real), 95)) * 1000, 2),
            "max": round(float(np.diff(real).max()) * 1000, 2),
        },
        "frame_time_err_ms": {
            "median_abs": round(float(np.median(np.abs(frame_err_ms))), 2),
            "max_abs": round(float(np.abs(frame_err_ms).max()), 2),
        },
    }

    if idx is None or idx.size < 3:
        row["onsets"] = None
        return row

    idx = np.clip(idx, 0, n - 1)
    t_real = real[idx]
    t_nom = idx / fps

    ioi_real = np.diff(t_real)
    ioi_nom = np.diff(t_nom)
    ioi_err_ms = (ioi_nom - ioi_real) * 1000.0

    # The pause rule, decided twice. Each clock derives its OWN threshold from
    # its OWN median — which is how the product works, and it is also what
    # partly self-corrects a uniform clock stretch.
    thr_r, thr_n = pause_threshold(ioi_real), pause_threshold(ioi_nom)
    pause_r, pause_n = ioi_real > thr_r, ioi_nom > thr_n
    flips = int((pause_r != pause_n).sum())

    hes_r, hes_n = float(ioi_real[pause_r].sum()), float(ioi_nom[pause_n].sum())
    exec_r = float(ioi_real[~pause_r].sum())
    exec_n = float(ioi_nom[~pause_n].sum())
    tps_r = (~pause_r).sum() / exec_r if exec_r > 0 else None
    tps_n = (~pause_n).sum() / exec_n if exec_n > 0 else None

    row["onsets"] = {
        "n": int(idx.size),
        "onset_time_err_ms": {
            "median": round(float(np.median(t_nom - t_real)) * 1000, 2),
            "median_abs": round(float(np.median(np.abs(t_nom - t_real))) * 1000, 2),
            "max_abs": round(float(np.abs(t_nom - t_real).max()) * 1000, 2),
        },
        "ioi_err_ms": {
            "median": round(float(np.median(ioi_err_ms)), 3),
            "median_abs": round(float(np.median(np.abs(ioi_err_ms))), 3),
            "mad": round(float(np.median(np.abs(ioi_err_ms
                                                - np.median(ioi_err_ms)))), 3),
            "p95_abs": round(float(np.percentile(np.abs(ioi_err_ms), 95)), 3),
            "max_abs": round(float(np.abs(ioi_err_ms).max()), 3),
        },
        "pause_flips": flips,
        "n_pauses_real": int(pause_r.sum()),
        "n_pauses_nominal": int(pause_n.sum()),
        "hesitation_seconds": [round(hes_r, 3), round(hes_n, 3)],
        "hesitation_err_pct": (round(100 * (hes_n - hes_r) / hes_r, 3)
                               if hes_r > 0 else None),
        "execution_tps": [round(tps_r, 4) if tps_r else None,
                          round(tps_n, 4) if tps_n else None],
        "execution_tps_err_pct": (round(100 * (tps_n - tps_r) / tps_r, 3)
                                  if tps_r else None),
    }
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="write the rows here")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dirs = sorted(d for d in SESSION_ROOT.iterdir() if d.is_dir())
    if args.limit:
        dirs = dirs[:args.limit]

    rows = []
    for d in dirs:
        r = audit_session(d)
        if r is None:
            continue
        rows.append(r)
        o = r["onsets"]
        print(f"{r['session']:38s} {r['n_frames']:5d} fr "
              f"{r['nominal_fps']:5.2f} fps  "
              f"frame-time err med {r['frame_time_err_ms']['median_abs']:7.1f} "
              f"max {r['frame_time_err_ms']['max_abs']:8.1f} ms"
              + ("" if o is None else
                 f"   IOI err med {o['ioi_err_ms']['median_abs']:6.2f} "
                 f"p95 {o['ioi_err_ms']['p95_abs']:7.2f} ms  "
                 f"pause flips {o['pause_flips']:2d}/{o['n']-1}"))

    withn = [r for r in rows if r["onsets"]]
    if not withn:
        raise SystemExit("no session had both frames.jsonl and a prepared "
                         "stream with onsets")

    ioi_med = np.array([r["onsets"]["ioi_err_ms"]["median_abs"] for r in withn])
    ioi_p95 = np.array([r["onsets"]["ioi_err_ms"]["p95_abs"] for r in withn])
    ioi_max = np.array([r["onsets"]["ioi_err_ms"]["max_abs"] for r in withn])
    frame_max = np.array([r["frame_time_err_ms"]["max_abs"] for r in rows])
    flips = np.array([r["onsets"]["pause_flips"] for r in withn])
    n_ioi = np.array([r["onsets"]["n"] - 1 for r in withn])
    hes = np.array([r["onsets"]["hesitation_err_pct"] or 0.0 for r in withn])
    tps = np.array([r["onsets"]["execution_tps_err_pct"] or 0.0 for r in withn])

    print(f"\n{'=' * 74}")
    print(f"  {len(rows)} sessions, {len(withn)} with prepared onsets "
          f"({int(n_ioi.sum())} intervals)")
    print(f"{'=' * 74}")
    print(f"\n  ABSOLUTE onset time (nominal - real), the number that looks bad:")
    print(f"    worst per-frame drift over a session: median "
          f"{np.median(frame_max):7.1f} ms, worst {frame_max.max():.1f} ms")
    print(f"    -> a real quantity, and IRRELEVANT on its own: it is a slow "
          f"drift, so it is\n       almost entirely common to both ends of "
          f"any interval short enough to be a\n       move.")
    print(f"\n  INTERVAL error (nominal - real), the number that decides it:")
    print(f"    median |err|  per session: median {np.median(ioi_med):.2f} ms, "
          f"worst {ioi_med.max():.2f} ms")
    print(f"    p95    |err|  per session: median {np.median(ioi_p95):.2f} ms, "
          f"worst {ioi_p95.max():.2f} ms")
    print(f"    max    |err|  per session: median {np.median(ioi_max):.2f} ms, "
          f"worst {ioi_max.max():.2f} ms")
    print(f"    measured 30fps jitter floor: {JITTER_FLOOR_MS:.0f} ms MAD "
          f"(onset_timing_s*.json)")
    print(f"\n  DOWNSTREAM, on the two shipped scalars:")
    print(f"    hesitation seconds: median {np.median(np.abs(hes)):.3f}% error, "
          f"worst {np.abs(hes).max():.3f}%")
    print(f"    execution TPS:      median {np.median(np.abs(tps)):.3f}% error, "
          f"worst {np.abs(tps).max():.3f}%")
    print(f"    pause-rule flips:   {int(flips.sum())} of {int(n_ioi.sum())} "
          f"intervals ({100*flips.sum()/max(1,n_ioi.sum()):.3f}%), "
          f"worst session {flips.max()}")

    worse = ioi_p95 > JITTER_FLOOR_MS
    print(f"\n  VERDICT: the clock error exceeds the 30fps jitter floor at p95 "
          f"on {int(worse.sum())}/{len(withn)}\n  sessions. ", end="")
    if worse.sum() == 0:
        print(f"It is real but strictly smaller than an error already\n  "
              f"present for a different reason, so switching to real "
              f"timestamps is a\n  correctness fix that will not move any "
              f"published number.")
    else:
        print(f"It is large enough to matter on those sessions, so the\n  "
              f"published metrics carry it.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
