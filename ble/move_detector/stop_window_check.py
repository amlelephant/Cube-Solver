"""
stop_window_check.py — does solved_check's 0-false-DQ result reach the LIVE
call? Two things change between the bench and the product, and this measures
both, separately.

WHAT WAS CALIBRATED
-------------------
`cv/detection/solved_check.py score` measures 0 false DQ on 28 legit solves,
and it is a real result — re-run fresh 2026-08-10, worst legit solve lands at
exactly `n_regions = 4.000` against a threshold of "strictly above 4". Note
how thin that is: the operating point is chosen as one step above the worst
observed legit session, so there is by construction **no margin at all** on
the false-DQ side. Anything that moves the statistic even slightly costs
false DQs immediately.

Two things move it, and neither is the lighting the module docstring warns
about:

  1. THE WINDOW. The calibration reads `tail_window`: 1.5 s ending 0.5 s
     before the last frame — seconds after the solve, cube held still and
     often deliberately presented. The live gate
     (`live_anticheat.stop_window_solved`, `verify_solve._solved_at_stop`)
     reads the frames STRADDLING THE TIMER STOP: the instant the last move
     finishes, hands still leaving the cube, cube at whatever angle it
     ended at.

  2. THE BOX SOURCE, which is the one nobody was looking at. The
     calibration takes boxes from `trajectory.npz` — raw
     `continuity_guard.detect_cubes` output. The live path takes them from
     `prepare_data.per_frame_boxes`, which squares the box and adds
     `crop_utils.CROP_MARGIN = 0.12`. A 12% margin puts strictly more
     BACKGROUND inside the crop, and background classifying as a solid
     region is solved_check's own documented failure mode. The threshold is
     therefore coupled to the box source, and nothing in either file says so.

So this runs the full 2x2 — {calibration tail, timer stop} x {trajectory
boxes, per-frame boxes} — over the same sessions with the same `solved_at`
and the same palette. Each cell differs from the calibration in exactly one
respect, so the cost of each factor is attributable rather than confounded.

    python stop_window_check.py
    python stop_window_check.py --limit 8 --out results/2026-08-10/stop_window.json

WHAT A DIFFERENCE MEANS
-----------------------
A false DQ here is worse than a miss. The gate's whole claim is 0 false DQ on
legitimate solves. If a factor costs that, the fix is to make the live call
match the calibration — or to re-calibrate on what the live call actually
sees — NOT to raise SOLVED_MAX_REGIONS, which hands the attack back exactly
the margin this test exists to take away.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_BLE_DIR = _HERE.parents[1]
_DETECTION_DIR = _HERE.parents[2] / "cv" / "detection"
for _p in (str(_BLE_DIR), str(_DETECTION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                                   # noqa: E402
from crop_utils import load_detector                         # noqa: E402
from prepare_data import per_frame_boxes                     # noqa: E402
from solved_check import SOLVED_MAX_REGIONS, solved_at       # noqa: E402

SESSION_ROOT = Path("../training_data")

#: The live gate's window, from live_anticheat.STOP_WINDOW_FRAMES: 20 frames
#: either side of the stop, ~1.3 s total at 30 fps.
STOP_WINDOW_FRAMES = 20
#: Seconds after the last true onset at which the timer is taken to stop. A
#: human presses the timer within a few hundred ms of the last move; 1.0 s is
#: generous in the direction that HELPS the test (more time for hands to
#: clear), so it does not manufacture the failure it is looking for.
STOP_GUARD_S = 1.0

#: solved_check.tail_window's shipped defaults — the window the 0-false-DQ
#: number was measured on.
TAIL_WINDOW_S = 1.5
TAIL_SKIP_LAST_S = 0.5


def frame_records(d: Path):
    recs = [json.loads(l) for l in open(d / "frames.jsonl") if l.strip()]
    recs = [r for r in recs if (d / "frames" / r["file"]).exists()]
    return recs


def trajectory_boxes(d: Path) -> dict[int, tuple] | None:
    """frame index -> raw guard box, from trajectory.npz.

    The calibration's box source. Highest-confidence box per frame, matching
    solved_check._load_session exactly — a different tie-break would be a
    third uncontrolled variable in a file whose whole job is controlling
    them.
    """
    p = d / "trajectory.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    by_frame: dict[int, tuple] = {}
    for fi, b in zip(z["frame_idx"], z["boxes"]):
        fi = int(fi)
        prev = by_frame.get(fi)
        if prev is None or b[4] > prev[4]:
            by_frame[fi] = tuple(float(x) for x in b)
    return by_frame or None


def _solved_over(d: Path, recs, lo: int, hi: int, detector,
                 traj: dict | None, source: str) -> dict:
    """solved_at over frames [lo, hi) of one session, with a chosen box source.

    `source` is "perframe" (crop_utils detect + square + CROP_MARGIN, what
    the live path feeds) or "traj" (raw guard boxes, what the calibration
    fed). Everything else is identical between the two.
    """
    lo, hi = max(0, lo), min(len(recs), hi)
    if hi - lo < 8:
        return {"solved": None, "reason": "too_few_frames", "n_regions": None,
                "n_frames": hi - lo}

    if source == "traj":
        if traj is None:
            return {"solved": None, "reason": "no_trajectory",
                    "n_regions": None, "n_frames": 0}
        pairs = []
        for i in range(lo, hi):
            box = traj.get(i)
            if box is None:
                continue
            f = cv2.imread(str(d / "frames" / recs[i]["file"]))
            if f is not None:
                pairs.append((f, box))
        if len(pairs) < 8:
            return {"solved": None, "reason": "unreadable",
                    "n_regions": None, "n_frames": len(pairs)}
        return solved_at(pairs)

    frames = [cv2.imread(str(d / "frames" / recs[i]["file"]))
              for i in range(lo, hi)]
    frames = [f for f in frames if f is not None]
    n = len(frames)
    if n < 8:
        return {"solved": None, "reason": "unreadable", "n_regions": None,
                "n_frames": n}
    boxes, _ = per_frame_boxes(detector,
                               lambda i: frames[i] if 0 <= i < n else None, n)
    return solved_at(list(zip(frames, boxes)))


#: The 2x2. Key -> (window, box source).
CELLS = {
    "tail_traj": ("tail", "traj"),        # the calibration itself
    "tail_perframe": ("tail", "perframe"),   # box source changed
    "stop_traj": ("stop", "traj"),        # window changed
    "stop_perframe": ("stop", "perframe"),   # both — what the live gate does
}


def check(d: Path, detector) -> dict | None:
    recs = frame_records(d)
    if len(recs) < 40:
        return None
    ts = np.array([r["ts"] for r in recs], dtype=np.float64)
    fps = (len(recs) - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else 30.0

    npz = d / "detector_stream_color.npz"
    if not npz.is_file():
        return None
    onsets = np.load(npz, allow_pickle=True)["onset_idx"].astype(int)
    if onsets.size == 0:
        return None

    stop = int(onsets.max()) + int(round(STOP_GUARD_S * fps))
    stop = int(np.clip(stop, STOP_WINDOW_FRAMES, len(recs) - 1))

    hi_t = ts[-1] - TAIL_SKIP_LAST_S
    lo_t = hi_t - TAIL_WINDOW_S
    bounds = {
        "stop": (stop - STOP_WINDOW_FRAMES, stop + STOP_WINDOW_FRAMES),
        "tail": (int(np.searchsorted(ts, lo_t)),
                 int(np.searchsorted(ts, hi_t))),
    }
    traj = trajectory_boxes(d)

    out = {
        "session": d.name,
        "truth": ("solved" if d.name.endswith("_solve") else
                  "scrambled" if d.name.endswith("_scramble") else None),
        "n_frames": len(recs),
        "fps": round(fps, 2),
        "stop_frame": stop,
        "seconds_from_stop_to_end": round(float(ts[-1] - ts[stop]), 2),
        "has_trajectory": traj is not None,
    }
    for key, (win, src) in CELLS.items():
        lo, hi = bounds[win]
        out[key] = _solved_over(d, recs, lo, hi, detector, traj, src)
    return out


def _verdict(row_key: str, r: dict) -> str:
    s = r[row_key]["solved"]
    return "solved" if s is True else "NOT" if s is False else "unreadable"


def summarise(rows: list[dict], key: str, label: str) -> dict:
    solved = [r for r in rows if r["truth"] == "solved"]
    scram = [r for r in rows if r["truth"] == "scrambled"]
    false_dq = [r for r in solved if r[key]["solved"] is False]
    abstain = [r for r in solved if r[key]["solved"] is None]
    caught = [r for r in scram if r[key]["solved"] is False]
    missed = [r for r in scram if r[key]["solved"] is True]

    print(f"\n  {label}")
    print(f"    legit solves : {len(solved) - len(false_dq) - len(abstain)}"
          f"/{len(solved)} pass, {len(false_dq)} FALSE DQ, "
          f"{len(abstain)} unreadable")
    if false_dq:
        for r in false_dq:
            print(f"       FALSE DQ  {r['session']:38s} "
                  f"{r[key]['n_regions']:.1f} regions > {SOLVED_MAX_REGIONS}")
    if scram:
        print(f"    scrambled    : {len(caught)}/{len(scram)} caught, "
              f"{len(missed)} missed")
    reg = [r[key]["n_regions"] for r in solved
           if r[key]["n_regions"] is not None]
    if reg:
        print(f"    n_regions on legit solves: median {np.median(reg):.1f}, "
              f"max {max(reg):.1f}  (threshold {SOLVED_MAX_REGIONS})")
    return {"n_solved": len(solved), "false_dq": len(false_dq),
            "abstain": len(abstain), "n_scrambled": len(scram),
            "caught": len(caught), "missed": len(missed),
            "regions_median": float(np.median(reg)) if reg else None,
            "regions_max": float(max(reg)) if reg else None}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    detector = load_detector()
    if detector is None:
        raise SystemExit("no cube detector — install ultralytics and put "
                         "detect_full_cube.pt in cv/detection/")

    dirs = sorted(d for d in SESSION_ROOT.iterdir()
                  if d.is_dir() and (d / "frames.jsonl").is_file())
    if args.limit:
        dirs = dirs[:args.limit]

    rows = []
    for i, d in enumerate(dirs):
        r = check(d, detector)
        if r is None:
            continue
        rows.append(r)
        cells = "  ".join(f"{k.split('_')[0][:4]}/{k.split('_')[1][:4]}="
                          f"{_verdict(k, r)[:3]}({r[k]['n_regions'] or 0:.0f})"
                          for k in CELLS)
        print(f"[{i+1}/{len(dirs)}] {d.name:38s} {cells}")

    if not rows:
        raise SystemExit("nothing scored")

    # Only sessions where every cell is computable can be compared; a session
    # missing trajectory.npz would otherwise make the traj columns describe a
    # different population than the perframe ones, which is the confound this
    # file exists to avoid.
    rows = [r for r in rows
            if all(r[k]["reason"] != "no_trajectory" for k in CELLS)]

    print(f"\n{'=' * 74}")
    print(f"  THE 2x2 — {len(rows)} sessions, same solved_at, same palette")
    print(f"  threshold: n_regions <= {SOLVED_MAX_REGIONS}")
    print(f"{'=' * 74}")
    summ = {
        "tail_traj": summarise(rows, "tail_traj",
                               "[A] CALIBRATION: tail window + trajectory "
                               "boxes  (the published 0-false-DQ cell)"),
        "tail_perframe": summarise(rows, "tail_perframe",
                                   "[B] box source changed: tail window + "
                                   "per-frame boxes (+12% margin)"),
        "stop_traj": summarise(rows, "stop_traj",
                               "[C] window changed: timer stop + trajectory "
                               "boxes"),
        "stop_perframe": summarise(rows, "stop_perframe",
                                   "[D] WHAT THE LIVE GATE DOES: timer stop "
                                   "+ per-frame boxes"),
    }

    print(f"\n{'=' * 74}")
    print(f"  ATTRIBUTION — false DQ on legit solves, out of "
          f"{summ['tail_traj']['n_solved']}")
    print(f"{'=' * 74}")
    print(f"    [A] calibration                        "
          f"{summ['tail_traj']['false_dq']}")
    print(f"    [B] + per-frame boxes (box source)     "
          f"{summ['tail_perframe']['false_dq']}")
    print(f"    [C] + timer-stop window (window)       "
          f"{summ['stop_traj']['false_dq']}")
    print(f"    [D] both, i.e. the live gate           "
          f"{summ['stop_perframe']['false_dq']}")
    d_box = summ["tail_perframe"]["false_dq"] - summ["tail_traj"]["false_dq"]
    d_win = summ["stop_traj"]["false_dq"] - summ["tail_traj"]["false_dq"]
    print(f"\n    box source costs {d_box:+d} false DQ; "
          f"window position costs {d_win:+d}.")

    # The threshold is not fixed by nature — so the fair question is not
    # "how many false DQ at 4" but "what is the catch rate once each cell is
    # retuned to zero false DQ". That is the operating curve, and it is what
    # says whether the test survives the move at all or merely needs a
    # different number.
    print(f"\n{'=' * 74}")
    print(f"  RETUNED — each cell at ITS OWN zero-false-DQ operating point")
    print(f"{'=' * 74}")
    print(f"    {'cell':<16}{'thr >':>7}{'false DQ':>10}{'catch':>12}")
    for k in CELLS:
        sol = [r[k]["n_regions"] for r in rows
               if r["truth"] == "solved" and r[k]["n_regions"] is not None]
        scr = [r[k]["n_regions"] for r in rows
               if r["truth"] == "scrambled" and r[k]["n_regions"] is not None]
        if not sol or not scr:
            continue
        thr = max(sol)
        caught = sum(1 for s in scr if s > thr)
        print(f"    {k:<16}{thr:>7.1f}{0:>10d}"
              f"{caught:>7d}/{len(scr):<4d} ({100*caught/len(scr):.0f}%)")
    print(f"\n  Read the catch column, not the false-DQ one — every row is "
          f"retuned to zero\n  false DQ, so the catch rate is the whole "
          f"price of moving the test to where\n  the live gate looks.")
    if summ["stop_perframe"]["false_dq"] > 0:
        print(f"\n  The live gate false-DQs {summ['stop_perframe']['false_dq']}"
              f" legitimate solves. That is not a\n  tuning problem: the "
              f"operating point is one step above the worst legit\n  session "
              f"BY CONSTRUCTION, so it has no false-DQ margin to spend and "
              f"any\n  change of input consumes it. Either make the live call "
              f"feed what was\n  calibrated, or re-calibrate on what the live "
              f"call feeds — but do not raise\n  SOLVED_MAX_REGIONS, which "
              f"returns to the attack the margin this test exists\n  to take "
              f"away.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"stop_guard_s": STOP_GUARD_S,
             "stop_window_frames": STOP_WINDOW_FRAMES,
             "tail_window_s": TAIL_WINDOW_S,
             "tail_skip_last_s": TAIL_SKIP_LAST_S,
             "threshold": SOLVED_MAX_REGIONS,
             "cells": summ, "rows": rows},
            indent=2, default=str))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
