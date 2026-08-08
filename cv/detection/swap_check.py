"""
swap_check.py — deterministic cube-substitution detector.

Premise (2026-08-04, from the threat-model review): to end a solve with a
solved cube on camera you either solved it or substituted a different cube.
Everything else physical reduces to that — including solving it out of view,
which returns a cube that looks different. So the anticheat's core question
is not "did the cube move oddly" but "was this the same cube the whole
time". That is an appearance question, and it needs no ML if legit and
substituted cubes separate cleanly.

The statistic: PERSISTENT appearance change
-------------------------------------------
Adjacent-frame histogram distance (what mark_swap.py ranks by) is the wrong
detector, because a solve changes appearance constantly: every turn
recolours a face, and motion blur spikes the histogram for a frame or two.
Those changes are TRANSIENT or SMALL. A substitution is LARGE and
PERSISTENT — the cube is different before and after, and stays different.

So at each candidate instant this compares a stable window BEFORE against a
stable window AFTER, using the per-channel median signature of each window
(median, so a couple of blurred frames cannot move it):

    jump(t) = bhattacharyya( median(sig[t-W..t)), median(sig[t..t+W)) )

and a session scores as its maximum over t. Blur-robust by construction, and
sensitive to exactly the thing being defended against.

WHAT THIS FILE CANNOT ESTABLISH ALONE: `baseline` measures only legit
sessions, so it bounds FALSE POSITIVES. Whether real substitutions exceed
that bound is a separate measurement needing recorded attacks
(cv/labeling/record_attack.py). A statistic can separate perfectly on legit
data and still fail to detect. Run `score` on attack sessions before
believing any threshold.

  cache     per-frame appearance signature of the tracked box -> signatures.npz
            (one-time per session; the JPEG decode is the expensive part)
  baseline  distribution of the statistic over legit sessions -> threshold
  score     score sessions, reporting AT-SWAP when attack.json has marks

Run from inside cv/detection (see CLAUDE.md). Needs trajectory.npz first:
    python dump_trajectories.py --root ../../ble/training_data
    python swap_check.py cache    --root ../../ble/training_data
    python swap_check.py baseline --root ../../ble/training_data
    python swap_check.py score    --root ../labeling/attack_sessions
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from continuity_guard import box_signature, dedup_boxes, pick_continuity, _sig_dist

SIG_FILE  = "signatures.npz"
WINDOW_S  = 0.40   # stable window compared before vs after. Long enough to
                   #   median away motion blur, short enough that a fast
                   #   substitution still has clean cube on both sides.
MIN_SIDE  = 4      # need this many signatures per side to trust a comparison
GUARD_S   = 0.10   # skip this much either side of the split: the frames
                   #   immediately around a substitution are mid-transition
                   #   and belong to neither cube.


def _load_traj(session_dir):
    p = os.path.join(session_dir, "trajectory.npz")
    if not os.path.isfile(p):
        return None
    return np.load(p, allow_pickle=True)


def _frame_paths(session_dir, n_frames):
    entries = []
    with open(os.path.join(session_dir, "frames.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda r: r["ts"])
    return [os.path.join(session_dir, "frames", r["file"]) for r in entries][:n_frames]


def cache_session(session_dir):
    """Per-frame signature of the tracked box. Absent frames get NaN rows."""
    d = _load_traj(session_dir)
    if d is None:
        return None
    n_frames = int(d["n_frames"])
    by_frame = {}
    for fi, t, b in zip(d["frame_idx"], d["t"], d["boxes"]):
        by_frame.setdefault(int(fi), {"t": float(t), "boxes": []})["boxes"].append(tuple(b))
    paths = _frame_paths(session_dir, n_frames)

    sigs, times, ref = [], [], None
    dim = None
    for i in range(len(paths)):
        rec = by_frame.get(i)
        if rec is None:
            sigs.append(None)
            times.append(np.nan)
            continue
        boxes = dedup_boxes(rec["boxes"])
        if not boxes:
            sigs.append(None)
            times.append(rec["t"])
            continue
        best = pick_continuity(boxes, ref)
        ref = best
        frame = cv2.imread(paths[i])
        s = box_signature(frame, best) if frame is not None else None
        if s is not None and dim is None:
            dim = len(s)
        sigs.append(s)
        times.append(rec["t"])

    if dim is None:
        return None
    arr = np.full((len(sigs), dim), np.nan, dtype=np.float32)
    for i, s in enumerate(sigs):
        if s is not None:
            arr[i] = s
    np.savez_compressed(os.path.join(session_dir, SIG_FILE),
                        sigs=arr, t=np.asarray(times, dtype=np.float64))
    return int(np.sum(~np.isnan(arr[:, 0])))


def session_jump(session_dir):
    """(max_jump, t_of_max) using the persistent-change statistic."""
    p = os.path.join(session_dir, SIG_FILE)
    if not os.path.isfile(p):
        return None
    d = np.load(p)
    sigs, t = d["sigs"], d["t"]
    ok = ~np.isnan(sigs[:, 0]) & ~np.isnan(t)
    if ok.sum() < 2 * MIN_SIDE:
        return None
    S, T = sigs[ok], t[ok]

    best, best_t = 0.0, None
    # candidate split at every detected frame; windows are in TIME, so a
    # variable capture rate cannot distort them
    for k in range(len(T)):
        t_k = T[k]
        pre = (T >= t_k - GUARD_S - WINDOW_S) & (T <= t_k - GUARD_S)
        post = (T >= t_k + GUARD_S) & (T <= t_k + GUARD_S + WINDOW_S)
        if pre.sum() < MIN_SIDE or post.sum() < MIN_SIDE:
            continue
        a = np.median(S[pre], axis=0)
        b = np.median(S[post], axis=0)
        dist = _sig_dist(a, b)
        if dist > best:
            best, best_t = dist, float(t_k)
    return best, best_t


def discover(root, need):
    return sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, need)))


def cmd_cache(args):
    sessions = discover(args.root, "trajectory.npz")
    if not sessions:
        raise SystemExit(f"no sessions with trajectory.npz under {args.root} "
                         "— run dump_trajectories.py first")
    t0 = time.time()
    for i, n in enumerate(sessions):
        sdir = os.path.join(args.root, n)
        if not args.force and os.path.isfile(os.path.join(sdir, SIG_FILE)):
            print(f"[{i+1}/{len(sessions)}] {n}: cached, skipping")
            continue
        k = cache_session(sdir)
        print(f"[{i+1}/{len(sessions)}] {n}: {k if k else 0} signatures")
    print(f"\n{time.time() - t0:.0f}s")


def cmd_baseline(args):
    sessions = discover(args.root, SIG_FILE)
    if not sessions:
        raise SystemExit(f"no cached signatures under {args.root} — run `cache`")
    rows = []
    for n in sessions:
        r = session_jump(os.path.join(args.root, n))
        if r:
            rows.append((r[0], r[1], n))
    rows.sort(reverse=True)

    vals = np.array([r[0] for r in rows])
    print(f"persistent appearance jump, {len(rows)} LEGIT sessions "
          f"(window {WINDOW_S}s, guard {GUARD_S}s)\n")
    print("  worst 12 sessions (these set the false-positive floor):")
    for v, t, n in rows[:12]:
        print(f"    {v:.4f}  t={t:7.1f}s  {n}")
    print(f"\n  median {np.median(vals):.4f}   p90 {np.percentile(vals,90):.4f}   "
          f"max {vals.max():.4f}")

    thr = float(vals.max())
    print(f"\n  legit maximum = {thr:.4f}")
    print(f"  any threshold at or below this WILL false-DQ a legit solve.")
    print(f"  suggested operating point: {thr * 1.1:.4f} (legit max +10%)")
    np.savez(os.path.join(args.root, "..", "swap_check_baseline.npz"),
             legit_max=np.float64(thr), values=vals,
             window_s=np.float64(WINDOW_S), guard_s=np.float64(GUARD_S))
    print(f"\n  This bounds FALSE POSITIVES only. It does not show that a real")
    print(f"  substitution exceeds it — record attacks and run `score`.")


def cmd_score(args):
    sessions = discover(args.root, SIG_FILE)
    if not sessions:
        raise SystemExit(f"no cached signatures under {args.root} — run `cache`")
    thr = args.threshold
    n_att = n_att_hit = n_att_at = n_legit = n_legit_flag = 0
    for n in sessions:
        sdir = os.path.join(args.root, n)
        r = session_jump(sdir)
        if not r:
            print(f"{n}: not scoreable")
            continue
        val, t = r
        flag = thr is not None and val > thr
        meta = {}
        mp = os.path.join(sdir, "attack.json")
        if os.path.isfile(mp):
            meta = json.loads(open(mp).read())
        tag = ""
        if meta:
            if meta.get("is_attack"):
                n_att += 1
                n_att_hit += flag
                marks = meta.get("marks") or []
                if marks:
                    near = min(abs(t - m["t"]) for m in marks)
                    at = near <= WINDOW_S + GUARD_S
                    n_att_at += bool(flag and at)
                    tag = (f"  [{meta.get('attack_type')}] "
                           + (f"AT-SWAP (+-{near:.2f}s)" if at
                              else f"NOT at swap ({near:.1f}s away)"))
                else:
                    tag = f"  [{meta.get('attack_type')}] (no marks)"
            else:
                n_legit += 1
                n_legit_flag += flag
                tag = "  [legit control]"
        state = ("FLAG" if flag else "ok  ") if thr is not None else "    "
        print(f"{state}  {val:.4f}  t={t:7.1f}s  {n}{tag}")
    if thr is not None:
        print(f"\nthreshold {thr:.4f}")
        if n_att:
            print(f"  attacks flagged      : {n_att_hit}/{n_att}")
            print(f"  ...and AT the swap   : {n_att_at}/{n_att}")
        if n_legit:
            print(f"  legit controls flagged: {n_legit_flag}/{n_legit}")
    else:
        print("\n(no --threshold given: scores only. Use the value from "
              "`baseline`.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache");    c.add_argument("--root", required=True)
    c.add_argument("--force", action="store_true")
    b = sub.add_parser("baseline"); b.add_argument("--root", required=True)
    s = sub.add_parser("score");    s.add_argument("--root", required=True)
    s.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    {"cache": cmd_cache, "baseline": cmd_baseline, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
