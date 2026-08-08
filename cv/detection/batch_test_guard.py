"""
batch_test_guard.py — run the continuity guard over already-recorded
ble/training_data/ sessions to measure false-DQ rate on known-legit solves.

These sessions were never deliberately swapped, so any "dq" verdict here is
a false positive (the guard being too strict) — this doesn't test whether
the guard catches real swaps, only whether it's too light or too tight on
ordinary solve footage. Real attack coverage still needs live_guard_test.py
sessions with a human actually swapping cubes.

Unlike continuity_guard.py's own analyze_frames_dir (which assumes constant
fps and reconstructs t = idx/fps), this reads the real per-frame timestamps
already recorded in each session's frames.jsonl, since capture rate in
these sessions is not constant and the guard's gap logic is timing-sensitive.

Run from inside cv/detection (bare model filenames — see CLAUDE.md):
    python batch_test_guard.py --root ../../ble/training_data
    python batch_test_guard.py --root ../../ble/training_data --limit 10
"""

import argparse
import json
import os
import time

import cv2

from continuity_guard import ContinuityGuard, box_signature, dedup_boxes, detect_cubes


def load_session(session_dir):
    manifest = os.path.join(session_dir, "frames.jsonl")
    frames_dir = os.path.join(session_dir, "frames")
    if not (os.path.isfile(manifest) and os.path.isdir(frames_dir)):
        return None
    entries = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            path = os.path.join(frames_dir, rec["file"])
            if os.path.isfile(path):
                entries.append((rec["ts"], path))
    entries.sort(key=lambda e: e[0])
    return entries


def run_session(session_dir, stride=1):
    entries = load_session(session_dir)
    if not entries:
        return None
    t0 = entries[0][0]
    first = cv2.imread(entries[0][1])
    if first is None:
        return None
    fh, fw = first.shape[:2]
    guard = ContinuityGuard(fw, fh)
    for i, (ts, path) in enumerate(entries):
        if i % stride:
            continue
        frame = cv2.imread(path)
        if frame is None:
            continue
        t = ts - t0
        boxes = detect_cubes(frame)
        deduped = dedup_boxes(boxes)
        sig = (box_signature(frame, max(deduped, key=lambda b: b[4]))
               if deduped else None)
        guard.update(t, boxes, sig)
    return guard.report()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                     help="dir containing solve_* session subfolders")
    ap.add_argument("--limit", type=int, default=0,
                     help="only test the first N sessions (0 = all)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="guard_batch_results.json")
    args = ap.parse_args()

    sessions = sorted(
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d))
        and os.path.isfile(os.path.join(args.root, d, "frames.jsonl"))
        and os.path.isdir(os.path.join(args.root, d, "frames"))
    )
    if args.limit:
        sessions = sessions[:args.limit]
    if not sessions:
        raise SystemExit(f"no sessions with frames/ + frames.jsonl under {args.root}")

    results = {}
    t_start = time.time()
    for i, name in enumerate(sessions):
        session_dir = os.path.join(args.root, name)
        t0 = time.time()
        rep = run_session(session_dir, stride=args.stride)
        dt = time.time() - t0
        if rep is None:
            print(f"[{i+1}/{len(sessions)}] {name}: SKIP (no loadable frames)")
            continue
        results[name] = rep
        flag = "PASS" if rep["verdict"] == "pass" else "*** DQ ***"
        print(f"[{i+1}/{len(sessions)}] {name}: {flag}  "
              f"reasons={rep['dq_reasons']}  notes={rep['notes']}  "
              f"det_rate={rep['detection_rate']:.2f}  max_gap={rep['max_gap_s']:.2f}s  "
              f"({dt:.1f}s, {rep['frames']} frames)")

    n = len(results)
    n_dq = sum(1 for r in results.values() if r["verdict"] == "dq")
    reason_counts = {}
    for r in results.values():
        for reason in r["dq_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print(f"\n=== {n} sessions tested in {time.time() - t_start:.0f}s ===")
    print(f"false-DQ rate: {n_dq}/{n} = {100 * n_dq / n:.1f}%"
          if n else "no sessions tested")
    if reason_counts:
        print("DQ reason breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {count}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull per-session reports written to {args.out}")


if __name__ == "__main__":
    main()
