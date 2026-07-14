"""
postprocess_session.py

Aligns BLE move timestamps to webcam frame timestamps after recording.
Run this after record_training.py finishes a session.

Usage:
    python postprocess_session.py --session training_data/solve_20260708_173218
    python postprocess_session.py --session training_data/solve_20260708_173218 --dry-run

Output added to the session folder:
    labeled/
        move_0001_before.jpg     STILL  (T - 150ms)
        move_0001_mid_00.jpg     MOVING (T + 30ms)
        move_0001_mid_01.jpg     MOVING (T + 60ms)
        move_0001_mid_02.jpg     MOVING (T + 100ms)
        move_0001_after.jpg      STILL  (T + 250ms)
    moves_labeled.jsonl
    metadata.json
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Time offsets relative to BLE event timestamp T
WINDOWS = {
    "before": -0.150,
    "mid_00": +0.030,
    "mid_01": +0.060,
    "mid_02": +0.100,
    "after":  +0.250,
}

MOTION_LABELS = {
    "before": "STILL",
    "mid_00": "MOVING",
    "mid_01": "MOVING",
    "mid_02": "MOVING",
    "after":  "STILL",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_nearest(frame_times: list[float], frame_recs: list[dict],
                 target: float, max_gap: float = 0.5) -> dict | None:
    if not frame_times:
        return None
    lo, hi = 0, len(frame_times) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if frame_times[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    best = lo
    if lo > 0 and abs(frame_times[lo-1]-target) < abs(frame_times[lo]-target):
        best = lo - 1
    return frame_recs[best] if abs(frame_times[best]-target) <= max_gap else None


def postprocess(session_dir: Path, dry_run: bool = False):
    # Check required files exist
    for req in ["moves.jsonl", "frames.jsonl", "config.json"]:
        if not (session_dir / req).exists():
            print(f"ERROR: {req} not found in {session_dir}")
            print("Run record_training.py first.")
            sys.exit(1)

    config     = json.loads((session_dir / "config.json").read_text())
    ble_meta   = {}
    if (session_dir / "ble_meta.json").exists():
        ble_meta = json.loads((session_dir / "ble_meta.json").read_text())

    moves      = load_jsonl(session_dir / "moves.jsonl")
    frame_recs = load_jsonl(session_dir / "frames.jsonl")
    frames_dir = session_dir / "frames"
    labeled_dir= session_dir / "labeled"

    frame_times = [r["ts"] for r in frame_recs]

    print(f"\n  Session:  {session_dir.name}")
    print(f"  Moves:    {len(moves)}")
    print(f"  Frames:   {len(frame_recs)}")
    if len(frame_times) >= 2:
        dur = frame_times[-1] - frame_times[0]
        fps = len(frame_recs) / dur if dur > 0 else 0
        print(f"  Duration: {dur:.1f}s  ({fps:.1f} fps avg)")
    print()

    if not dry_run:
        labeled_dir.mkdir(exist_ok=True)

    labeled_moves = []
    missing       = 0
    total_frames  = 0
    still_count   = 0
    moving_count  = 0

    for move in moves:
        t        = move["timestamp"]
        move_num = move["move_num"]
        label    = f"move_{move_num:04d}"
        frame_paths = {}

        for win_name, offset in WINDOWS.items():
            nearest = find_nearest(frame_times, frame_recs, t + offset)
            if nearest is None:
                frame_paths[win_name] = None
                missing += 1
                if dry_run:
                    print(f"  [{label}] {win_name:<8}  NO FRAME FOUND "
                          f"(target t+{offset*1000:.0f}ms)")
                continue

            src = frames_dir / nearest["file"]
            if not src.exists():
                frame_paths[win_name] = None
                missing += 1
                continue

            rel_path = f"labeled/{label}_{win_name}.jpg"
            gap_ms = abs(nearest["ts"] - (t + offset)) * 1000

            if dry_run:
                print(f"  [{label}] {win_name:<8}  gap={gap_ms:5.1f}ms"
                      f"  [{MOTION_LABELS[win_name]}]"
                      f"  ← {nearest['file']}")
            else:
                shutil.copy2(src, labeled_dir / f"{label}_{win_name}.jpg")

            frame_paths[win_name] = rel_path
            total_frames += 1
            if MOTION_LABELS[win_name] == "STILL":
                still_count  += 1
            else:
                moving_count += 1

        labeled_moves.append({
            **move,
            "frames":        frame_paths,
            "motion_labels": MOTION_LABELS,
        })

    if not dry_run:
        with open(session_dir / "moves_labeled.jsonl", "w") as f:
            for m in labeled_moves:
                f.write(json.dumps(m) + "\n")

        metadata = {
            **config,
            **ble_meta,
            "move_count":     len(labeled_moves),
            "total_frames":   total_frames,
            "still_frames":   still_count,
            "moving_frames":  moving_count,
            "missing_windows":missing,
            "wca_sequence":   [m["wca_notation"] for m in moves
                               if m.get("wca_notation")],
        }
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"  Labeled:  {len(labeled_moves)} moves  "
          f"({total_frames} frames: {still_count} STILL, {moving_count} MOVING)")
    if missing:
        print(f"  Missing:  {missing} windows (frame capture gap or timing)")
    else:
        print(f"  Coverage: 100%")

    if not dry_run:
        print(f"\n  → {session_dir}/moves_labeled.jsonl")
        print(f"  → {labeled_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, required=True,
                        help="Session folder from record_training.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview alignment without writing files")
    args = parser.parse_args()

    d = Path(args.session)
    if not d.exists():
        print(f"ERROR: {d} not found")
        sys.exit(1)

    postprocess(d, dry_run=args.dry_run)