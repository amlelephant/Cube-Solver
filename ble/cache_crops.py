"""
cache_crops.py

Offline pass: run the cube detector over each session's labeled/ move
snapshots and cache one shared square crop box per move to crops.json in
the session folder. train_move_classifier.py picks the cache up
automatically; sessions without crops.json train on full frames.

A move with no detection in any of its snapshots inherits the box of the
nearest move (by move number) that did resolve — the cube barely
translates between consecutive moves, so a neighbor's box is a far better
prior than the full frame. If nothing in the whole session resolves, no
crops.json is written.

Usage:
    python cache_crops.py --sessions training_data/solve_*/
    python cache_crops.py --sessions training_data/solve_*/ --force
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from crop_utils import (load_detector, detect_box, median_box,
                        square_with_margin, CROP_MARGIN)


def cache_session(session_dir: Path, model, force: bool = False) -> None:
    out_path = session_dir / "crops.json"
    if out_path.exists() and not force:
        print(f"  {session_dir.name}: crops.json exists, skipping (--force to redo)")
        return

    labeled = session_dir / "moves_labeled.jsonl"
    if not labeled.exists():
        print(f"  {session_dir.name}: no moves_labeled.jsonl, skipping")
        return

    moves = [json.loads(l) for l in open(labeled) if l.strip()]

    resolved = {}    # move_num -> raw median box
    shapes   = {}    # move_num -> frame shape
    for m in moves:
        boxes, shape = [], None
        for rel in m.get("frames", {}).values():
            if not rel:
                continue
            img = cv2.imread(str(session_dir / rel))
            if img is None:
                continue
            shape = img.shape
            b = detect_box(model, img)
            if b is not None:
                boxes.append(b)
        if boxes:
            resolved[m["move_num"]] = median_box(boxes)
            shapes[m["move_num"]]   = shape

    if not resolved:
        print(f"  {session_dir.name}: cube never detected — no crops.json written")
        return

    # Fill gaps from the nearest resolved move
    resolved_nums = sorted(resolved)
    crops = {}
    for m in moves:
        num = m["move_num"]
        src = num if num in resolved else \
            min(resolved_nums, key=lambda r: abs(r - num))
        crops[f"move_{num:04d}"] = list(
            square_with_margin(resolved[src], shapes[src]))

    out = {
        "margin":   CROP_MARGIN,
        "detected": len(resolved),
        "total":    len(moves),
        "boxes":    crops,
    }
    with open(out_path, "w") as f:
        json.dump(out, f)

    pct = len(resolved) / len(moves) * 100
    print(f"  {session_dir.name}: {len(resolved)}/{len(moves)} moves "
          f"detected directly ({pct:.0f}%), rest filled from neighbors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache per-move cube crop boxes for training")
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="Session folder(s) — supports globs")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if crops.json already exists")
    args = parser.parse_args()

    session_dirs = [Path(p) for pattern in args.sessions
                    for p in (Path(".").glob(pattern)
                              if "*" in pattern else [Path(pattern)])]
    session_dirs = [d for d in session_dirs if d.is_dir()]
    if not session_dirs:
        sys.exit("No session directories found.")

    model = load_detector()
    if model is None:
        sys.exit("Cube detector unavailable (ultralytics not installed or "
                 "cv/detection/detect_full_cube.pt missing).")

    print(f"\nCaching crop boxes for {len(session_dirs)} session(s)...")
    for d in sorted(session_dirs):
        cache_session(d, model, force=args.force)
    print("Done.")
