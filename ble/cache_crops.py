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

Run this before EVERY classifier retrain, not once at project start. It is
idempotent (skips sessions that already have a crops.json), and skipping it
is not a no-op: `train_move_classifier.py` falls back to full frames for any
session without one, while every inference path always crops. That mismatch
— not data volume — is what held the live-regime classifier back until
2026-07-24. `--check` is the preflight that answers "is the dataset ready"
without recomputing anything.

Usage:
    python cache_crops.py --sessions training_data/solve_*/
    python cache_crops.py --sessions training_data/solve_*/ --force
    python cache_crops.py --sessions training_data/solve_*/ --check
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from crop_utils import (load_detector, detect_box, median_box,
                        square_with_margin, CROP_MARGIN)

STREAM_FILE = "detector_stream.npz"


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


def check_session(session_dir: Path) -> tuple[str, str]:
    """
    Audit one session's crop readiness without recomputing anything.

    Returns (status, detail). Checks BOTH artifacts that carry a crop, since
    the classifier and the onset detector read different ones:
      crops.json           per-move boxes, read by train_move_classifier.py
      detector_stream.npz  pre-cropped grey stream, read by train.py

    The failure this exists to catch is not "no crops.json" — that one at
    least shows up as a count in the training header. It is a crops.json
    that is STALE: written before postprocess_session.py added more moves,
    or cached under a different CROP_MARGIN than the live crop now uses.
    Both leave a session that reports as cached while feeding the model
    something inference never produces.
    """
    labeled = session_dir / "moves_labeled.jsonl"
    crops   = session_dir / "crops.json"
    stream  = session_dir / STREAM_FILE

    notes = []
    if not labeled.exists():
        cls = "no labels"
        notes.append("run postprocess_session.py")
    elif not crops.exists():
        cls = "MISSING"
        notes.append("run cache_crops.py")
    else:
        data  = json.loads(crops.read_text())
        boxes = data.get("boxes", {})
        moves = [json.loads(l) for l in open(labeled) if l.strip()]
        absent = [m["move_num"] for m in moves
                  if f"move_{m['move_num']:04d}" not in boxes]
        margin = float(data.get("margin", -1))
        if absent:
            cls = "STALE"
            notes.append(f"{len(absent)} of {len(moves)} moves have no box "
                         f"(crops.json predates moves_labeled.jsonl) — "
                         f"--force")
        elif abs(margin - CROP_MARGIN) > 1e-9:
            cls = "STALE"
            notes.append(f"margin {margin} != live {CROP_MARGIN} — --force")
        else:
            cls = "ok"
            det = data.get("detected", 0)
            tot = data.get("total", len(moves))
            if tot and det / tot < 0.9:
                notes.append(f"only {det}/{tot} moves detected directly, "
                             f"rest inherited from neighbours")

    if not stream.exists():
        det_mode = "-"
    else:
        import numpy as np
        d = np.load(stream, allow_pickle=True)
        det_mode = str(d["crop_mode"]) if "crop_mode" in d else "unknown"
        if det_mode not in ("cropped", "-"):
            notes.append(f"detector stream is '{det_mode}' — "
                         f"prepare_data.py --force")

    return f"{cls:<9} {det_mode:<13}", "; ".join(notes)


def run_check(session_dirs: list[Path]) -> int:
    print(f"\n{'session':<34} {'classifier':<9} {'detector':<13} notes")
    print(f"{'-'*34} {'-'*9} {'-'*13} {'-'*30}")
    bad = 0
    for d in sorted(session_dirs):
        status, notes = check_session(d)
        if not status.startswith("ok") or "prepare_data" in notes:
            bad += 1
        print(f"{d.name:<34} {status} {notes}")
    print()
    if bad:
        print(f"{bad} of {len(session_dirs)} session(s) are not crop-ready. "
              f"Training and evaluation\nboth refuse a mixed regime, so fix "
              f"these before quoting any accuracy number.")
    else:
        print(f"All {len(session_dirs)} session(s) crop-ready.")
    return 1 if bad else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache per-move cube crop boxes for training")
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="Session folder(s) — supports globs")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if crops.json already exists")
    parser.add_argument("--check", action="store_true",
                        help="Audit crop readiness (classifier crops.json and "
                             "detector stream) and exit non-zero if anything "
                             "is missing or stale. Computes nothing — run it "
                             "before a retrain or before quoting a number")
    args = parser.parse_args()

    session_dirs = [Path(p) for pattern in args.sessions
                    for p in (Path(".").glob(pattern)
                              if "*" in pattern else [Path(pattern)])]
    session_dirs = [d for d in session_dirs if d.is_dir()]
    if not session_dirs:
        sys.exit("No session directories found.")

    if args.check:
        sys.exit(run_check(session_dirs))

    model = load_detector()
    if model is None:
        sys.exit("Cube detector unavailable (ultralytics not installed or "
                 "cv/detection/detect_full_cube.pt missing).")

    print(f"\nCaching crop boxes for {len(session_dirs)} session(s)...")
    for d in sorted(session_dirs):
        cache_session(d, model, force=args.force)
    print("Done.")
