"""
backfill_camera_frame.py

Add the camera-relative move name to already-recorded sessions.

Every session recorded before 2026-08-03 carries only `wca_notation`, which
is the move in the CUBE's own frame — and after a middle-slice turn that
frame has rotated away from the camera, so the label names the wrong face.
See `slice_frame.py` for the mechanism and the measured evidence, and
`move_detector/LABEL_FRAME_PLAN.md` for the plan this is step 2 of.

What it writes, per move record:

    camera_notation   the move as the CAMERA saw it — what the vision model
                      must predict, and what prepare_data.py now builds
                      onset_class from
    orientation       the camera<-cube frame in force for that move, so the
                      relabelling is auditable after the fact rather than
                      having to be re-derived

`wca_notation` is left completely alone. It is the cube-frame name and
several consumers depend on that: `session_check.py` applies it to a
centre-relative `CubieCube`, and `reconstruct.start_from_gt` builds the
start state from it. Silently redefining a field is how the original bug
happened; this adds fields instead.

Additive and idempotent — running it twice changes nothing, and it never
removes or rewrites an existing field. On a session with no slices it is a
no-op by construction, which is 69 of the 76 recorded sessions.

After running this, re-prepare the affected sessions so the change reaches
the training stream:

    python move_detector/prepare_data.py --sessions <session> --color --force

Usage (run from inside ble/, per this repo's convention):
    python backfill_camera_frame.py --sessions training_data/solve_*/
    python backfill_camera_frame.py --sessions training_data/solve_*/ --dry-run
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from slice_frame import annotate


def load_moves(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def write_moves(path: Path, moves: list[dict]) -> None:
    """
    Rewrite moves.jsonl atomically.

    `training_data/` is gitignored, so these files are the only copy of the
    ground truth. A half-written moves.jsonl from an interrupted run would
    be unrecoverable, and worse, would look like a short session rather
    than a broken one.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            for m in moves:
                fh.write(json.dumps(m) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def process(session_dir: Path, dry_run: bool) -> dict | None:
    path = session_dir / "moves.jsonl"
    if not path.exists():
        return None
    moves = load_moves(path)
    if not moves:
        return None

    before = [m.get("camera_notation") for m in moves]
    summary = annotate(moves)
    changed = [m.get("camera_notation") for m in moves] != before

    if changed and not dry_run:
        write_moves(path, moves)
    summary["written"] = changed and not dry_run
    summary["already_current"] = not changed
    return summary


def main():
    p = argparse.ArgumentParser(
        description="Add camera-relative move names to recorded sessions")
    p.add_argument("--sessions", nargs="+", required=True,
                   help="session folder(s) — supports globs")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change, write nothing")
    args = p.parse_args()

    dirs = [Path(q) for pattern in args.sessions
            for q in (sorted(Path(".").glob(pattern)) if "*" in pattern
                      else [Path(pattern)])]
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        sys.exit("No session directories found.")

    print(f"\n{'session':<40} {'moves':>6} {'slices':>7} {'relabelled':>11}"
          f"  final frame")
    print("-" * 84)

    n_touched = n_slices = n_relabelled = 0
    for d in sorted(dirs):
        s = process(d, args.dry_run)
        if s is None:
            continue
        if s["n_slices"] == 0 and s["n_relabelled"] == 0:
            continue                       # silent on the 69 unaffected
        n_touched += 1
        n_slices += s["n_slices"]
        n_relabelled += s["n_relabelled"]
        print(f"{d.name:<40} {s['n_moves']:>6} {s['n_slices']:>7} "
              f"{s['n_relabelled']:>11}  {s['final_orientation']}"
              + ("  (already current)" if s["already_current"] else ""))

    total = sum(1 for d in dirs if (d / "moves.jsonl").exists())
    print(f"\n{n_touched} of {total} session(s) contain slices: "
          f"{n_slices} slice pair(s), {n_relabelled} move(s) relabelled.")
    print(f"{total - n_touched} session(s) have no slices and are unchanged "
          f"by construction.")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
    elif n_touched:
        print(f"\nRe-prepare the affected sessions so the change reaches the "
              f"training stream:\n"
              f"    python move_detector/prepare_data.py --sessions "
              f"training_data/<session>/ --color --force")


if __name__ == "__main__":
    main()
