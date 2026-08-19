"""
export_example_session.py — package one recorded session for someone outside
this repo.

`ble/training_data/` is gitignored (large, and the corpus is not source), so
sharing a session with a collaborator means copying it somewhere tracked and
telling them what the files mean. This does that reproducibly instead of by
hand, and adds the two things a recipient cannot derive on their own:

  cube_states.jsonl   the GROUND-TRUTH 54-facelet cube state after every
                      move, with the timestamp it took effect. This is the
                      valuable part. Anything that reads colour off the video
                      — a lattice fit, a sticker classifier, a segmentation —
                      can be scored against it at any instant, without a
                      smart cube and without trusting the recipient's own
                      reader. It is derived from the BLE move log, which is a
                      LABEL captured alongside the video and never an input
                      to any model here.

  boxes.jsonl         the cube's bounding box in every frame, from the
                      fine-tuned YOLO detector, median-smoothed and
                      interpolated across gaps by `prepare_data.per_frame_boxes`
                      — the same crop path the models train and infer on. A
                      recipient without the weights would otherwise have to
                      find the cube themselves before doing anything else.

Everything else copied is the raw recording as `record_training.py` wrote it
(frames, frame index, move log, capture config, cube health snapshot).
Derived model artifacts — `detector_stream_color.npz`, `ctc_post_*.npz`,
`signatures.npz` — are deliberately NOT copied: they are this project's
internal preprocessing, regenerable, and would triple the size while telling
a recipient nothing about the cube.

    python export_example_session.py training_data/solve_20260809_155709_solve \\
        --out "../example solve"

Run from inside ble/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for _p in (str(_HERE.parent), str(_HERE.parent / "move_detector")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Raw recording files copied verbatim. Anything not listed is derived.
COPY_FILES = ("frames.jsonl", "moves.jsonl", "config.json", "ble_meta.json")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def cube_states(session: Path) -> tuple[list[dict], dict]:
    """Ground-truth facelet string after every move.

    The word used is `camera_notation` where the recording has it, falling
    back to `wca_notation`. That choice matters and is not cosmetic: the
    smart cube names moves in ITS own frame (by centre colour), while the
    video shows the cube in the camera's frame, and the two differ by
    however the solver was holding it. A recipient working from the video
    wants the camera frame — see `orientation_tracker.py`.
    """
    import reconstruct as RC

    cfg = json.loads((session / "config.json").read_text(encoding="utf-8"))
    moves = load_jsonl(session / "moves.jsonl")
    scramble = (cfg.get("claimed_start") or cfg.get("prescribed") or "").split()

    word = [m.get("camera_notation") or m.get("wca_notation") for m in moves]
    if any(w is None for w in word):
        raise SystemExit("moves.jsonl has unnamed moves; cannot build states")

    state = RC.seq_to_state(scramble) if scramble else RC.SOLVED.copy()
    rows = [{
        "i": 0,
        "move": None,
        "timestamp": None,
        "note": "state at the start of the recording, i.e. after the scramble",
        "facelets": RC._vec_to_cubie(state).to_facecube().to_string(),
    }]
    for i, (m, w) in enumerate(zip(moves, word), start=1):
        state = RC.compose_seq(state, [w])
        rows.append({
            "i": i,
            "move": w,
            "timestamp": m["timestamp"],
            "facelets": RC._vec_to_cubie(state).to_facecube().to_string(),
        })

    solved = bool((state == RC.SOLVED).all())
    meta = {
        "scramble": " ".join(scramble),
        "n_moves": len(word),
        "solve_word": " ".join(word),
        "ends_solved": solved,
        "notation_frame": ("camera" if moves and moves[0].get("camera_notation")
                           else "cube"),
    }
    return rows, meta


def cube_boxes(session: Path) -> list[dict] | None:
    """Per-frame cube bounding box, or None if the detector is unavailable."""
    import cv2
    from crop_utils import load_detector
    from prepare_data import per_frame_boxes

    detector = load_detector()
    if detector is None:
        return None

    recs = load_jsonl(session / "frames.jsonl")
    paths = [session / "frames" / r["file"] for r in recs]
    keep = [i for i, p in enumerate(paths) if p.exists()]
    recs = [recs[i] for i in keep]
    paths = [paths[i] for i in keep]
    n = len(paths)
    if n < 2:
        return None

    boxes, n_det = per_frame_boxes(detector,
                                   lambda i: cv2.imread(str(paths[i])), n)
    print(f"  boxes: {n_det} raw detections over {n} frames "
          f"-> smoothed + interpolated to all")
    return [{"idx": r["idx"], "file": r["file"], "ts": r["ts"],
             "box": [int(v) for v in b]} for r, b in zip(recs, boxes)]


README = """\
# Example solve — one recorded session

A single Rubik's Cube solve recorded from a webcam, with a smart cube
logging every turn it felt on the same wall clock. The video is the data;
the move log is a **label** recorded alongside it, and is never an input to
anything that reads the video.

Session: `{name}`
Recorded: {started_at} — {n_frames} frames at ~{fps:.1f} fps ({duration:.1f} s),
{n_moves} quarter turns.

## Files

| file | what |
|---|---|
| `frames/` | the video, one JPEG per frame, named `frame_<idx>_<unix ts>.jpg` |
| `frames.jsonl` | frame index -> filename and capture timestamp |
| `moves.jsonl` | every turn the cube reported: timestamp, face, direction |
| `cube_states.jsonl` | **ground-truth cube state after every move** |
| `boxes.jsonl` | cube bounding box in every frame |
| `config.json` | capture settings and the scramble the solve started from |
| `ble_meta.json` | cube health snapshot at the end of the take |

## The two derived files, and why they are the useful ones

**`cube_states.jsonl`** — for each move `i`, the 54-character facelet string
of the cube *after* that move, plus the timestamp it happened. Facelets are
in URFDLB order (the standard Kociemba layout): 9 Up, 9 Right, 9 Front, 9
Down, 9 Left, 9 Back, each read left-to-right, top-to-bottom, and each
character naming the FACE that colour belongs to rather than the colour
itself. Row `i = 0` is the state at the start of the recording.

This is what makes the session worth having: anything that reads colour off
the video can be scored against a known-correct state at any instant. Take a
frame's timestamp, find the last move at or before it in this file, and you
have the cube's exact configuration in that frame.

The frame -> state mapping has one honest caveat. A move takes ~100–200 ms of
real turning, so frames *during* a turn show a cube that is between two
states and matches neither. Timestamps here are the instant the cube reported
the turn complete. If you want frames that are unambiguously in one state,
take those at least ~150 ms clear of any move timestamp.

```python
import json

frames = [json.loads(l) for l in open("frames.jsonl")]
states = [json.loads(l) for l in open("cube_states.jsonl")]
boxes  = {{b["idx"]: b["box"] for b in map(json.loads, open("boxes.jsonl"))}}

def state_at(ts):
    "The cube's configuration in the frame captured at `ts`."
    last = states[0]
    for s in states[1:]:
        if s["timestamp"] is None or s["timestamp"] > ts:
            break
        last = s
    return last["facelets"]

f = frames[500]
print(f["file"], boxes[f["idx"]], state_at(f["ts"]))
```

**`boxes.jsonl`** — `[x1, y1, x2, y2]` in pixels for every frame, from a
YOLO detector fine-tuned on this cube, median-smoothed over time and
interpolated across frames where detection failed. Smoothed rather than
per-frame raw because raw boxes jitter frame to frame, and that jitter reads
as global motion to anything doing temporal differencing.

## Ground truth is verified, but the cube's own end-state flag is not

`ble_meta.json` says `cube_reported_end_solved: false` with
`end_facelets_wrong: 36`. **Ignore that field — the solve is good.** It is a
known defect in the cube's internal state tracking (one quarter turn moves 6
facelets, so 36 wrong is not a partly-finished solve; it is drift, and it
reports 36 constantly). What is trustworthy is the move log, and it was
checked: replaying the scramble followed by all {n_moves} logged moves from a
solved cube returns exactly to solved. `cube_states.jsonl`'s last row is the
solved state, and `ends_solved` in `export_meta.json` records that check.

## Notation

Standard WCA quarter turns — `R`, `R'` for the two directions of the right
face, and similarly `U D L F B`. Half turns appear as the same quarter turn
twice. `moves.jsonl` carries both `wca_notation` (the cube's own frame, named
by centre colour) and `camera_notation` (the camera's frame). **Use
`camera_notation`** if you are working from the video — it is what the
pictures show. `cube_states.jsonl` is built from it.

The solver held one orientation throughout: {front} facing the camera,
{top} on top. There are no whole-cube rotations in this take.
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="path to a training_data session dir")
    ap.add_argument("--out", required=True, help="destination folder")
    ap.add_argument("--no-boxes", action="store_true",
                    help="skip the detector pass")
    args = ap.parse_args()

    src = Path(args.session)
    dst = Path(args.out)
    if not (src / "frames.jsonl").is_file():
        raise SystemExit(f"{src} does not look like a session")
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} exists and is not empty — refusing to "
                         f"overwrite a folder someone may have edited")

    (dst / "frames").mkdir(parents=True, exist_ok=True)
    print(f"  copying frames...")
    n_frames = 0
    for p in sorted((src / "frames").iterdir()):
        shutil.copy2(p, dst / "frames" / p.name)
        n_frames += 1
    for name in COPY_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    print(f"  {n_frames} frames + {len(COPY_FILES)} metadata files")

    rows, meta = cube_states(src)
    with open(dst / "cube_states.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  cube_states.jsonl: {len(rows)} states, "
          f"ends solved: {meta['ends_solved']}")
    if not meta["ends_solved"]:
        print(f"  WARNING: the move log does NOT return this cube to solved. "
              f"Do not ship it\n           as an example solve without "
              f"finding out why (session_check.py).")

    boxes = None if args.no_boxes else cube_boxes(src)
    if boxes:
        with open(dst / "boxes.jsonl", "w", encoding="utf-8") as fh:
            for b in boxes:
                fh.write(json.dumps(b) + "\n")

    recs = load_jsonl(src / "frames.jsonl")
    ts = np.array([r["ts"] for r in recs], dtype=np.float64)
    dur = float(ts[-1] - ts[0])
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))

    (dst / "export_meta.json").write_text(json.dumps({
        "source_session": src.name,
        "exported_by": "ble/export_example_session.py",
        "n_frames": n_frames,
        "duration_seconds": round(dur, 2),
        "fps": round((len(recs) - 1) / dur, 2) if dur > 0 else None,
        "has_boxes": bool(boxes),
        **meta,
    }, indent=2) + "\n", encoding="utf-8")

    (dst / "README.md").write_text(README.format(
        name=src.name,
        started_at=cfg.get("started_at", "?"),
        n_frames=n_frames,
        fps=(len(recs) - 1) / dur if dur > 0 else 0.0,
        duration=dur,
        n_moves=meta["n_moves"],
        front=cfg.get("front_color", "?"),
        top=cfg.get("top_color", "?"),
    ), encoding="utf-8")

    total = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"\n  -> {dst}  ({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
