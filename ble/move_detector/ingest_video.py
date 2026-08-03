"""
ingest_video.py

Turn an arbitrary video file into a session directory the rest of this
pipeline already understands — the entry point for footage that did not
come from `ble/record_training.py` and has no smart cube behind it.

Output layout, identical to what record_training.py writes:

    <out>/<name>/frames/frame_000000_<ts>.jpg   ...
    <out>/<name>/frames.jsonl                   {"idx", "ts", "file"}
    <out>/<name>/metadata.json                  provenance for the source

and nothing else. There is deliberately no `moves.jsonl`: this script does
not know what the moves were, which is the whole reason review.py exists.

Then:

    python prepare_data.py --sessions <out>/<name>/ --color --labels none
    python review.py --session <out>/<name>/ --model checkpoints/move_ctc_aug44_s0.pt \\
        --start-moves "<the scramble>"
    python prepare_data.py --sessions <out>/<name>/ --color \\
        --labels review --force

Frame rate is the thing to get right
------------------------------------
The models are trained at ~30fps and their temporal geometry is quoted in
FRAMES, not seconds: the TCN's receptive field is 61 frames (model.py),
decode.MIN_SEP is 2 frames, and the CTC collapse rule needs a blank frame
between repeated labels. Feed a 60fps video in unchanged and every one of
those halves in real-time terms — the receptive field covers 1.0s instead
of 2.0s, and two moves 67ms apart go from "crowded" to "comfortable" while
the model still sees them the way it was trained to. So the default is to
RESAMPLE to --fps by picking the source frame nearest each target time,
and the written timestamps are those frames' true presentation times, so
prepare_data's own `n / duration` recovers ~30 rather than the source rate.

`--fps 0` disables resampling and takes every frame, which is right only
when the source is already ~30fps.

Timestamps come from the container's presentation time (CAP_PROP_POS_MSEC)
rather than index/fps arithmetic, because phone and screen-captured video
is frequently variable-rate and the two answers diverge by seconds over a
minute. Where the container reports nothing usable, this falls back to
index/fps and says so.

What this cannot do for you
---------------------------
Getting frames out is easy; the two things that actually decide whether
external footage is usable are the cube detector finding the cube, and you
knowing the start state. `--check` answers the first (it samples frames
and reports the detection rate before you spend the preparation on it) —
a session where the cube is never detected falls back to a centred square
crop, which is a different input scale from every trained model and which
dataset.check_crop_regime will refuse. The second is on you: review.py
needs --start-moves, --start-state or nothing at all (in which case the
group-theory flags are off and you are reviewing on the timeline alone).

Usage:
    python ingest_video.py --video take.mp4 --probe
    python ingest_video.py --video take.mp4 --out ../captures
    python ingest_video.py --video take.mp4 --out ../captures \\
        --start 12.5 --end 71.0 --name wr_attempt --check
    python ingest_video.py --video "clips/*.mp4" --out ../captures
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_BLE_DIR = Path(__file__).resolve().parents[1]
if str(_BLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLE_DIR))

TARGET_FPS = 30.0     # what every trained model expects — see the docstring
MAX_WIDTH = 1280      # matches the recorded corpus (1280x720)
JPEG_QUALITY = 85     # matches record_training.py's config.json
CHECK_SAMPLES = 24    # frames sampled by --check


def probe(path: Path) -> dict:
    """Container-level facts, without decoding the whole file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    info = {
        "path": str(path),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "rotation": int(cap.get(getattr(cv2, "CAP_PROP_ORIENTATION_META", -1))
                        or 0) if hasattr(cv2, "CAP_PROP_ORIENTATION_META")
        else 0,
    }
    info["duration"] = (info["n_frames"] / info["fps"]
                        if info["fps"] > 0 else 0.0)
    cap.release()
    return info


def print_probe(info: dict) -> None:
    print(f"  {Path(info['path']).name}")
    print(f"    {info['width']}x{info['height']}  {info['fps']:.2f}fps  "
          f"{info['n_frames']} frames  {info['duration']:.1f}s"
          + (f"  rotation {info['rotation']}deg" if info["rotation"] else ""))
    if info["fps"] > 0 and abs(info["fps"] - TARGET_FPS) > 1.0:
        print(f"    NOTE: source is {info['fps']:.1f}fps; frames will be "
              f"resampled to {TARGET_FPS:.0f}fps (--fps 0 to disable)")


def _rotate(img: np.ndarray, deg: int) -> np.ndarray:
    if deg % 360 == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg % 360 == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg % 360 == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _downscale(img: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    s = max_width / img.shape[1]
    return cv2.resize(img, (max_width, int(round(img.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)


def safe_name(stem: str) -> str:
    """A session directory name that is safe on Windows and greppable."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.")
    return s or "take"


def extract(video: Path, out_dir: Path, fps: float, start: float, end: float,
            max_width: int, rotate: int, quality: int,
            probe_info: dict) -> dict:
    """
    Decode `video` into out_dir/frames/ + frames.jsonl. Returns a summary.

    Decoding is SEQUENTIAL with a per-frame accept/reject test rather than
    seeking to each target time. Seeking a variable-rate or long-GOP file
    lands on the nearest keyframe, not the requested one, so a seek-based
    resampler silently returns duplicated frames — which would look to the
    onset detector like the cube stopping dead, the one signal it is most
    sensitive to.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.jpg"):
        stale.unlink()

    src_fps = probe_info["fps"] or TARGET_FPS
    period = 1.0 / fps if fps > 0 else 0.0
    # An arbitrary but real wall-clock origin, so the written timestamps
    # look like the recorded corpus and metadata stays human-readable.
    t0 = video.stat().st_mtime

    recs = []
    idx = 0
    src_i = 0
    next_t = start
    used_pos_msec = True
    encode = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos is None or pos <= 0.0:
            # Container reported nothing usable for this frame; fall back to
            # index arithmetic for it. Tracked, and reported at the end.
            t = src_i / src_fps
            if src_i > 0:
                used_pos_msec = False
        else:
            t = pos / 1000.0
        src_i += 1

        if t < start:
            continue
        if end is not None and t > end:
            break
        if period and t < next_t:
            continue
        if period:
            # Advance past every target slot this frame satisfies, so a
            # dropped/slow source frame does not put the sampler into a
            # catch-up burst that emits several near-identical frames.
            while next_t <= t:
                next_t += period

        img = _downscale(_rotate(frame, rotate), max_width)
        ts = t0 + t
        name = f"frame_{idx:06d}_{ts:.3f}.jpg"
        cv2.imwrite(str(frames_dir / name), img, encode)
        recs.append({"idx": idx, "ts": ts, "file": name})
        idx += 1

    cap.release()
    if not recs:
        raise SystemExit(f"{video.name}: no frames in the requested range "
                         f"(--start {start} --end {end})")

    with open(out_dir / "frames.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    span = recs[-1]["ts"] - recs[0]["ts"]
    return {"n": len(recs), "duration": span,
            "fps": len(recs) / span if span > 0 else float(fps or src_fps),
            "used_pos_msec": used_pos_msec,
            "shape": [int(img.shape[1]), int(img.shape[0])]}


def check_detection(out_dir: Path, samples: int = CHECK_SAMPLES) -> dict | None:
    """
    Will the cube detector actually find the cube in this footage?

    Worth answering BEFORE preparing a stream, because the failure is
    quiet: prepare_data falls back to a centred square crop when nothing is
    detected, which is a different input scale from every trained model,
    and the first thing that says so is a trainer refusing the whole set
    much later. Also reports how big the cube is in frame — competition
    wide shots put it across a few dozen pixels, which survives the crop
    but not much else.
    """
    try:
        from crop_utils import load_detector, detect_box
    except Exception as exc:                                # pragma: no cover
        print(f"    --check unavailable: {exc}")
        return None
    detector = load_detector()
    if detector is None:
        print(f"    --check skipped: no cube detector (needs ultralytics + "
              f"cv/detection/detect_full_cube.pt)")
        return None

    recs = [json.loads(l) for l in open(out_dir / "frames.jsonl") if l.strip()]
    picks = np.linspace(0, len(recs) - 1, min(samples, len(recs))).astype(int)
    hits, fracs = 0, []
    for i in picks:
        img = cv2.imread(str(out_dir / "frames" / recs[int(i)]["file"]))
        if img is None:
            continue
        box = detect_box(detector, img)
        if box is None:
            continue
        hits += 1
        x1, y1, x2, y2 = box
        fracs.append(max(x2 - x1, y2 - y1) / img.shape[1])

    rate = hits / max(len(picks), 1)
    med = float(np.median(fracs)) if fracs else 0.0
    print(f"    cube detected in {hits}/{len(picks)} sampled frames "
          f"({rate*100:.0f}%)"
          + (f", median box {med*100:.0f}% of frame width" if fracs else ""))
    if rate < 0.5:
        print(f"    WARNING: the detector rarely finds the cube here. "
              f"prepare_data.py will fall")
        print(f"             back to a centred square crop, which is a "
              f"different input scale from")
        print(f"             every trained model — check_crop_regime will "
              f"refuse the set later.")
    elif med and med < 0.15:
        print(f"    NOTE: the cube is small in frame. The crop still works, "
              f"but detail per")
        print(f"          sticker is well below the recorded corpus.")
    return {"rate": rate, "median_box_frac": med, "n_sampled": len(picks)}


def ingest(video: Path, out_root: Path, args) -> bool:
    info = probe(video)
    print_probe(info)

    name = safe_name(args.name or video.stem)
    out_dir = out_root / name
    if (out_dir / "frames.jsonl").exists() and not args.force:
        print(f"    {out_dir} already ingested, skipping (--force to redo)")
        return True

    rotate = args.rotate if args.rotate is not None else info["rotation"]
    t0 = time.time()
    res = extract(video, out_dir, args.fps, args.start, args.end,
                  args.max_width, rotate, args.quality, info)

    print(f"    -> {out_dir}")
    print(f"       {res['n']} frames, {res['duration']:.1f}s, "
          f"{res['fps']:.2f}fps, {res['shape'][0]}x{res['shape'][1]}"
          + (f", rotated {rotate}deg" if rotate else "")
          + f"  ({time.time()-t0:.1f}s)")
    if not res["used_pos_msec"]:
        print(f"       NOTE: the container did not report presentation "
              f"times for every frame;")
        print(f"             those fell back to index/fps, which drifts on "
              f"variable-rate video.")

    check = check_detection(out_dir) if args.check else None

    (out_dir / "metadata.json").write_text(json.dumps({
        "session_id": name,
        "source": "ingest_video",
        "source_video": str(video.resolve()),
        "source_fps": info["fps"],
        "source_size": [info["width"], info["height"]],
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_fps": args.fps,
        "trim": [args.start, args.end],
        "rotate": rotate,
        "max_width": args.max_width,
        "jpeg_quality": args.quality,
        "total_frames": res["n"],
        "measured_fps": res["fps"],
        "used_presentation_times": res["used_pos_msec"],
        "detector_check": check,
        # Stated, not implied. Everything downstream that asks "what is the
        # ground truth here" must get an explicit no rather than find a
        # missing file and guess.
        "has_ble_truth": False,
        "note": "No move log. Label with review.py; see its docstring.",
    }, indent=1))
    return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", nargs="+", required=True,
                   help="video file(s); globs allowed")
    p.add_argument("--out", default="../captures",
                   help="root directory for the session folder(s). Defaults "
                        "to ../captures, kept separate from ../training_data "
                        "so external footage never silently joins the "
                        "smart-cube corpus")
    p.add_argument("--name", default=None,
                   help="session directory name (default: the video's stem). "
                        "Only meaningful for a single --video")
    p.add_argument("--fps", type=float, default=TARGET_FPS,
                   help=f"resample to this frame rate (default "
                        f"{TARGET_FPS:.0f}, what every model is trained at); "
                        f"0 keeps every source frame")
    p.add_argument("--start", type=float, default=0.0,
                   help="skip everything before this many seconds")
    p.add_argument("--end", type=float, default=None,
                   help="stop at this many seconds")
    p.add_argument("--max-width", type=int, default=MAX_WIDTH,
                   help=f"downscale wider frames (default {MAX_WIDTH}, the "
                        f"recorded corpus width); 0 keeps full resolution")
    p.add_argument("--rotate", type=int, default=None,
                   choices=(0, 90, 180, 270),
                   help="rotate frames clockwise. Defaults to the "
                        "container's own orientation metadata, which OpenCV "
                        "does not always apply on its own")
    p.add_argument("--quality", type=int, default=JPEG_QUALITY)
    p.add_argument("--check", action="store_true",
                   help="after extracting, sample frames and report whether "
                        "the cube detector finds the cube — do this before "
                        "spending a stream preparation on the footage")
    p.add_argument("--probe", action="store_true",
                   help="print the video's properties and exit")
    p.add_argument("--force", action="store_true",
                   help="re-ingest even if the session directory exists")
    args = p.parse_args()

    videos = [Path(q) for pattern in args.video
              for q in (sorted(Path().glob(pattern)) if "*" in pattern
                        else [Path(pattern)])]
    videos = [v for v in videos if v.is_file()]
    if not videos:
        sys.exit("No video files matched --video.")
    if args.name and len(videos) > 1:
        sys.exit("--name is only meaningful with a single --video.")

    if args.probe:
        print()
        for v in videos:
            print_probe(probe(v))
        return

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\nIngesting {len(videos)} video(s) into {out_root}...")
    ok = sum(ingest(v, out_root, args) for v in videos)
    print(f"\nDone: {ok}/{len(videos)}.\n")
    print(f"Next, per session:")
    print(f"    python prepare_data.py --sessions {out_root}/<name>/ "
          f"--color --labels none")
    print(f"    python review.py --session {out_root}/<name>/ "
          f"--model checkpoints/move_ctc_aug44_s0.pt --start-moves \"<scramble>\"")
    print(f"    python prepare_data.py --sessions {out_root}/<name>/ "
          f"--color --labels review --force\n")


if __name__ == "__main__":
    main()
