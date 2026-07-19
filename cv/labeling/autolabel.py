"""
autolabel.py — expedited bounding-box labeling for hands-on solve frames
(ROADMAP §2.1 dataset dependency).

Model-assisted flow — you never draw a box:

  1. propose : detector auto-labels confident frames; boxes are linearly
               interpolated through short detection gaps (the occluded,
               hands-on frames — exactly the training data we're missing).
               Every candidate gets a preview jpg with the box drawn:
               green = detector, orange = interpolated.
  2. (you)   : open the review/ folder, DELETE previews whose box is wrong.
               Keeping = accepting. This is ~10x faster than drawing boxes.
  3. finalize: keeps only surviving labels, splits train/val, writes a
               YOLO dataset (dataset.yaml) ready for ultralytics training
               or Roboflow import.

Run from inside cv/labeling/ (bare model filenames — see CLAUDE.md):
  python record_frames.py --output sessions
  python autolabel.py propose  --session sessions/solve_XXX
  python autolabel.py finalize --out autolabel_out

(A ble/ move-tracking session also works as --session, e.g.
../../ble/training_data/solve_XXX — propose() just reads session/frames/.)
"""

import argparse
import os
import shutil
import sys
import time

import cv2

# continuity_guard/cube_detector live in the sibling cv/detection/ folder —
# this script is run from inside cv/labeling/, so add it to sys.path to
# keep the `from continuity_guard import ...` below bare.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "detection"))

ACCEPT_CONF       = 0.3   # detector box auto-accepted at/above this
FLOOR_CONF        = 0.10   # detections below this are ignored entirely
INTERP_MAX_FRAMES = 45     # interpolate gaps up to this many frames (~1.5s)
STRIDE            = 2      # keep every Nth frame (dataset diversity vs size)
VAL_FRAC          = 0.15

OUT = "autolabel_out"


def _write_retrying(path, write_fn, retries=5, delay=0.1):
    """Write via write_fn(path), tolerating a freshly-created parent folder
    not being immediately visible to a subsequent write (observed on
    OneDrive-synced folders under rapid file creation — os.makedirs()
    returns before the cloud-sync filter driver has materialized the
    directory). Re-issues the makedirs and retries a few times before
    giving up for real."""
    for attempt in range(retries):
        try:
            ok = write_fn(path)
            if ok is False:  # cv2.imwrite() failure signal (no exception)
                raise FileNotFoundError(path)
            return
        except FileNotFoundError:
            if attempt == retries - 1:
                raise
            os.makedirs(os.path.dirname(path), exist_ok=True)
            time.sleep(delay)


def _yolo_line(box, fw, fh):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh
    w, h = (x2 - x1) / fw, (y2 - y1) / fh
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def propose(session, stride):
    from continuity_guard import detect_cubes, dedup_boxes

    frames_dir = os.path.join(session, "frames")
    names = sorted(n for n in os.listdir(frames_dir)
                   if n.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        raise SystemExit(f"no frames in {frames_dir}")

    tag = os.path.basename(os.path.normpath(session))
    dirs = {k: os.path.join(OUT, tag, k) for k in ("images", "labels", "review")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # pass 1: detect on every frame (cheap relative to labeling by hand)
    det = {}   # frame index -> (box, conf)
    fw = fh = None
    for i, n in enumerate(names):
        frame = cv2.imread(os.path.join(frames_dir, n))
        if frame is None:
            continue
        if fw is None:
            fh, fw = frame.shape[:2]
        boxes = dedup_boxes(detect_cubes(frame, conf_floor=FLOOR_CONF))
        if boxes:
            best = max(boxes, key=lambda b: b[4])
            if best[4] >= ACCEPT_CONF:
                det[i] = (best[:4], best[4])

    # pass 2: interpolate through short gaps between accepted detections
    interp = {}
    keys = sorted(det)
    for a, b in zip(keys, keys[1:]):
        gap = b - a
        if 1 < gap <= INTERP_MAX_FRAMES:
            (ax1, ay1, ax2, ay2), _ = det[a]
            (bx1, by1, bx2, by2), _ = det[b]
            for j in range(a + 1, b):
                t = (j - a) / gap
                interp[j] = tuple(
                    p + (q - p) * t
                    for p, q in zip((ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2)))

    # pass 3: write images + labels + review previews
    n_det = n_int = 0
    for i, n in enumerate(names):
        if i % stride:
            continue
        if i in det:
            box, kind = det[i][0], "det"
        elif i in interp:
            box, kind = interp[i], "int"
        else:
            continue
        frame = cv2.imread(os.path.join(frames_dir, n))
        if frame is None:
            continue
        stem = f"{tag}_{i:05d}"
        _write_retrying(os.path.join(dirs["images"], stem + ".jpg"),
                         lambda p: cv2.imwrite(p, frame))

        def _write_label(p, _text=_yolo_line(box, fw, fh)):
            with open(p, "w") as f:
                f.write(_text)
        _write_retrying(os.path.join(dirs["labels"], stem + ".txt"), _write_label)

        prev = frame.copy()
        x1, y1, x2, y2 = map(int, box)
        color = (0, 220, 60) if kind == "det" else (0, 150, 255)
        cv2.rectangle(prev, (x1, y1), (x2, y2), color, 3)
        _write_retrying(os.path.join(dirs["review"], stem + ".jpg"),
                         lambda p: cv2.imwrite(p, prev))
        n_det += kind == "det"
        n_int += kind == "int"

    total = len(range(0, len(names), stride))
    print(f"{tag}: {n_det} detector labels + {n_int} interpolated "
          f"(coverage {(n_det + n_int) / max(total, 1):.0%} of {total} sampled frames)")
    print(f"review:  DELETE bad previews in {dirs['review']}")
    print(f"then:    python autolabel.py finalize")


def finalize():
    import random
    random.seed(0)
    kept, dropped = [], 0
    for tag in sorted(os.listdir(OUT)):
        base = os.path.join(OUT, tag)
        rev = os.path.join(base, "review")
        if not os.path.isdir(rev):
            continue
        surviving = {os.path.splitext(n)[0] for n in os.listdir(rev)}
        for n in os.listdir(os.path.join(base, "images")):
            stem = os.path.splitext(n)[0]
            if stem in surviving:
                kept.append((os.path.join(base, "images", n),
                             os.path.join(base, "labels", stem + ".txt")))
            else:
                dropped += 1
    if not kept:
        raise SystemExit("nothing kept — run propose first / don't delete everything")

    ds = os.path.join(OUT, "dataset")
    random.shuffle(kept)
    n_val = max(1, int(len(kept) * VAL_FRAC))
    for split, items in (("val", kept[:n_val]), ("train", kept[n_val:])):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(ds, sub, split), exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, os.path.join(ds, "images", split))
            shutil.copy2(lbl, os.path.join(ds, "labels", split))
    with open(os.path.join(ds, "dataset.yaml"), "w") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\n"
                "names:\n  0: cube\n")
    print(f"dataset: {len(kept) - n_val} train / {n_val} val "
          f"({dropped} rejected in review) -> {ds}")
    print("train:   yolo detect train data=autolabel_out/dataset/dataset.yaml "
          "model=detect_full_cube.pt epochs=40 imgsz=640")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--session", required=True)
    p.add_argument("--stride", type=int, default=STRIDE)
    sub.add_parser("finalize")
    args = ap.parse_args()
    if args.cmd == "propose":
        propose(args.session, args.stride)
    else:
        finalize()
