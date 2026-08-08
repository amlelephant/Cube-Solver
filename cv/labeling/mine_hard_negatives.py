"""
mine_hard_negatives.py — find detector false positives in already-recorded
sessions and stage them for review as YOLO hard negatives.

Today's continuity-guard false-DQ investigation found the detector
confidently (0.5-0.85) mislabeling static room fixtures (a fish-tank stand,
a dresser corner) as a second cube. That's a detector training-data gap, not
a guard-logic bug — this script mines exactly that failure mode instead of
proposing more ordinary positive cube frames (autolabel.py already does
that).

Unlike autolabel.py, this NEVER copies a training-resolution frame anywhere.
Every candidate's actual image data stays at its original path under
ble/training_data/ or cv/labeling/sessions/ — only a small downscaled
preview (for human review) and a JSON manifest entry (frame path + box) are
written. See finalize()'s docstring for why, and materialize_dataset.py (not
this file) for the one step that touches the filesystem beyond that, via
hardlinks, not copies.

Flow:
  1. scan     : for each frame, track "the real cube" by position continuity
               (continuity_guard.pick_continuity — the exact same rule that
               stopped a background object from hijacking guard tracking
               today). Any OTHER confident box in the same frame is a
               hard-negative candidate — something the detector currently
               believes is cube-shaped but isn't the cube we're tracking.
               Candidates are clustered by position (a static fixture
               produces near-identical boxes for the whole time it's
               visible) and capped per cluster, so one 30-second phantom
               doesn't turn into hundreds of near-duplicate frames.
  2. (you)    : open mine_out/<session>/review/, DELETE previews that are
               WRONG (i.e. genuinely a second real cube, not background
               clutter) — keeping a preview accepts it as a hard negative.
  3. finalize : merges every session's surviving candidates into one master
               JSON manifest.

Run from inside cv/labeling/ (bare model filenames — see CLAUDE.md):
  python mine_hard_negatives.py scan --root ../../ble/training_data
  python mine_hard_negatives.py finalize
"""

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "detection"))

from continuity_guard import detect_cubes, dedup_boxes, pick_continuity, _iou

MINE_CONF       = 0.30   # candidate secondary-box confidence floor (matches
                         #   autolabel.py's ACCEPT_CONF — anything the current
                         #   model finds this plausible is worth correcting)
CLUSTER_IOU     = 0.5    # secondary boxes this overlapping, across the whole
                         #   session, are treated as the same phantom
MAX_PER_CLUSTER = 6      # representative frames kept per distinct phantom
                         #   (a 30s-sustained fixture would otherwise yield
                         #   hundreds of near-duplicate candidates)
PREVIEW_WIDTH   = 480    # review thumbnails are for human eyeballing only —
                         #   downscaled so mine_out/ stays small regardless
                         #   of corpus size (see module docstring)
STRIDE          = 1      # frame sampling stride during the scan pass

OUT = "mine_out"


def _repo_relpath(path):
    # store paths relative to the repo root so the manifest is portable and
    # never needs the frame's bytes copied anywhere.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.relpath(os.path.abspath(path), repo_root).replace(os.sep, "/")


def scan(root, stride):
    sessions = sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "frames.jsonl"))
        and os.path.isdir(os.path.join(root, d, "frames"))
    )
    if not sessions:
        raise SystemExit(f"no sessions with frames/ + frames.jsonl under {root}")

    total_candidates = 0
    for si, name in enumerate(sessions):
        session_dir = os.path.join(root, name)
        candidates = _scan_session(session_dir, name, stride)
        total_candidates += len(candidates)
        print(f"[{si + 1}/{len(sessions)}] {name}: {len(candidates)} hard-negative candidates")
        if candidates:
            _write_review(name, candidates)
    print(f"\n{total_candidates} candidates staged across {len(sessions)} sessions")
    print(f"review:  DELETE bad previews in {OUT}/<session>/review/")
    print(f"then:    python mine_hard_negatives.py finalize")


def _scan_session(session_dir, tag, stride):
    manifest = os.path.join(session_dir, "frames.jsonl")
    frames_dir = os.path.join(session_dir, "frames")
    entries = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda r: r["ts"])

    ref_box = None          # the box we believe is the real, tracked cube
    clusters = []            # [{"box": representative box, "hits": [(frame_path, box)]}]
    for i, rec in enumerate(entries):
        if i % stride:
            continue
        path = os.path.join(frames_dir, rec["file"])
        frame = cv2.imread(path)
        if frame is None:
            continue
        boxes = dedup_boxes(detect_cubes(frame))
        if not boxes:
            continue
        primary = pick_continuity(boxes, ref_box)
        ref_box = primary
        for b in boxes:
            if b is primary or b[4] < MINE_CONF:
                continue
            _add_to_cluster(clusters, path, b)

    out = []
    for c in clusters:
        for path, box in c["hits"][:MAX_PER_CLUSTER]:
            out.append((path, box))
    return out


def _add_to_cluster(clusters, path, box):
    for c in clusters:
        if _iou(box, c["box"]) >= CLUSTER_IOU:
            c["hits"].append((path, box))
            return
    clusters.append({"box": box, "hits": [(path, box)]})


def _write_review(tag, candidates):
    review_dir = os.path.join(OUT, tag, "review")
    os.makedirs(review_dir, exist_ok=True)
    manifest_path = os.path.join(OUT, tag, "candidates.json")
    manifest = []
    for i, (path, box) in enumerate(candidates):
        stem = f"{tag}_{i:04d}"
        frame = cv2.imread(path)
        if frame is None:
            continue
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box[:4])
        prev = frame.copy()
        cv2.rectangle(prev, (x1, y1), (x2, y2), (0, 60, 230), 3)
        cv2.putText(prev, f"conf {box[4]:.2f} -- NOT a cube?", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 230), 2)
        scale = PREVIEW_WIDTH / fw
        prev = cv2.resize(prev, (PREVIEW_WIDTH, int(fh * scale)))
        cv2.imwrite(os.path.join(review_dir, stem + ".jpg"), prev)
        manifest.append({"stem": stem, "frame": _repo_relpath(path),
                         "box": [round(v, 1) for v in box[:4]],
                         "conf": round(box[4], 3)})
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def finalize():
    kept, dropped = [], 0
    for tag in sorted(os.listdir(OUT)):
        base = os.path.join(OUT, tag)
        review_dir = os.path.join(base, "review")
        manifest_path = os.path.join(base, "candidates.json")
        if not (os.path.isdir(review_dir) and os.path.isfile(manifest_path)):
            continue
        surviving = {os.path.splitext(n)[0] for n in os.listdir(review_dir)}
        with open(manifest_path) as f:
            candidates = json.load(f)
        for c in candidates:
            if c["stem"] in surviving:
                kept.append({"frame": c["frame"], "box": c["box"], "label": "negative",
                            "session": tag})
            else:
                dropped += 1
    if not kept:
        raise SystemExit("nothing kept — run scan first / don't delete everything")

    out_path = os.path.join(OUT, "hard_negatives.json")
    with open(out_path, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"{len(kept)} confirmed hard negatives ({dropped} rejected in review) -> {out_path}")
    print("No image data was copied — every entry points at its original frame in place.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--root", required=True)
    p.add_argument("--stride", type=int, default=STRIDE)
    sub.add_parser("finalize")
    args = ap.parse_args()
    if args.cmd == "scan":
        scan(args.root, args.stride)
    else:
        finalize()
