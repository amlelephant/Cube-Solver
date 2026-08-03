"""
prepare_data.py

One-time preprocessing: turn a recorded session's raw frames/ directory
into a compact array the onset detector can train on fast.

For each session it writes  <session>/detector_stream.npz  containing:
    frames     uint8 (N, 96, 96)  grayscale, cube-cropped, downscaled
    onset_idx  int32 (M,)         frame index of each BLE move event
    onset_ts   float64 (M,)       original BLE timestamps (for reference)
    fps        float
    name       str

Why precompute
--------------
Training samples 96-frame clips at random. Decoding 96 JPEGs per sample on
every __getitem__ makes the data loader the bottleneck (Windows runs
num_workers=0 here). A whole session at 96x96 grayscale is ~15MB, so every
session fits in memory and clip sampling becomes an array slice.

Cropping
--------
Uses the same cv/detection cube detector as cache_crops.py, but the unit
is different: cache_crops.py resolves ONE box per move window, whereas
this needs a box for every frame of a continuous stream.

Detection runs every --crop-stride frames, the resulting box sequence is
median-smoothed and then linearly interpolated to every frame. That tracks
the cube slowly drifting around the frame as you turn it, without
injecting the detector's frame-to-frame jitter — jitter would look like
global motion, which is exactly the signal the detector is trying to read
(same rule as crop_utils' module docstring, applied over time).

Sessions where the cube is never detected fall back to a centered square
crop, so a missing/failed detector degrades quality rather than blocking
training.

Where the onset labels come from
--------------------------------
`--labels ble` (the default, and the only behaviour that existed before
2026-08-03) reads `moves.jsonl`: the smart cube's own move log, aligned to
frames by wall-clock timestamp.

`--labels review` reads `review_labels.json` — the output of review.py,
where a human corrected the model's decode of a session that has no smart
cube behind it at all. Two differences matter and are handled explicitly:

  * The labels carry a FRAME INDEX, not just a timestamp, because a
    reviewer placed them against the video. That index is used directly
    instead of being round-tripped through `searchsorted`, so the
    ~1-frame alignment noise the BLE path reports (and the 10.1% of BLE
    moves that share a 30ms tick and are structurally unmatchable to a
    distinct frame — GROUND_TRUTH_ARTIFACTS.md) simply does not arise.
  * A reviewed session is only accepted when its label set REACHES SOLVED
    from the scanned start state (`reaches_solved` in the file). That is
    not a formality: an incomplete review is a label set with a move
    missing from it, and a missing move is not a neutral omission — it
    actively teaches the model not to fire there, which is the single
    largest term in the measured error budget (see review.py). The group
    theory is the only cheap certificate that nothing is missing, so it
    gates the corpus. `--allow-unsolved` overrides, loudly.

`--labels auto` prefers a reviewed file where one exists and falls back to
BLE.

`--labels none` writes a stream with NO onsets at all. That exists to break
a circular dependency in the external-video path: review.py needs a scored
posteriorgram, scoring needs `detector_stream_color.npz`, and building that
stream normally needs the very labels the review is about to produce. An
unlabelled stream is for INFERENCE ONLY and is poison in a training set —
every frame of it is a negative, so it teaches the model that a whole
session contains no moves. dataset.check_label_regime refuses one outright
rather than trusting anybody to remember that.

Which source was used travels WITH the stream as `label_source`, for
exactly the reason `crop_mode` does — see dataset.check_label_regime.

The external-video pipeline end to end:

    python ingest_video.py --video take.mp4 --out ../captures
    python prepare_data.py --sessions ../captures/take/ --color --labels none
    python review.py --session ../captures/take/ --model checkpoints/move_ctc_aug44_s0.pt \\
        --start-moves "<the scramble>"
    python prepare_data.py --sessions ../captures/take/ --color \\
        --labels review --force

Usage:
    python prepare_data.py --sessions ../training_data/solve_*/
    python prepare_data.py --sessions ../training_data/solve_*/ --force
    python prepare_data.py --sessions ../training_data/solve_*/ --no-crop
    python prepare_data.py --sessions ../captures/take_*/ --color \\
        --labels review
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# crop_utils lives in ble/ — same sys.path bootstrap pattern the cv/
# cross-topic scripts use (see CLAUDE.md).
_BLE_DIR = Path(__file__).resolve().parents[1]
if str(_BLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLE_DIR))

FRAME_SIZE   = 96    # must match model.FRAME_SIZE
CROP_STRIDE  = 10    # detect every Nth frame (~3Hz at 30fps)
SMOOTH_WIN   = 5     # rolling median over this many detections (~1.7s)
STREAM_FILE  = "detector_stream.npz"

# MODEL_REWORK_PLAN.md Stage A: a second, colour stream alongside the
# original grayscale one — NOT a replacement. detector_stream.npz and the
# deployed onset-only model/pipeline are completely untouched; --color
# writes this file instead, read only by the new joint model's dataset
# loader (dataset.JointSessionStream). Kept as a separate file rather than
# widening the existing one so a stale/partial Stage A prototype can never
# silently change what the shipped detector trains or scores on.
STREAM_FILE_COLOR = "detector_stream_color.npz"

try:
    from crop_utils import CROP_MARGIN
except Exception:                                       # pragma: no cover
    CROP_MARGIN = -1.0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def center_square(shape) -> tuple:
    """Fallback crop: largest centered square. Keeps aspect ratio sane."""
    h, w = shape[:2]
    side = min(h, w)
    x, y = (w - side) // 2, (h - side) // 2
    return (x, y, x + side, y + side)


def _rolling_median(arr: np.ndarray, win: int) -> np.ndarray:
    """Median filter along axis 0, edge-padded. arr is (N, 4)."""
    if len(arr) < 2 or win < 2:
        return arr
    half = win // 2
    padded = np.pad(arr, ((half, half), (0, 0)), mode="edge")
    return np.stack([np.median(padded[i:i + win], axis=0)
                     for i in range(len(arr))])


def per_frame_boxes(detector, load_frame, n_frames: int,
                    stride: int = CROP_STRIDE) -> tuple[np.ndarray, int]:
    """
    A crop box for every frame: detect every `stride` frames, median-smooth
    the detections, then linearly interpolate across all frames.

    `load_frame` is a callable i -> BGR image (or None). Taking a callable
    rather than a list of paths lets live capture reuse this unchanged on
    in-memory frames, so inference crops exactly the way training cropped.

    Returns (boxes (N, 4) int, n_detected). Falls back to a centered square
    for the whole session if nothing is ever detected.
    """
    from crop_utils import detect_box, square_with_margin

    probe_idx, raw = [], []
    shape = None
    for i in range(0, n_frames, stride):
        img = load_frame(i)
        if img is None:
            continue
        shape = img.shape
        box = detect_box(detector, img)
        if box is not None:
            probe_idx.append(i)
            raw.append(box)

    if shape is None:
        raise RuntimeError("no readable frames")

    if not raw:
        fallback = center_square(shape)
        return np.tile(fallback, (n_frames, 1)).astype(np.int32), 0

    smoothed = _rolling_median(np.array(raw, dtype=np.float32), SMOOTH_WIN)

    # Interpolate each coordinate across every frame index
    all_idx = np.arange(n_frames)
    interp = np.stack([np.interp(all_idx, probe_idx, smoothed[:, c])
                       for c in range(4)], axis=1)

    boxes = np.array([square_with_margin(tuple(b), shape) for b in interp],
                     dtype=np.int32)
    return boxes, len(raw)


def crop_to_box(img: np.ndarray, box) -> np.ndarray:
    """Clamp `box` to the image and crop; returns img unchanged if degenerate."""
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img
    return img[y1:y2, x1:x2]


def build_gray_stream(load_frame, boxes: np.ndarray, n_frames: int
                      ) -> np.ndarray:
    """
    (N, FRAME_SIZE, FRAME_SIZE) uint8 grayscale, each frame cropped to its
    box. This is the detector's input format — shared by offline
    preparation and live capture so the two cannot drift apart.

    An unreadable frame repeats its predecessor, which produces a
    zero-motion diff rather than a spurious one.
    """
    out = np.empty((n_frames, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    for i in range(n_frames):
        img = load_frame(i)
        if img is None:
            out[i] = out[i - 1] if i > 0 else 0
            continue
        # Branch on channel COUNT, not ndim. Importing ultralytics
        # monkeypatches cv2.imread (ultralytics.utils.patches) to return
        # im[..., None] for 2-D reads, so an IMREAD_GRAYSCALE load comes back
        # (H, W, 1) rather than (H, W) — but only once a detector has been
        # loaded, which is why a detector-free read of the same file looks
        # fine. An ndim check then feeds 1 channel into BGR2GRAY and throws.
        if img.ndim == 3:
            img = img[:, :, 0] if img.shape[2] == 1 else \
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out[i] = cv2.resize(crop_to_box(img, boxes[i]),
                            (FRAME_SIZE, FRAME_SIZE),
                            interpolation=cv2.INTER_AREA)
    return out


def build_color_stream(load_frame, boxes: np.ndarray, n_frames: int
                       ) -> np.ndarray:
    """
    (N, FRAME_SIZE, FRAME_SIZE, 3) uint8 BGR, each frame cropped to its box.

    Stage A's colour counterpart to build_gray_stream — same crop/resize
    geometry, so the two streams see identical framing and differ only in
    whether the model can see colour. Kept as a separate function (not a
    `color: bool` branch inside build_gray_stream) because the two are
    consumed by completely different pipelines and must never silently
    swap into each other's callers.
    """
    out = np.empty((n_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    for i in range(n_frames):
        img = load_frame(i)
        if img is None:
            out[i] = out[i - 1] if i > 0 else 0
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        out[i] = cv2.resize(crop_to_box(img, boxes[i]),
                            (FRAME_SIZE, FRAME_SIZE),
                            interpolation=cv2.INTER_AREA)
    return out


REVIEW_FILE = "review_labels.json"


def load_review_labels(session_dir: Path, allow_unsolved: bool = False
                       ) -> tuple[list[dict], dict] | None:
    """
    review.py's corrected labels -> (moves, provenance), or None.

    `moves` comes back in the same shape prepare_session wants from
    moves.jsonl — `timestamp` and `wca_notation` — plus a `frame` key the
    caller prefers over the timestamp when it is present. See the module
    docstring for why the solved gate is a gate and not a warning.
    """
    path = session_dir / REVIEW_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    moves = data.get("moves") or []
    if not moves:
        print(f"  {session_dir.name}: {REVIEW_FILE} has no moves, skipping")
        return None

    if not data.get("reaches_solved"):
        note = data.get("completion", "unknown")
        if not allow_unsolved:
            print(f"  {session_dir.name}: review is INCOMPLETE "
                  f"({note}) — refusing to prepare it.")
            print(f"      A label set that does not reach solved has a move "
                  f"missing or misnamed,")
            print(f"      and a missing move trains the model NOT to fire "
                  f"there. Finish the review")
            print(f"      (python review.py --session {session_dir}) or pass "
                  f"--allow-unsolved.")
            return None
        print(f"  {session_dir.name}: review does NOT reach solved ({note}) "
              f"— proceeding on --allow-unsolved")

    prov = {
        "reviewer": data.get("reviewer", ""),
        "reviewed_at": data.get("reviewed_at", ""),
        "model": data.get("model", ""),
        "decoder": data.get("decoder", ""),
        "reaches_solved": bool(data.get("reaches_solved")),
        "n_confirmed": sum(1 for m in moves if m.get("confirmed")),
        "n_inserted": sum(1 for m in moves if m.get("source") == "inserted"),
        "n_renamed": sum(1 for m in moves if m.get("source") == "renamed"),
        "audit": data.get("audit", {}),
    }
    return moves, prov


def class_name_of(move: dict) -> str | None:
    """
    The move name the VISION model must predict — camera-relative.

    `camera_notation` is written by ble/backfill_camera_frame.py (and, for
    new sessions, by the recorder). `wca_notation` is the CUBE-frame name,
    which differs from it after a middle-slice turn rotates the centres:
    the cube reports a turn of the physically-top face as `D` once a slice
    has carried the bottom centre up there, and the camera correctly sees
    `U`. Training a camera model on the cube-frame name is the bug this
    exists to close — see ble/slice_frame.py.

    Falls back to `wca_notation` so a session that predates the backfill
    still prepares; `label_frame` in the stream records which was used, and
    prepare_session warns rather than letting the fallback be silent.
    """
    return move.get("camera_notation") or move.get("wca_notation")


def _wca_class_index() -> dict:
    """WCA move name -> 0-11, matching reconstruct.WCA12 / train_move_
    classifier.WCA_CLASS_NAMES exactly (imported so Stage A's onset_class
    array can never drift out of that ordering)."""
    from train_move_classifier import WCA_INDEX
    return WCA_INDEX


def prepare_session(session_dir: Path, detector, force: bool = False,
                    color: bool = False, labels: str = "ble",
                    allow_unsolved: bool = False) -> bool:
    out_path = session_dir / (STREAM_FILE_COLOR if color else STREAM_FILE)
    if out_path.exists() and not force:
        print(f"  {session_dir.name}: {out_path.name} exists, skipping "
              f"(--force to redo)")
        return True

    frames_dir = session_dir / "frames"
    frames_idx = session_dir / "frames.jsonl"
    moves_path = session_dir / "moves.jsonl"

    # -- where the labels come from (see the module docstring)
    review, label_prov, label_source = None, {}, "ble"
    if labels == "none":
        label_source, moves_src = "none", []
    elif labels in ("review", "auto"):
        review = load_review_labels(session_dir, allow_unsolved)
        if review is None and labels == "review":
            return False
    if review is not None:
        moves_src, label_prov = review
        label_source = "review"

    if not frames_dir.is_dir() or not any(frames_dir.iterdir()):
        print(f"  {session_dir.name}: no frames/ — SKIPPED.")
        print(f"      This session was postprocessed without --keep-frames, "
              f"so the continuous")
        print(f"      stream is gone. Only labeled/ move snapshots survive, "
              f"which cannot")
        print(f"      train an onset detector (no inter-move negatives). "
              f"Re-record.")
        return False

    if not frames_idx.exists():
        print(f"  {session_dir.name}: missing frames.jsonl, skipping")
        return False
    if label_source == "none" and not color:
        print(f"  {session_dir.name}: --labels none only makes sense with "
              f"--color (the grayscale stream has no inference-only use)")
        return False
    if label_source == "ble" and not moves_path.exists():
        print(f"  {session_dir.name}: missing moves.jsonl and no usable "
              f"{REVIEW_FILE}, skipping")
        return False

    frame_recs = load_jsonl(frames_idx)
    moves = (moves_src if label_source in ("review", "none")
             else load_jsonl(moves_path))

    paths = [frames_dir / r["file"] for r in frame_recs]
    keep  = [i for i, p in enumerate(paths) if p.exists()]
    if len(keep) < len(paths):
        print(f"  {session_dir.name}: {len(paths) - len(keep)} indexed frames "
              f"missing on disk, using the {len(keep)} present")
        frame_recs = [frame_recs[i] for i in keep]
        paths      = [paths[i] for i in keep]

    n = len(paths)
    if n < 2:
        print(f"  {session_dir.name}: fewer than 2 frames, skipping")
        return False

    ts  = np.array([r["ts"] for r in frame_recs], dtype=np.float64)
    dur = ts[-1] - ts[0]
    fps = n / dur if dur > 0 else 30.0

    # Crop boxes
    load_color = lambda i: cv2.imread(str(paths[i]))
    if detector is not None:
        boxes, n_det = per_frame_boxes(detector, load_color, n)
        crop_note = f"{n_det} detections -> per-frame boxes"
        if n_det == 0:
            crop_note = "cube never detected -> centered square"
    else:
        n_det = 0
        probe = cv2.imread(str(paths[0]))
        boxes = np.tile(center_square(probe.shape), (n, 1)).astype(np.int32)
        crop_note = "no detector -> centered square"
    crop_mode = "cropped" if n_det else "center-square"

    # Decode, crop, downscale. Colour mode needs the full BGR frame (Stage
    # A's class head); the deployed onset-only path reads grayscale only,
    # cheaper to decode, and is left byte-for-byte unchanged.
    if color:
        out = build_color_stream(load_color, boxes, n)
    else:
        out = build_gray_stream(
            lambda i: cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE),
            boxes, n)

    # Map each move to a frame index. A reviewed label already IS one (a
    # human placed it against this video), so it is used directly; a BLE
    # move is a wall-clock timestamp and has to be matched to the nearest
    # frame, which is where the alignment noise below comes from.
    move_ts = np.array([m["timestamp"] for m in moves], dtype=np.float64)
    if label_source == "none":
        onset_idx = np.array([], dtype=np.int32)
        move_ts = np.array([], dtype=np.float64)
        align_ms = np.array([0.0])
    elif label_source == "review":
        # Resolve by FILENAME, not by index. review.py records both, and
        # the filename is the one that survives frames being added or
        # removed between the review and this run; an index would shift
        # silently and slide every label off its onset, with nothing in
        # the .npz to show it (the WORD stays right, only its alignment
        # goes wrong — the one corruption the solved gate cannot catch).
        by_file = {r["file"]: i for i, r in enumerate(frame_recs)}
        onset_idx, unresolved = [], 0
        for m in moves:
            i = by_file.get(m.get("frame_file"))
            if i is None:
                i = int(np.clip(m.get("frame", 0), 0, n - 1))
                unresolved += 1
            onset_idx.append(i)
        onset_idx = np.array(onset_idx, dtype=np.int32)
        align_ms = np.zeros(len(onset_idx))
        if unresolved:
            print(f"      NOTE: {unresolved} reviewed label(s) had no "
                  f"resolvable frame_file; fell back to the stored index")
    else:
        onset_idx = np.clip(np.searchsorted(ts, move_ts), 0, n - 1)
        left      = np.clip(onset_idx - 1, 0, n - 1)
        take_left = np.abs(ts[left] - move_ts) < np.abs(ts[onset_idx] - move_ts)
        onset_idx = np.where(take_left, left, onset_idx).astype(np.int32)
        align_ms = np.abs(ts[onset_idx] - move_ts) * 1000

    # Onsets must be ascending for the crowding stats and for the dense
    # targets built from them; a reviewer inserting a move keeps the list
    # sorted by frame, but an inserted move can legitimately land on the
    # same frame as its neighbour.
    order = np.argsort(onset_idx, kind="stable")
    onset_idx, move_ts, align_ms = onset_idx[order], move_ts[order], align_ms[order]
    moves = [moves[i] for i in order]

    dupes = len(onset_idx) - len(np.unique(onset_idx))

    extra = {}
    n_unresolved = 0
    label_frame = "camera" if label_source == "review" else "n/a"
    if color and label_source == "none":
        extra = {"onset_class": np.array([], dtype=np.int32)}
    elif color:
        # Stage A's class head needs a WCA index per onset. Onsets whose
        # move could not be resolved to a quarter turn (half-turns the
        # tracker coalesced, or a dropped BLE decode — see ble_truth.py's
        # words_between) carry no usable label and are DROPPED from this
        # stream entirely (not kept with a sentinel class): a class head
        # cannot be supervised on "some move happened, I won't say which",
        # and keeping them in onset_idx while omitting them from
        # onset_class would desync the two arrays' meaning.
        wca_idx = _wca_class_index()
        resolved = [i for i, m in enumerate(moves)
                   if class_name_of(m) in wca_idx]
        n_unresolved = len(moves) - len(resolved)
        onset_idx = onset_idx[resolved]
        move_ts = move_ts[resolved]
        align_ms = align_ms[resolved]
        onset_class = np.array(
            [wca_idx[class_name_of(moves[i])] for i in resolved],
            dtype=np.int32)
        extra = {"onset_class": onset_class}
        n_camera_frame = sum(1 for m in moves if m.get("camera_notation"))
        label_frame = ("camera" if label_source == "review"
                       or n_camera_frame == len(moves) else "cube")

    # crop_mode / crop_margin travel WITH the stream. The classifier had
    # exactly this hole until 2026-07-25 — a mix of cropped and full-frame
    # training data with nothing in the artifact to say so — and here it
    # would be worse, because the .npz is a black box of 96x96 pixels that
    # nobody can eyeball. dataset.py refuses a mixed set on the strength of
    # these two fields.
    # label_source travels with the stream for the same reason crop_mode
    # does: a corpus that silently mixes smart-cube ground truth with
    # human-reviewed labels is a corpus nobody can characterise afterwards.
    # Unlike crop_mode this is NOT refused when mixed (see dataset.
    # check_label_regime) — growing the corpus with reviewed sessions is
    # the point — but it must be visible.
    np.savez_compressed(out_path, frames=out, onset_idx=onset_idx,
                        onset_ts=move_ts, fps=fps, name=session_dir.name,
                        crop_mode=crop_mode,
                        crop_margin=(CROP_MARGIN if n_det else -1.0),
                        n_detections=n_det, label_source=label_source,
                        label_frame=label_frame,
                        label_meta=json.dumps(label_prov), **extra)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  {session_dir.name}: {n} frames, {len(onset_idx)} onsets"
          + (f" ({n_unresolved} unresolved dropped)" if n_unresolved else "")
          + f", {fps:.1f}fps, {size_mb:.0f}MB")
    print(f"      crop: {crop_note}  [{crop_mode}]")
    if label_source == "review":
        a = label_prov.get("audit") or {}
        print(f"      labels: HUMAN REVIEW by "
              f"{label_prov.get('reviewer') or '(unnamed)'} on "
              f"{label_prov.get('reviewed_at', '?')}, "
              f"model {label_prov.get('model', '?')}")
        print(f"              {label_prov['n_confirmed']} confirmed, "
              f"{label_prov['n_renamed']} renamed, "
              f"{label_prov['n_inserted']} inserted; reaches solved: "
              f"{label_prov['reaches_solved']}")
        blind, sighted = a.get("blind_agreement"), a.get("sighted_agreement")
        if blind is not None:
            print(f"              blind audit {blind*100:.0f}% over "
                  f"{a.get('n_blind_answered', 0)} hidden move(s)"
                  + (f", sighted {sighted*100:.0f}%" if sighted is not None
                     else ""))
        elif a.get("n_in_head_unconfirmed"):
            print(f"              NOTE: {a['n_in_head_unconfirmed']} label(s) "
                  f"inside the review head were never individually confirmed")
    if crop_mode != "cropped":
        print(f"      WARNING: this stream is NOT cube-cropped. Live "
              f"inference always crops, so")
        print(f"               training on it mixes two input scales — "
              f"train.py will refuse the")
        print(f"               mixed set. Fix the detector and re-run with "
              f"--force.")
    if color and label_frame == "cube":
        print(f"      WARNING: onset_class came from the CUBE-frame "
              f"wca_notation. After a middle-slice")
        print(f"               turn that names the wrong face — see "
              f"ble/slice_frame.py. Run:")
        print(f"                   python ble/backfill_camera_frame.py "
              f"--sessions {session_dir}")
        print(f"               then re-prepare with --force.")
    if label_source == "none":
        print(f"      labels: NONE — inference-only stream. Review it, then "
              f"re-prepare with")
        print(f"              --labels review --force. Training on it is "
              f"refused (check_label_regime).")
    elif label_source == "review":
        print(f"      onset->frame alignment: exact (reviewed labels carry "
              f"frame indices)")
    else:
        print(f"      onset->frame alignment: median "
              f"{np.median(align_ms):.0f}ms, max {align_ms.max():.0f}ms")

    # Solve SPEED, reported because it drifted unmonitored for two weeks and
    # turned out to be a third regime axis next to time-of-day and recording
    # day: 1.30 -> 2.83 moves/s between 07-21 and 08-02, with adjacent-onset
    # crowding rising 0.9% -> 18%. That gap is most of what separates the
    # training corpus from the held-out set, and it was being read as
    # cross-day generalisation. See GAMEPLAN.md §1b.
    #
    # Two different numbers, and conflating them overstates the damage.
    # Crowded pairs are merely HARD. Only a crowded pair of the SAME class
    # is structurally unreachable, because CTC cannot emit a repeated label
    # without an intervening blank frame (ctc_decode.py) — that one is a
    # floor no model or decoder gets under.
    if len(onset_idx) > 1:
        order = np.argsort(onset_idx)
        oi = onset_idx[order]
        gaps = np.diff(oi)
        crowded = int((gaps <= 2).sum())
        same = 0
        if "onset_class" in extra:
            oc = extra["onset_class"][order]
            same = int(sum(1 for i, g in enumerate(gaps)
                           if g <= 2 and oc[i] == oc[i + 1]))
        rate = len(oi) / (n / fps)
        print(f"      speed: {rate:.2f} moves/s; {crowded} crowded pair(s) "
              f"<=2 frames ({100 * crowded / len(oi):.1f}%), "
              f"{same} same-class ({100 * same / len(oi):.2f}% — CTC floor)")
    if dupes:
        print(f"      NOTE: {dupes} onset(s) share a frame with another — "
              f"two-handed simultaneous")
        print(f"            moves closer together than one frame at "
              f"{fps:.0f}fps. Unresolvable here;")
        print(f"            see decode.py on the resulting recall ceiling.")
    return True


def backfill_crop_mode(session_dir: Path, samples: int = 5) -> str | None:
    """
    Determine an existing stream's crop mode from the stream ITSELF.

    Streams written before 2026-07-25 carry no crop_mode. Stamping them
    "cropped" on faith would defeat the point of recording it, so instead
    this re-derives the answer from evidence: render a few of the session's
    raw frames through the centered-square path and compare against what is
    actually stored in the .npz. A center-square stream matches almost
    exactly; a cube-cropped one is a completely different image.

    Returns the mode written, or None if the session could not be judged
    (no raw frames left on disk — then the honest answer stays "unknown").
    """
    path = session_dir / STREAM_FILE
    if not path.exists():
        return None
    data = dict(np.load(path, allow_pickle=True))
    if "crop_mode" in data:
        return str(data["crop_mode"])

    frames_dir = session_dir / "frames"
    idx_file   = session_dir / "frames.jsonl"
    if not frames_dir.is_dir() or not idx_file.exists():
        return None

    recs   = load_jsonl(idx_file)
    stored = data["frames"]
    n      = min(len(recs), len(stored))
    if n < samples:
        return None

    errs = []
    for i in np.linspace(n * 0.1, n * 0.9, samples).astype(int):
        p = frames_dir / recs[int(i)]["file"]
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cs = cv2.resize(crop_to_box(img, center_square(img.shape)),
                        (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        errs.append(float(np.abs(cs.astype(np.int16)
                                 - stored[int(i)].astype(np.int16)).mean()))
    if not errs:
        return None

    # A center-square stream reproduces the center-square render to within
    # JPEG noise; measured on 28 real cube-cropped sessions the same
    # comparison sits at 40-61 mean absolute grey levels. The gap is wide
    # enough that any threshold in between decides it.
    mode = "center-square" if float(np.median(errs)) < 5.0 else "cropped"
    data["crop_mode"]   = mode
    data["crop_margin"] = CROP_MARGIN if mode == "cropped" else -1.0
    np.savez_compressed(path, **data)
    print(f"  {session_dir.name}: crop_mode={mode} "
          f"(center-square distance {np.median(errs):.1f})")
    return mode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess sessions into detector_stream.npz")
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="Session folder(s) — supports globs: "
                             "../training_data/solve_*/")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if detector_stream.npz exists")
    parser.add_argument("--no-crop", action="store_true",
                        help="Skip cube detection; use a centered square crop")
    parser.add_argument("--backfill-crop-mode", action="store_true",
                        help="Add the crop_mode field to streams prepared "
                             "before it existed, deciding it by comparing "
                             "each stream against a centered-square render "
                             "of its own raw frames. Rebuilds nothing")
    parser.add_argument("--color", action="store_true",
                        help="MODEL_REWORK_PLAN.md Stage A: write "
                             "detector_stream_color.npz (BGR frames + "
                             "onset_class) instead of the deployed "
                             "grayscale-only detector_stream.npz. The two "
                             "are independent files; this never touches "
                             "the shipped stream.")
    parser.add_argument("--labels",
                        choices=("ble", "review", "auto", "none"),
                        default="ble",
                        help="Where onset labels come from: 'ble' = "
                             "moves.jsonl (default, the shipped behaviour), "
                             "'review' = review_labels.json from review.py, "
                             "'auto' = a reviewed file when one exists, else "
                             "BLE, 'none' = no onsets at all, for scoring an "
                             "unlabelled video before it has been reviewed "
                             "(inference only; training on it is refused). "
                             "See the module docstring.")
    parser.add_argument("--allow-unsolved", action="store_true",
                        help="Accept a reviewed session whose labels do not "
                             "reach solved. They have a move missing or "
                             "misnamed; a missing move actively trains the "
                             "model not to fire there.")
    args = parser.parse_args()

    session_dirs = [Path(p) for pattern in args.sessions
                    for p in (Path(".").glob(pattern)
                              if "*" in pattern else [Path(pattern)])]
    session_dirs = [d for d in session_dirs if d.is_dir()]
    if not session_dirs:
        sys.exit("No session directories found. Check --sessions.")

    if args.backfill_crop_mode:
        print(f"\nBackfilling crop_mode for {len(session_dirs)} session(s)...")
        seen = {}
        for d in sorted(session_dirs):
            m = backfill_crop_mode(d)
            if m:
                seen[m] = seen.get(m, 0) + 1
        print(f"\n  {dict(seen)}")
        undecided = [d.name for d in session_dirs
                     if (d / STREAM_FILE).exists()
                     and "crop_mode" not in np.load(d / STREAM_FILE,
                                                    allow_pickle=True)]
        if undecided:
            print(f"  {len(undecided)} stream(s) could not be judged (raw "
                  f"frames gone) and stay 'unknown'.")
        sys.exit(0)

    detector = None
    if not args.no_crop:
        from crop_utils import load_detector
        detector = load_detector()
        if detector is None:
            print("WARNING: cube detector unavailable (needs ultralytics + "
                  "cv/detection/detect_full_cube.pt).\n"
                  "         Falling back to centered square crops — the cube "
                  "will fill less of\n"
                  "         the input. Pass --no-crop to silence this.\n")

    print(f"\nPreparing {len(session_dirs)} session(s)"
          + (" [--color: Stage A stream]" if args.color else "")
          + (f" [labels: {args.labels}]" if args.labels != "ble" else "")
          + "...")
    ok = sum(prepare_session(d, detector, force=args.force, color=args.color,
                             labels=args.labels,
                             allow_unsolved=args.allow_unsolved)
             for d in sorted(session_dirs))
    print(f"\nDone: {ok}/{len(session_dirs)} session(s) ready.")
    if ok == 0:
        sys.exit("No sessions prepared — nothing to train on.")
