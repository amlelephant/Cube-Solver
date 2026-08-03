"""
live_detect.py

End-to-end move reading: detector finds WHEN moves happen, classifier says
WHICH move each one was.

Replaces ble/live_test.py. The difference is the segmentation step:

    live_test.py     global frame-delta threshold + a STILL/MOVING state
                     machine, closing a window only after 133ms of confirmed
                     stillness. During a real solve the cube never rests
                     (median gap 420ms, p10 180ms), so fast moves merged into
                     one window, and cube rotations — which produce a LARGER
                     delta than a layer turn — fired phantom windows.

    this file        buffers the whole solve, then scores every frame with
                     the trained onset detector and picks peaks. No stillness
                     required, and rotations are learned negatives rather
                     than something a threshold has to be tricked past.

Because detection is a peak in a score curve rather than an edge in a state
machine, it needs future context — so this is buffer-then-analyse, not
frame-by-frame streaming. A 54s solve is ~250MB of JPEG in RAM and analyses
in a few seconds.

Two modes:

    live      capture from the webcam, SPACE to start/stop, then analyse
    replay    re-run a recorded session and score it against BLE ground
              truth. Use this to check the wiring without a cube in hand —
              it reports onset F1 AND per-move classification accuracy.

Usage:
    python live_detect.py
    python live_detect.py --camera 1 --classifier ../move_classifier_wca.pt
    python live_detect.py --session ../training_data/solve_20260721_102711/
    python live_detect.py --session ../training_data/solve_*/ --review
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except ImportError:
    sys.exit("PyTorch not installed. Run: pip install torch torchvision")

_BLE_DIR = Path(__file__).resolve().parents[1]
if str(_BLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLE_DIR))

from dataset import ArrayStream                                # noqa: E402
from model import build_model, score_stream                    # noqa: E402
from decode import (peak_pick, onset_windows, match_onsets,    # noqa: E402
                    format_metrics, align_sequences, MIN_SEP, TOLERANCE)
from prepare_data import (per_frame_boxes, build_gray_stream,   # noqa: E402
                          crop_to_box, center_square)
from train_move_classifier import (FRAME_ORDER, predict_probs,  # noqa: E402
                                   inference_crop_regime)

# checkpoints/move_detector_all28.pt — 28 frame-bearing sessions (adds 20260723 and
#                      20260724, neither of which the previous checkpoints/move_detector.pt
#                      had ever seen), --holdout session with one name per
#                      recording day (20260721, 20260722, 20260723, 20260724):
#                      93.9% F1 / 95.8% recall aggregate, tuned threshold
#                      0.25. Replaced checkpoints/move_detector.pt (a same-day "final fit"
#                      on 12 sessions with no validation of its own) on
#                      2026-07-24 after it scored only 89.8% F1 across all 16
#                      sessions from 20260723/20260724 it had never seen.
# move_classifier_all39_cropped.pt — all 39 sessions, ALL of them cropped
#                      to the cube box before diffing (cache_crops.py run on
#                      every session first). 94.6% on the same 5-session
#                      cross-environment holdout used for all23/all27/all39.
#                      This superseded move_classifier_all39.pt within the
#                      same day: all39 trained on a MIX of cropped (the 23
#                      recorded sessions) and full-frame (the 16 live
#                      sessions from 20260723/20260724, cache_crops.py had
#                      never been run on them) — but live_detect.py's
#                      inference path ALWAYS crops, so every one of those 16
#                      sessions trained on a different scale than inference
#                      ever shows the model. Fixing that (not "more live
#                      data", which was the story at the time) is most of
#                      what actually closed the live-regime gap. Per-session
#                      on that holdout, evaluated fairly (cropped) for all four:
#                        20260720   20260721   20260722   20260723   20260724
#           all23        97.2%      93.2%      90.4%      90.8%      84.1%
#           all27        95.8%      88.4%      83.2%      81.6%      73.0%
#           all39        95.8%      89.7%      91.0%      93.9%      78.6%
#           all39_cropped 97.2%     94.5%      94.0%      93.9%      94.4%
#                      Note all23 alone — same weights, no retraining —
#                      scores 90.6% aggregate here once evaluated on cropped
#                      images; the 61.5%/7-9% numbers once reported for it
#                      were mostly a broken eval, not a broken model. See
#                      ../move_detector/README.md "Correction, same day"
#                      for the full story and the end-to-end (not just
#                      isolated-classifier) numbers.
# move_classifier_all39_jitter.pt — the SAME 39 sessions and the SAME five
#                      named holdout sessions as all39_cropped, retrained
#                      with --anchor-jitter so the augmentation is the only
#                      variable. 94.7% on the canonical holdout, i.e. no
#                      change in-distribution (all39_cropped: 94.6%) — the
#                      gain is entirely in the live regime, which is the
#                      point.
#
#                      Measured on the 10 saved live takes NEITHER model has
#                      seen, scoring predictions by timestamp match
#                      (metric_audit.py; the detector is untouched, so both
#                      models are scored on an identical set of found moves):
#
#                                          all39_cropped   all39_jitter
#                        classifier of found    81.5%          85.8%
#                        free solves only       78.6%          83.2%
#                        end-to-end             69.5%          73.2%
#
#                      McNemar on the paired moves: 54 fixed, 29 broken,
#                      p=0.008. It does NOT help on the two takes whose
#                      detector did worst (134516_solve at 69% recall,
#                      100142_solve with 14 phantoms) — a bad onset stream
#                      is a different failure and jitter does not touch it.
#
#                      WHY jitter: window_audit.py showed classifier accuracy
#                      falls with how many of the 5 window slots differ from
#                      the trainer's own choice, while the same moves on the
#                      canonical window stay flat — and inference CANNOT
#                      reproduce that window, because its anchor is a frame
#                      index while training used the sub-frame BLE timestamp.
#                      Fixing the frame selection at inference was tried
#                      first and measured p=0.40. So the model is trained on
#                      the shifted windows instead. Live anchor offsets are
#                      ~2x wider than recorded ones (52% vs 72% exact), which
#                      is why this shows up live and not in replay.
DETECTOR_PATH  = "checkpoints/move_detector_all28.pt"
CLASSIFIER_PATH = "../move_classifier_all39_jitter.pt"
JPEG_QUALITY   = 85


# ---------------------------------------------------------------------------
# Analysis — shared by live capture and offline replay
# ---------------------------------------------------------------------------

# Warn once per (classifier, regime) rather than once per session: the
# replay loops call analyse() dozens of times and a repeated banner trains
# you to skip it, which is how the crop mismatch survived a whole day of
# training logs that were technically reporting it.
_CROP_WARNED: set = set()


def _warn_crop_mismatch(classifier: str, cropped: bool) -> None:
    """
    Loudly flag a classifier being fed an input scale it never trained on.

    The classifier checkpoint records `crop_regime` (train_move_classifier.py).
    This is the inference-side half of that contract: if a cube-cropped model
    is about to be handed centered-square frames — because ultralytics or
    detect_full_cube.pt is missing, or because the cube was never detected in
    this take — the resulting accuracy describes the mismatch, not the model.
    """
    regime = inference_crop_regime(classifier)
    key = (str(classifier), regime, cropped)
    if key in _CROP_WARNED:
        return
    _CROP_WARNED.add(key)

    if regime is None:
        if not cropped:
            print(f"  WARNING: no cube crop this run, and {Path(classifier).name} "
                  f"predates crop\n           provenance (retrain to record "
                  f"it). If it was trained cropped — the\n           deployed "
                  f"ones were — these predictions are on the wrong scale.")
        return
    if regime == "cropped" and not cropped:
        print(f"\n  WARNING: {Path(classifier).name} was trained on CUBE CROPS "
              f"but this run has no\n           crop (centered square "
              f"fallback). Expect a large, misleading accuracy\n           "
              f"drop — an unchanged checkpoint measured 61.5% vs 90.6% purely "
              f"on this.\n           Install ultralytics + "
              f"cv/detection/detect_full_cube.pt.\n")
    elif regime in ("full-frame", "mixed") and cropped:
        print(f"\n  WARNING: {Path(classifier).name} was trained "
              f"{regime.upper()} but is being fed cube\n           crops. "
              f"Retrain it after `python cache_crops.py --sessions "
              f"training_data/solve_*/`.\n")


def analyse(load_color, n_frames: int, fps: float, detector, det_model,
            device, threshold: float, min_sep: int, classifier: str,
            verbose: bool = True, frame_times=None,
            peak_threshold: float | None = None) -> dict:
    """
    Full pipeline over a buffered stream.

    `load_color` is i -> BGR frame. Everything downstream runs through the
    same helpers prepare_data.py uses offline, so what the detector sees
    here is byte-identical in construction to what it trained on.

    `frame_times` is the capture timestamp of each frame, in seconds and
    ascending — index-aligned with what `load_color` returns, so a resampled
    stream must pass the RESAMPLED timeline. It lets the classifier windows
    be carved the way postprocess_session.py carved the training ones
    (nearest frame in time) instead of by index arithmetic against a global
    mean fps. Measured effect on accuracy: +0.3 points, NOT significant —
    see decode.window_from_anchor for why the real residual is the anchor
    rather than the frame selection. Omitting it degrades to the legacy
    behaviour.

    `peak_threshold`, when given (and lower than `threshold`), is used for
    peak-picking INSTEAD of `threshold` — it widens the onset list to
    include sub-threshold peaks the model saw but the deployed operating
    point discards (see decode.py's miss-bucket diagnostic: "subthreshold"
    misses have a real peak, just below the F-beta-tuned cutoff). Every
    extra onset still gets classified normally; `threshold` is still
    reported on each move's `score` field so a downstream cost model (see
    reconstruct.score_del_costs's `candidate_threshold` ramp) can price a
    weak candidate near-free to delete instead of at the flat onset cost —
    this is PATH_TO_VERIFICATION.md's D1 lever, and it only ever ADDS
    onsets relative to the `threshold`-only call. None (the default)
    reproduces the exact prior behaviour.
    """
    t0 = time.time()
    pick_threshold = threshold if peak_threshold is None \
        else min(peak_threshold, threshold)

    cropped = detector is not None
    if detector is not None:
        boxes, n_det = per_frame_boxes(detector, load_color, n_frames)
        # n_det == 0 means per_frame_boxes already fell back to a centered
        # square for the WHOLE stream. That used to print as "0 cube
        # detections -> per-frame boxes", which reads like a successful crop.
        cropped = n_det > 0
        crop_note = (f"{n_det} cube detections -> per-frame boxes" if cropped
                     else "cube NEVER detected -> centered square crop "
                          "(both models trained on cube crops!)")
    else:
        probe = load_color(0)
        boxes = np.tile(center_square(probe.shape), (n_frames, 1)).astype(np.int32)
        crop_note = "no detector -> centered square crop"
    if verbose:
        print(f"  crop:   {crop_note}  ({time.time()-t0:.1f}s)")
        _warn_crop_mismatch(classifier, cropped)

    gray = build_gray_stream(
        lambda i: (lambda f: None if f is None else
                   (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f)
                   )(load_color(i)),
        boxes, n_frames)

    stream = ArrayStream(gray, name="live", fps=fps)
    scores = score_stream(det_model, stream, device)
    onsets = peak_pick(scores, threshold=pick_threshold, min_sep=min_sep)
    if verbose:
        tag = (f"{pick_threshold} (candidate; operating point {threshold})"
               if pick_threshold != threshold else str(threshold))
        print(f"  detect: {len(onsets)} onsets at threshold {tag} "
              f"({time.time()-t0:.1f}s)")

    windows = onset_windows(onsets, fps, n_frames, frame_times)
    if verbose and frame_times is None:
        print(f"  note:   no frame timestamps — classifier windows fall back "
              f"to index arithmetic")

    moves = []
    for o, w in zip(onsets, windows):
        idxs = [w[k] for k in FRAME_ORDER]
        # One shared box for the whole move window — per-frame boxes would
        # inject detector jitter into the classifier's temporal diffs as
        # fake global motion (crop_utils' rule).
        box = np.median(boxes[idxs], axis=0).astype(int)
        frames = []
        for i in idxs:
            f = load_color(i)
            frames.append(crop_to_box(f, box) if f is not None else None)
        if any(f is None for f in frames):
            continue
        probs, class_names = predict_probs(frames, classifier)
        cls = int(np.argmax(probs))
        moves.append({"frame": int(o), "time": float(o / fps),
                      "move": class_names[cls], "conf": float(probs[cls]),
                      "probs": [float(p) for p in probs],
                      "score": float(scores[o]),
                      "window": idxs, "box": box.tolist()})

    if verbose:
        print(f"  classify: {len(moves)} moves ({time.time()-t0:.1f}s total)")

    return {"scores": scores, "onsets": onsets, "moves": moves, "boxes": boxes,
            "class_names": class_names if moves else None,
            "cropped": cropped,
            "fps": fps, "n_frames": n_frames}


def print_sequence(moves: list[dict], fps: float):
    if not moves:
        print("\n  No moves detected.")
        return
    print(f"\n  {len(moves)} moves detected:\n")
    seq = " ".join(m["move"] for m in moves)
    for i in range(0, len(seq), 72):
        print(f"    {seq[i:i+72]}")

    weak = [m for m in moves if m["conf"] < 0.6]
    print(f"\n  mean classifier confidence: "
          f"{np.mean([m['conf'] for m in moves])*100:.0f}%")
    if weak:
        print(f"  {len(weak)} move(s) below 60% confidence:")
        for m in weak[:8]:
            print(f"    t={m['time']:5.1f}s  frame {m['frame']:5d}  "
                  f"{m['move']:<3}  conf {m['conf']*100:3.0f}%  "
                  f"onset score {m['score']:.2f}")


def report_by_gap(matched_gt, gt_onset, fps,
                  edges=(2, 3, 4, 6, 10, 10 ** 9)):
    """
    Recall stratified by how CROWDED each move is — the gap in frames to its
    nearest neighbouring move, on either side.

    Aggregate F1 cannot measure the fast-burst problem and will mislead you
    if you try. Moves closer than 4 frames are only ~6% of the recorded
    data, so a change that perfectly resolved every one of them moves
    overall F1 by ~3 points — inside the noise between two training runs —
    while completely changing behaviour during algorithm execution, which
    is where a solver actually spends its time.

    So: bucket by crowding and read the tail. A detector can be 95% overall
    and 0% on the sub-3-frame bucket, and those two numbers describe
    completely different products.

    Nearest neighbour on EITHER side, not just the previous move, because
    a peak is lost to whichever adjacent onset merges with it first.
    """
    gt_onset = np.asarray(sorted(gt_onset))
    if len(gt_onset) < 2:
        return
    matched = set(int(x) for x in matched_gt)

    stats = {}
    for i, o in enumerate(gt_onset):
        prev = o - gt_onset[i - 1] if i > 0 else 10 ** 9
        nxt  = gt_onset[i + 1] - o if i < len(gt_onset) - 1 else 10 ** 9
        gap  = int(min(prev, nxt))
        edge = next(e for e in edges if gap < e)
        t, c = stats.get(edge, (0, 0))
        stats[edge] = (t + 1, c + (int(o) in matched))

    print(f"\n  RECALL BY CROWDING - gap to the nearest neighbouring move:")
    print(f"    {'gap':<10} {'moves':>6} {'found':>6} {'recall':>8}   "
          f"{'~moves/sec':>10}")
    lo = 0
    for e in edges:
        if e not in stats:
            lo = e
            continue
        total, corr = stats[e]
        label = f"<{e}f" if lo == 0 else (f">={lo}f" if e > 10 ** 8
                                          else f"{lo}-{e-1}f")
        tps = f"{fps/lo:.0f}" if lo else f">{fps/e:.0f}"
        bar = "#" * int(corr / total * 20)
        print(f"    {label:<10} {total:>6} {corr:>6} "
              f"{corr/total*100:>7.1f}%   {tps:>10}  {bar}")
        lo = e


# ---------------------------------------------------------------------------
# Prescribed-scramble accuracy test
# ---------------------------------------------------------------------------
#
# Ground truth without BLE. You are shown a scramble, you perform it, and the
# predicted sequence is aligned against the one you were told to do.
#
# This is the only way to measure a cube that has no Bluetooth in it. Every
# other number in this repo is supervised by smart-cube move events, which
# means every number in this repo describes one specific cube. A prescribed
# scramble is cube-agnostic: the label comes from the instruction, not the
# hardware.
#
# What it gives up: BLE supplies per-move TIMESTAMPS, so it can score onset
# placement to the frame. A prescribed scramble only fixes the ORDER, so
# scoring has to be sequence alignment (below) rather than onset matching.
# That is enough to separate missed moves from misclassified ones, which is
# the split that actually drives what to fix next.

FACES = ("U", "D", "R", "L", "F", "B")
AXIS  = {"U": 0, "D": 0, "R": 1, "L": 1, "F": 2, "B": 2}


def generate_scramble(n: int, seed: int | None = None) -> list[str]:
    """
    Random QUARTER-TURN scramble of n moves.

    Quarter turns only, deliberately. A WCA competition scramble contains
    half turns (R2), but the classifier's 12 classes are exactly the
    quarter turns — so an R2 would be scored wrong no matter how well the
    pipeline performed, and the measurement would be reporting a fixture
    artifact instead of a model property. This is a test fixture, not a
    competition scramble.

    Constraints:
      * no two consecutive moves on the same face — R R' is a no-op, and
        R R is a half turn expressed as two moves, which people physically
        perform as one motion and the detector would see as one onset.
      * no three consecutive on the same axis — R L R is redundant.
    """
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        face = rng.choice(FACES)
        if out and face == out[-1][0]:
            continue
        if len(out) >= 2 and AXIS[face] == AXIS[out[-1][0]] == AXIS[out[-2][0]]:
            continue
        out.append(face + rng.choice(("", "'")))
    return out


def format_scramble(scramble: list[str], per_row: int = 8) -> list[str]:
    """The scramble as short display lines, numbered so you can keep place."""
    lines = []
    for i in range(0, len(scramble), per_row):
        chunk = scramble[i:i + per_row]
        lines.append(f"{i+1:>2}-{i+len(chunk):<2} " +
                     " ".join(f"{m:<2}" for m in chunk))
    return lines


def report_scramble(truth: list[str], moves: list[dict],
                    truth_label: str = "prescribed",
                    title: str = "SCRAMBLE ACCURACY") -> dict:
    """
    Print the alignment and the per-stage accuracy breakdown.

    `truth_label` names where the ground truth came from, because that is
    not always the printed scramble: with a smart cube attached it is the
    move list the cube actually felt, which is a different — and stronger —
    claim about what these numbers mean.
    """
    pred = [m["move"] for m in moves]
    ops  = align_sequences(truth, pred)

    counts = {k: sum(1 for o, _, _ in ops if o == k)
              for k in ("ok", "sub", "miss", "phantom")}
    n_truth  = len(truth)
    detected = counts["ok"] + counts["sub"]

    print(f"\n{'='*66}")
    print(f"  {title} - {n_truth} {truth_label} moves, "
          f"{len(pred)} predicted")
    print(f"{'='*66}\n")

    mark = {"ok": " ", "sub": "X", "miss": "-", "phantom": "+"}
    print(f"  {'#':>3}  {'expected':<9} {'predicted':<9}  conf")
    pi = ti = 0
    for op, t, p in ops:
        conf = ""
        if p is not None:
            conf = f"{moves[pi]['conf']*100:3.0f}%"
            pi += 1
        if t is not None:
            ti += 1
        idx = f"{ti}" if t is not None else ""
        print(f"  {idx:>3}{mark[op]} {t or '.':<9} {p or '.':<9}  {conf}")

    print(f"\n  {'-'*44}")
    print(f"  correct          {counts['ok']:>4}   "
          f"({counts['ok']/n_truth*100:5.1f}% of {truth_label})")
    print(f"  misclassified    {counts['sub']:>4}   X  detected, named wrong")
    print(f"  missed           {counts['miss']:>4}   -  detector never fired")
    print(f"  phantom          {counts['phantom']:>4}   +  detector fired "
          f"spuriously")
    print(f"  {'-'*44}")

    det_recall = detected / n_truth if n_truth else 0.0
    cls_acc    = counts["ok"] / detected if detected else 0.0
    print(f"  detector recall      {det_recall*100:5.1f}%   "
          f"({detected}/{n_truth} moves found)")
    print(f"  classifier accuracy  {cls_acc*100:5.1f}%   "
          f"({counts['ok']}/{detected} of those named right)")
    print(f"  END-TO-END           {counts['ok']/n_truth*100:5.1f}%   "
          f"({counts['ok']}/{n_truth})")
    if counts["phantom"]:
        print(f"  precision            "
              f"{detected/len(pred)*100:5.1f}%   "
              f"({counts['phantom']} spurious of {len(pred)} predicted)")
    print(f"\n  exact sequence: "
          f"{'YES' if pred == truth else 'no'}")

    if counts["miss"] > counts["sub"] and counts["miss"] > 2:
        print(f"\n  Dominant failure is MISSED moves - that is the detector, "
              f"not the\n  classifier. Try a lower --threshold before "
              f"touching either model.")
    elif counts["sub"] > counts["miss"] and counts["sub"] > 2:
        print(f"\n  Dominant failure is MISCLASSIFICATION - the detector is "
              f"finding the\n  moves and the classifier is naming them wrong.")
    if counts["phantom"] > 2:
        print(f"\n  {counts['phantom']} phantom move(s) — the threshold may "
              f"be too LOW, or cube\n  rotations/regrips are firing the "
              f"detector.")

    return {"truth": truth, "pred": pred, "ops": ops, "counts": counts}


# ---------------------------------------------------------------------------
# Live capture
# ---------------------------------------------------------------------------

def capture(camera: int, preview_every: int = 2,
            overlay: list[str] | None = None,
            status_fn=None) -> tuple[list, np.ndarray]:
    """
    Buffer JPEG-encoded frames from the webcam. Returns (frames, timestamps).

    Capture rate is the thing that matters here. The detector trained
    exclusively on 30fps streams, and measured offline it loses recall fast
    below that — at 20fps one held-out session drops from 100% to 90%
    recall, at 15fps to 85%. Missed moves, not phantom ones, which is
    exactly what "found 11 of 25" looks like.

    So JPEG encoding runs on a worker thread rather than inline: encoding a
    1280x720 frame costs ~5-10ms, which on the capture thread alone is
    enough to pull 30fps down to low 20s, and the effect worsens as the
    buffer grows. The preview is also throttled, since imshow is not free
    either. Same decoupling record_training.py already uses.

    Timestamps come back per frame rather than as a single average, so
    analysis can detect and correct for a rate that drifted mid-capture.

    `status_fn` is an optional zero-arg callable returning one short line to
    draw under the REC indicator. It exists so a BLE ground-truth logger can
    show its move count live: a cube that silently stopped reporting and a
    cube that is reporting fine look identical until the take is over and
    the truth is empty, which is an expensive way to find out.
    """
    import queue
    import threading

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    buf, stamps = [], []
    work: "queue.Queue" = queue.Queue(maxsize=240)
    stop = threading.Event()

    def encoder():
        while True:
            try:
                item = work.get(timeout=0.2)
            except queue.Empty:
                if stop.is_set():
                    return
                continue
            ts, frame = item
            ok, enc = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                buf.append(enc)
                stamps.append(ts)

    thread = threading.Thread(target=encoder, daemon=True)
    thread.start()

    recording, dropped, shown = False, 0, 0
    recent: list[float] = []
    print("\n  SPACE = start/stop recording    Q = quit\n")

    while True:
        ok, frame = cap.read()
        now = time.time()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        if recording:
            try:
                work.put_nowait((now, frame))
            except queue.Full:
                dropped += 1          # encoder fell behind; frame is lost
            recent.append(now)
            if len(recent) > 45:
                recent.pop(0)

        shown += 1
        if shown % preview_every == 0:
            disp = frame.copy()
            if recording:
                live_fps = ((len(recent) - 1) / (recent[-1] - recent[0])
                            if len(recent) > 1 and recent[-1] > recent[0] else 0)
                warn = live_fps < 28
                col = (0, 165, 255) if warn else (255, 255, 255)
                cv2.circle(disp, (28, 30), 11, (0, 0, 220), -1)
                cv2.putText(disp, f"REC  {len(buf)}f  "
                                  f"{now - recent[0] if recent else 0:.0f}s  "
                                  f"{live_fps:.0f}fps",
                            (48, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2,
                            cv2.LINE_AA)
                if warn:
                    cv2.putText(disp, "LOW FPS - moves will be missed",
                                (48, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 165, 255), 1, cv2.LINE_AA)
                if status_fn is not None:
                    cv2.putText(disp, status_fn(), (48, 84 if warn else 64),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (120, 255, 120), 1, cv2.LINE_AA)
            else:
                cv2.putText(disp, "SPACE to record" if not buf
                            else f"SPACE to re-record ({len(buf)}f held)",
                            (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 220, 60), 2, cv2.LINE_AA)

            # Scramble stays on screen while recording — it is the ground
            # truth you are being asked to reproduce, so it has to be
            # readable without looking away from the cube.
            if overlay:
                y = disp.shape[0] - 18 * len(overlay) - 14
                cv2.rectangle(disp, (10, y - 20),
                              (30 + 13 * max(len(s) for s in overlay),
                               y + 18 * len(overlay) - 4),
                              (0, 0, 0), -1)
                for k, line in enumerate(overlay):
                    cv2.putText(disp, line, (20, y + 18 * k),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("CubeArena - solve capture", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if recording:
                break
            buf.clear()
            stamps.clear()
            recent.clear()
            recording = True
        elif key in (ord("q"), 27):
            buf.clear()
            break

    stop.set()
    thread.join(timeout=5.0)
    cap.release()
    cv2.destroyAllWindows()

    if dropped:
        print(f"  WARNING: {dropped} frame(s) dropped — the encoder could "
              f"not keep up.")
    return list(buf), np.array(stamps, dtype=np.float64)


def report_capture_rate(stamps: np.ndarray) -> float:
    """
    Print capture-rate diagnostics and return the mean fps.

    Reports the worst one-second window as well as the mean, because a
    capture that averages 29fps but stalls to 12 for a stretch loses moves
    in that stretch — and the average alone hides it.
    """
    if len(stamps) < 2:
        return 30.0
    dur = stamps[-1] - stamps[0]
    mean_fps = (len(stamps) - 1) / dur if dur > 0 else 30.0

    gaps = np.diff(stamps)
    win = max(2, int(round(mean_fps)))
    worst = min(((win) / (stamps[i + win] - stamps[i])
                 for i in range(0, len(stamps) - win)), default=mean_fps)

    print(f"  capture: {len(stamps)} frames, {dur:.1f}s, "
          f"mean {mean_fps:.1f}fps, worst 1s window {worst:.1f}fps, "
          f"max gap {gaps.max()*1000:.0f}ms")
    if mean_fps < 28 or worst < 24:
        print(f"  WARNING: capture ran below the 30fps the detector was "
              f"trained on.")
        print(f"           Measured offline, recall falls to ~90% at 20fps "
              f"and ~85% at 15fps —")
        print(f"           it MISSES moves rather than inventing them. "
              f"Close other GPU/CPU load,")
        print(f"           or lower the capture resolution.")
    return mean_fps


def resample_to_30fps(stamps: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Indices that map the captured frames onto a uniform 30fps timeline
    (nearest neighbour in time).

    This adds no information — dropped frames stay dropped — but it
    restores the time SCALE the detector's temporal filters expect. Offline
    it recovered most of the recall lost at 20fps on one held-out session
    and was roughly neutral on another, so it is applied only when capture
    actually fell short.
    """
    t = stamps - stamps[0]
    grid = np.arange(0, t[-1], 1.0 / 30.0)
    idx = np.abs(t[None, :] - grid[:, None]).argmin(1)
    return idx, 30.0


# ---------------------------------------------------------------------------
# Missed-onset diagnosis
# ---------------------------------------------------------------------------

INVISIBLE = 0.10      # peak score below this = the model saw nothing


def _is_local_max(scores: np.ndarray, i: int) -> bool:
    """The same local-maximum test peak_pick uses, for one index."""
    if i <= 0 or i >= len(scores) - 1:
        return False
    return scores[i] >= scores[i - 1] and scores[i] > scores[i + 1]


def report_missed(missed_gt, scores, gt_onset, gt_labels, fps,
                  threshold: float, radius: int = 3, show: int = 12):
    """
    For every ground-truth onset the decoder missed, report the model's peak
    score near it.

    Which bucket a miss falls into determines the fix, and the three fixes
    are completely different — so a bare recall number cannot tell you what
    to do next:

      invisible    peak < 0.10. The model does not see the turn at all. No
                   threshold recovers these; it needs training data that
                   looks like the footage being tested (different lighting,
                   different day, or — the case this was written to check —
                   turns performed slowly, which the speedsolving training
                   set barely contains).

      subthreshold 0.10 <= peak < threshold. The model DID see it and the
                   operating point is simply too high. Lowering --threshold
                   recovers these immediately, at some cost in precision.

      suppressed   peak >= threshold. A qualifying peak existed but the
                   decoder dropped it — either it was not a local maximum,
                   or min_sep's refractory window killed it, or a neighbour
                   claimed the match. Two-handed simultaneous moves land
                   here and are irreducible (see decode.py).

    A miss bucketed subthreshold/suppressed is a decoder problem. A miss
    bucketed invisible is a data problem. Do not spend a week on the second
    when the histogram says the first.
    """
    if len(missed_gt) == 0:
        print("\n  MISSED       none — every ground-truth onset was matched.")
        return

    gt_onset = np.asarray(gt_onset)
    rows, buckets = [], {"invisible": 0, "subthreshold": 0, "suppressed": 0}

    for o in missed_gt:
        lo, hi = max(0, o - radius), min(len(scores), o + radius + 1)
        win  = scores[lo:hi]
        peak = float(win.max())
        at   = int(lo + win.argmax())

        if peak < INVISIBLE:
            bucket = "invisible"
        elif peak < threshold:
            bucket = "subthreshold"
        else:
            bucket = "suppressed"
        buckets[bucket] += 1

        # Gap to the nearest OTHER ground-truth onset: a miss crowded within
        # min_sep of a neighbour is the two-handed case, not a model failure.
        others = gt_onset[gt_onset != o]
        gap_f  = int(np.abs(others - o).min()) if len(others) else -1

        j     = int(np.argmin(np.abs(gt_onset - o)))
        label = gt_labels[j] if j < len(gt_labels) and gt_labels[j] else "?"
        rows.append((o, label, peak, at - o, bucket, gap_f))

    n = len(rows)
    print(f"\n  MISSED       {n} ground-truth onset(s) unmatched — "
          f"peak score within ±{radius}f of each:")
    print(f"    {'frame':>6} {'t(s)':>6} {'move':<5} {'peak':>6} "
          f"{'@':>4} {'nbr':>5}  bucket")
    for o, label, peak, off, bucket, gap_f in rows[:show]:
        nbr = f"{gap_f}f" if gap_f >= 0 else "-"
        print(f"    {o:>6} {o/fps:>6.1f} {label:<5} {peak:>6.2f} "
              f"{off:>+4d} {nbr:>5}  {bucket}")
    if n > show:
        print(f"    ... and {n - show} more")

    print(f"\n    invisible    {buckets['invisible']:>3}  "
          f"(peak < {INVISIBLE:.2f} — model sees nothing; needs DATA)")
    print(f"    subthreshold {buckets['subthreshold']:>3}  "
          f"(peak < {threshold:.2f} — operating point too high; "
          f"lower --threshold)")
    print(f"    suppressed   {buckets['suppressed']:>3}  "
          f"(peak >= {threshold:.2f} — decoder dropped a valid peak; "
          f"min_sep or a crowded neighbour)")

    # What a lower operating point would buy. This counts peaks that clear
    # the candidate threshold AND are local maxima, which is what peak_pick
    # requires — but it ignores min_sep, so treat it as an upper bound.
    recoverable = [peak for (o, _, peak, off, _, _) in rows
                   if _is_local_max(scores, o + off)]
    if recoverable:
        print(f"\n    Recoverable by lowering the threshold "
              f"(upper bound — ignores min_sep):")
        for t in (0.50, 0.40, 0.30, 0.20, 0.10):
            if t >= threshold:
                continue
            k = sum(1 for peak in recoverable if peak >= t)
            if k:
                print(f"      threshold {t:.2f}  ->  +{k} of {n} "
                      f"({k/n*100:.0f}% of the misses)")


# ---------------------------------------------------------------------------
# Offline replay against BLE ground truth
# ---------------------------------------------------------------------------

def replay(session_dir: Path, detector, det_model, device, args):
    frames_dir = session_dir / "frames"
    npz        = session_dir / "detector_stream.npz"
    if not frames_dir.is_dir():
        print(f"  {session_dir.name}: no frames/ — cannot replay (needs the "
              f"raw stream, not just labeled/)")
        return None
    if not npz.exists():
        print(f"  {session_dir.name}: no detector_stream.npz — run "
              f"prepare_data.py first")
        return None

    recs  = [json.loads(l) for l in open(session_dir / "frames.jsonl") if l.strip()]
    paths = [frames_dir / r["file"] for r in recs]
    keep  = [i for i, p in enumerate(paths) if p.exists()]
    paths = [paths[i] for i in keep]
    ts    = np.array([recs[i]["ts"] for i in keep])
    n     = len(paths)
    fps   = n / (ts[-1] - ts[0]) if n > 1 else 30.0

    data     = np.load(npz, allow_pickle=True)
    gt_onset = data["onset_idx"].astype(int)
    gt_moves = [json.loads(l) for l in open(session_dir / "moves.jsonl") if l.strip()]
    gt_labels = [m.get("wca_notation") for m in gt_moves]

    print(f"\n  {session_dir.name}: {n} frames, {fps:.1f}fps, "
          f"{len(gt_onset)} ground-truth onsets")

    cache = {}

    def load_color(i):
        if i not in cache:
            if len(cache) > 600:      # bound memory on long sessions
                cache.clear()
            cache[i] = cv2.imread(str(paths[i]))
        return cache[i]

    res = analyse(load_color, n, fps, detector, det_model, device,
                  args.threshold, args.min_sep, args.classifier,
                  frame_times=ts)

    # Onset detection quality
    m = match_onsets(res["onsets"], gt_onset, res["scores"], args.tolerance)
    print(f"\n  {format_metrics(m, 'ONSETS')}")

    # Why each miss was missed — data problem or decoder problem
    report_missed(m["missed_gt"], res["scores"], gt_onset, gt_labels,
                  fps, args.threshold)
    report_by_gap(m["matched_gt"], gt_onset, fps)

    # Move classification quality on the onsets that matched
    correct = compared = 0
    wrong = []
    for mv in res["moves"]:
        d = np.abs(gt_onset - mv["frame"])
        j = int(np.argmin(d))
        if d[j] <= args.tolerance and j < len(gt_labels) and gt_labels[j]:
            compared += 1
            if gt_labels[j] == mv["move"]:
                correct += 1
            else:
                wrong.append((mv["frame"], gt_labels[j], mv["move"], mv["conf"]))

    if compared:
        print(f"  MOVES        {correct}/{compared} correct "
              f"({correct/compared*100:.1f}%) on matched onsets")
        if wrong:
            print(f"\n  first {min(10, len(wrong))} misclassifications "
                  f"(frame: truth -> predicted, conf):")
            for f, t, p, c in wrong[:10]:
                print(f"    {f:5d}: {t:<3} -> {p:<3}  {c*100:3.0f}%")

    # End-to-end: fraction of the true sequence recovered exactly in order
    pred_seq = [mv["move"] for mv in res["moves"]]
    true_seq = [l for l in gt_labels if l]
    exact = pred_seq == true_seq
    print(f"\n  SEQUENCE     {len(pred_seq)} predicted vs {len(true_seq)} true"
          f"  — exact match: {'YES' if exact else 'no'}")

    return {"session": session_dir.name, "onsets": m,
            "moves_correct": correct, "moves_compared": compared,
            "exact": exact, "res": res}


# ---------------------------------------------------------------------------
# Review UI
# ---------------------------------------------------------------------------

def review(res: dict, load_color):
    """Step through detected moves showing each one's classifier window."""
    moves = res["moves"]
    if not moves:
        return
    print("\n  Review: any key = next, B = back, Q = quit")
    i = 0
    while 0 <= i < len(moves):
        mv = moves[i]
        tiles = []
        for j in mv["window"]:
            f = load_color(j)
            if f is None:
                continue
            tiles.append(cv2.resize(crop_to_box(f, mv["box"]), (180, 180)))
        if not tiles:
            break
        sheet = np.hstack(tiles)
        cv2.putText(sheet, f"{i+1}/{len(moves)}  {mv['move']}  "
                           f"{mv['conf']*100:.0f}%  t={mv['time']:.1f}s",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 120), 2, cv2.LINE_AA)
        cv2.imshow("move review", sheet)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        i += -1 if key == ord("b") else 1
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.detector, map_location=device)
    det_model = build_model(device)
    det_model.load_state_dict(ckpt["state_dict"])
    det_model.eval()

    # An explicit --threshold wins; otherwise use the one tuned on held-out
    # data and stored in the checkpoint.
    if args.threshold is None:
        args.threshold = ckpt.get("threshold", 0.5)
    if args.min_sep is None:
        args.min_sep = ckpt.get("min_sep", MIN_SEP)

    trained_on = ckpt.get("train_session_names") or ckpt.get("val_session_names")
    print(f"\nDetector:   {args.detector}  (epoch {ckpt['epoch']}, "
          f"holdout={ckpt.get('holdout','?')}, threshold {args.threshold})")
    if ckpt.get("val_f1") is not None:
        print(f"            held-out onset F1 {ckpt['val_f1']*100:.1f}%")
    print(f"Classifier: {args.classifier}")
    print(f"Device:     {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

    from crop_utils import load_detector
    detector = load_detector()
    if detector is None:
        print("WARNING: cube detector unavailable (needs ultralytics + "
              "cv/detection/detect_full_cube.pt).\n"
              "         Falling back to centered square crops — the model "
              "was trained on cube\n"
              "         crops, so expect degraded accuracy.")

    if args.session:
        dirs = [Path(p) for pattern in args.session
                for p in (Path(".").glob(pattern) if "*" in pattern
                          else [Path(pattern)])]
        dirs = sorted(d for d in dirs if d.is_dir())
        if not dirs:
            sys.exit("No session directories matched --session.")
        results = [r for d in dirs
                   if (r := replay(d, detector, det_model, device, args))]
        if len(results) > 1:
            tp = sum(r["onsets"]["tp"] for r in results)
            fp = sum(r["onsets"]["fp"] for r in results)
            fn = sum(r["onsets"]["fn"] for r in results)
            c  = sum(r["moves_correct"] for r in results)
            n  = sum(r["moves_compared"] for r in results)
            prec = tp / (tp + fp) if tp + fp else 0
            rec  = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            print(f"\n{'='*66}")
            print(f"  ACROSS {len(results)} SESSION(S)")
            print(f"  onsets  F1 {f1*100:.1f}%  P {prec*100:.1f}%  "
                  f"R {rec*100:.1f}%   tp {tp} fp {fp} fn {fn}")
            if n:
                print(f"  moves   {c}/{n} correct ({c/n*100:.1f}%)")
            print(f"  exact sequences: "
                  f"{sum(r['exact'] for r in results)}/{len(results)}")
        return

    scramble = None
    if args.scramble:
        scramble = generate_scramble(args.scramble, seed=args.seed)
        print(f"\n{'='*66}")
        print(f"  PERFORM THIS SCRAMBLE — {len(scramble)} moves")
        print(f"{'='*66}")
        for line in format_scramble(scramble):
            print(f"    {line}")
        print(f"\n  Hold the cube in ONE orientation for the whole sequence.")
        print(f"  The classifier predicts CAMERA-RELATIVE layers, so rotating")
        print(f"  the cube mid-scramble makes the comparison meaningless —")
        print(f"  every move after the rotation is relabeled. Regrip without")
        print(f"  turning the whole cube.")
        print(f"\n  SPACE to start, perform the moves, SPACE to stop.\n")

    buf, stamps = capture(args.camera,
                          overlay=format_scramble(scramble) if scramble else None)
    if not buf or len(stamps) < 2:
        print("Nothing recorded.")
        return

    print()
    fps = report_capture_rate(stamps)

    order = np.arange(len(buf))
    if fps < 28 and not args.no_resample:
        order, fps = resample_to_30fps(stamps)
        print(f"  resampling {len(buf)} captured frames onto a 30fps "
              f"timeline ({len(order)} frames)")
        print(f"  (--no-resample to disable)")

    print()
    load_color = lambda i: cv2.imdecode(buf[order[i]], cv2.IMREAD_COLOR)
    # `order` may repeat or skip captured frames (resample_to_30fps), so the
    # timeline handed to analyse has to be indexed the same way load_color is
    # — stamps[order], not stamps.
    res = analyse(load_color, len(order), fps, detector, det_model, device,
                  args.threshold, args.min_sep, args.classifier,
                  frame_times=stamps[order])
    print_sequence(res["moves"], fps)

    if scramble:
        report_scramble(scramble, res["moves"])

    if args.review:
        review(res, load_color)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Detector + classifier end-to-end move reading")
    p.add_argument("--detector",   type=str, default=DETECTOR_PATH)
    p.add_argument("--classifier", type=str, default=CLASSIFIER_PATH)
    p.add_argument("--camera",     type=int, default=0)
    p.add_argument("--session",    nargs="+",
                   help="Replay recorded session(s) against BLE ground truth "
                        "instead of using the webcam — supports globs")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Peak threshold (default: the value tuned on "
                        "held-out data, stored in the checkpoint)")
    p.add_argument("--min-sep",    type=int, default=None, dest="min_sep",
                   help="Refractory period in FRAMES between accepted peaks "
                        "(default: the checkpoint's value, normally 3). This "
                        "is the hard ceiling on move rate: at 30fps min_sep=3 "
                        "blocks anything above 10 moves/sec, which is inside "
                        "algorithm execution speed. Lower it to resolve fast "
                        "bursts, at the risk of one turn double-firing.")
    p.add_argument("--tolerance",  type=int, default=TOLERANCE,
                   help="Frames of slack when matching to ground truth "
                        "(replay mode only)")
    p.add_argument("--review",     action="store_true",
                   help="Step through each detected move's window visually")
    p.add_argument("--scramble",   type=int, nargs="?", const=25, default=None,
                   metavar="N",
                   help="Show an N-move scramble to perform (default 25), "
                        "then score the predicted sequence against it. "
                        "Works with ANY cube - no BLE needed, since the "
                        "label is the instruction rather than the hardware.")
    p.add_argument("--seed",       type=int, default=None,
                   help="Seed the scramble so the same sequence can be "
                        "repeated across runs when comparing settings")
    p.add_argument("--no-resample", action="store_true",
                   help="Do not resample a slow capture onto a 30fps "
                        "timeline (live mode only)")
    main(p.parse_args())
