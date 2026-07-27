"""
window_audit.py

Asks one question: when the detector finds a move, is the 5-frame window it
hands the classifier the SAME KIND of window the classifier trained on?

Why this is not answered by onset F1
------------------------------------
The detector's metric is "did a peak land within +/-2 frames of the BLE
timestamp" (decode.TOLERANCE). That tolerance is 67ms at 30fps. The
classifier's window offsets are +30ms / +60ms / +100ms — 0.9, 1.8 and 3.0
frames. So an onset that scores as a perfect true positive can still shift
the entire diff stack by more than the spacing between its own mid frames.
A detector can be 96% recall and still be feeding the classifier inputs it
never saw in training, and no detector metric will say so.

Worse, the mistakes compound sideways. `decode.onset_windows` derives each
move's window SQUEEZE from the gaps to its neighbouring DETECTED onsets,
whereas `postprocess_session.py` derived it from the gaps to the
neighbouring BLE events. One missed move doubles the apparent gap for the
two moves on either side of it, so they get a wide window where training
used a squeezed one — the detector's error lands on its neighbours' inputs,
not on its own.

The ladder
----------
Five variants of the same moves, each adding exactly one ingredient of the
live regime to the trainer's own input. Every variant is scored on the same
move set (onsets the detector matched), so the differences are attributable
and detector recall is held out of it:

  T  trainer's window                  labeled/*.jpg + crops.json box.
                                       Literally what train_move_classifier
                                       reads. The reference.
  Q  + inference windowing             the window builder decode.py uses,
                                       still on the GT anchor. This is the
                                       cost of a PERFECT detector.
  A  + detector anchor                 anchor moves to the detected peak.
  G  + detector-derived gaps           squeeze from detected neighbours.
  L  + live crop path                  per_frame_boxes median instead of
                                       crops.json. == live_detect.analyse.

T -> L is the total cost of detector-sourced data. Each step names which
part of the pipeline owns which share of it.

Rung Q is the one this tool was built to find, and it is not a detector
problem at all: with the anchor held PERFECT, the inference window builder
still disagrees with the trainer's frame choice on most moves, and costs
1.8 points.

Read that rung by frame_diff, not by its headline, or you will draw the
wrong conclusion — as happened on 2026-07-26. The headline said "index
arithmetic against a global mean fps"; switching to nearest-frame-in-time
(`--windowing index` still replays the old behaviour) raised exact
agreement 11% -> 33% but bought only +0.3 points, McNemar p=0.40. The
per-bucket table is what actually explains the rung: accuracy falls
monotonically with the number of differing frames while the trainer's own
accuracy on those same moves stays flat, so the cost is real and caused by
the window — there is simply no way to pick the right frames from an anchor
that is already quantised to a frame index. See decode.window_from_anchor.

It also reports, with no classifier involved at all:
  * the anchor-offset histogram, and accuracy stratified by it;
  * how many of the 5 window frames the live path picks differently;
  * squeeze disagreement caused by detector miss/phantom neighbours;
  * crop-box agreement (IoU / centre shift / scale) between the two crop
    paths, which are genuinely different algorithms — cache_crops.py takes
    the median detection over the move's own 5 frames, prepare_data.py
    detects every 10th frame, rolling-medians over ~1.7s and interpolates.

Usage:
    python window_audit.py --sessions ../training_data/solve_2026072[1-4]*/
    python window_audit.py --sessions ../training_data/solve_20260721_102711/
    python window_audit.py --sessions ../training_data/solve_*/ --json out.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
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

from dataset import ArrayStream                                   # noqa: E402
from model import build_model, score_stream                       # noqa: E402
from decode import (peak_pick, onset_windows, window_from_anchor,  # noqa: E402
                    MIN_SEP, TOLERANCE)
from prepare_data import (per_frame_boxes, build_gray_stream,     # noqa: E402
                          crop_to_box)
from postprocess_session import (move_window, WINDOWS,            # noqa: E402
                                 find_nearest, HALF_FRAME)
from train_move_classifier import predict_probs                   # noqa: E402
from live_detect import DETECTOR_PATH, CLASSIFIER_PATH            # noqa: E402

# The ladder, in order. Each entry is (key, label, description). Rung Q's
# description depends on --windowing, since that rung IS the windowing method.
def variants(windowing: str) -> list[tuple[str, str, str]]:
    return [
        ("T", "trainer window", "labeled/*.jpg + crops.json box"),
        ("Q", "+ inference windowing",
         "GT anchor, nearest frame in time" if windowing == "timestamp"
         else "GT anchor, round(o + dt*fps)"),
        ("A", "+ detector anchor",  "anchor = detected peak"),
        ("G", "+ detected gaps",    "squeeze from detected neighbours"),
        ("L", "+ live crop path",   "per_frame_boxes median (= live_detect)"),
    ]


VARIANT_KEYS = ["T", "Q", "A", "G", "L"]


# ---------------------------------------------------------------------------
# Matching — same greedy rule as decode.match_onsets, but keeps the PAIRS
# ---------------------------------------------------------------------------

def match_pairs(pred: np.ndarray, gt: np.ndarray, scores: np.ndarray,
                tolerance: int = TOLERANCE) -> dict[int, int]:
    """
    {gt_index_into_gt_array: predicted_frame}. Strongest predictions claim
    their ground truth first — identical policy to decode.match_onsets, which
    is the function whose numbers this audit has to stay comparable with.
    """
    order = np.argsort(-scores[pred]) if len(pred) else np.arange(0)
    unmatched = list(range(len(gt)))
    pairs: dict[int, int] = {}
    for p in np.asarray(pred)[order]:
        if not unmatched:
            break
        best = min(unmatched, key=lambda j: abs(int(gt[j]) - int(p)))
        if abs(int(gt[best]) - int(p)) <= tolerance:
            unmatched.remove(best)
            pairs[best] = int(p)
    return pairs


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def trainer_window(t: float, offsets: dict, frame_times: list[float],
                   frame_recs: list[dict]) -> dict[str, int] | None:
    """
    postprocess_session's own frame choice, re-derived: nearest frame IN TIME
    to (T + offset), constrained to the move's sandwich. Returns frame
    indices (postprocess stores only the copied filename, so the index has to
    be recomputed — same function, same inputs, so it is the same answer).
    None if any window key had no frame in range, which is exactly when
    postprocess wrote a null and the trainer skipped the move.
    """
    lo_ts = t + offsets["before"] - HALF_FRAME
    hi_ts = t + offsets["after"] + HALF_FRAME
    out = {}
    for key, off in offsets.items():
        rec = find_nearest(frame_times, frame_recs, t + off)
        if rec is None or not (lo_ts <= rec["ts"] <= hi_ts):
            return None
        out[key] = rec["_idx"]
    return out


# ---------------------------------------------------------------------------
# Crop-box comparison
# ---------------------------------------------------------------------------

def box_agreement(a, b) -> tuple[float, float, float]:
    """(IoU, centre shift as a fraction of a's side, side ratio b/a)."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
    iou = inter / (area_a + area_b - inter)
    side_a = max(1e-9, ax2 - ax1)
    shift = np.hypot(((bx1 + bx2) - (ax1 + ax2)) / 2,
                     ((by1 + by2) - (ay1 + ay2)) / 2) / side_a
    return iou, float(shift), (bx2 - bx1) / side_a


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------

def audit_session(session_dir: Path, detector, det_model, device,
                  args) -> dict | None:
    frames_dir = session_dir / "frames"
    npz        = session_dir / "detector_stream.npz"
    labeled    = session_dir / "moves_labeled.jsonl"
    crops_file = session_dir / "crops.json"

    for path, why in ((frames_dir, "no frames/"), (npz, "no detector_stream.npz"),
                      (labeled, "no moves_labeled.jsonl"),
                      (crops_file, "no crops.json")):
        if not path.exists():
            print(f"  {session_dir.name}: {why} — skipped")
            return None

    recs = [json.loads(l) for l in open(session_dir / "frames.jsonl") if l.strip()]
    paths_all = [frames_dir / r["file"] for r in recs]
    keep = [i for i, p in enumerate(paths_all) if p.exists()]
    # prepare_data.py applied this same filter before computing onset_idx, so
    # the surviving order is the index space the .npz refers to.
    recs  = [recs[i] for i in keep]
    paths = [paths_all[i] for i in keep]
    for i, r in enumerate(recs):
        r["_idx"] = i
    frame_times = [r["ts"] for r in recs]
    n = len(paths)
    if n < 2:
        print(f"  {session_dir.name}: fewer than 2 frames — skipped")
        return None
    fps = n / (frame_times[-1] - frame_times[0])

    data     = np.load(npz, allow_pickle=True)
    gt_onset = data["onset_idx"].astype(int)
    moves_lab = [json.loads(l) for l in open(labeled) if l.strip()]
    crop_boxes = json.loads(crops_file.read_text())["boxes"]

    if len(moves_lab) != len(gt_onset):
        print(f"  {session_dir.name}: {len(moves_lab)} labeled moves vs "
              f"{len(gt_onset)} onsets in the stream — skipped (re-run "
              f"postprocess_session.py and prepare_data.py together)")
        return None

    print(f"\n  {session_dir.name}: {n} frames, {fps:.1f}fps, "
          f"{len(gt_onset)} moves")

    # ---- detector pass (identical construction to live_detect.analyse) ----
    cache: dict[int, np.ndarray] = {}

    def load_color(i):
        if i not in cache:
            if len(cache) > 800:
                cache.clear()
            cache[i] = cv2.imread(str(paths[i]))
        return cache[i]

    boxes, n_det = per_frame_boxes(detector, load_color, n)
    if n_det == 0:
        print(f"    cube NEVER detected — skipped (would audit a fallback "
              f"crop, not the pipeline)")
        return None

    gray = build_gray_stream(
        lambda i: (lambda f: None if f is None else
                   cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))(load_color(i)),
        boxes, n)
    scores  = score_stream(det_model, ArrayStream(gray, name=session_dir.name,
                                                  fps=fps), device)
    onsets  = peak_pick(scores, threshold=args.threshold, min_sep=args.min_sep)
    # None reproduces the pre-2026-07-26 index-arithmetic windowing, so the
    # two methods can be compared on identical detections.
    ts_arg  = np.asarray(frame_times) if args.windowing == "timestamp" else None
    det_win = onset_windows(onsets, fps, n, ts_arg)
    det_pos = {int(o): k for k, o in enumerate(onsets)}
    pairs   = match_pairs(onsets, gt_onset, scores, args.tolerance)

    print(f"    detector: {len(onsets)} onsets, {len(pairs)}/{len(gt_onset)} "
          f"matched (recall {len(pairs)/len(gt_onset)*100:.1f}%)")

    # ---- per-move ladder --------------------------------------------------
    tally   = {k: [0, 0] for k in VARIANT_KEYS}      # key -> [correct, total]
    rows    = []
    skipped = 0

    for j, det_frame in sorted(pairs.items()):
        m     = moves_lab[j]
        truth = m.get("wca_notation")
        box_t = crop_boxes.get(f"move_{m['move_num']:04d}")
        if not truth or box_t is None:
            skipped += 1
            continue

        t = m["timestamp"]
        gap_prev = t - moves_lab[j - 1]["timestamp"] if j > 0 else None
        gap_next = moves_lab[j + 1]["timestamp"] - t \
            if j < len(moves_lab) - 1 else None
        ble_off = move_window(gap_prev, gap_next)

        w_train = trainer_window(t, ble_off, frame_times, recs)
        if w_train is None:
            skipped += 1          # postprocess wrote nulls; trainer skips it
            continue

        w_q = window_from_anchor(int(gt_onset[j]), ble_off, fps, n, ts_arg)
        w_a = window_from_anchor(det_frame,        ble_off, fps, n, ts_arg)
        w_g = det_win[det_pos[det_frame]]           # detected anchor AND gaps
        box_l = np.median(boxes[[w_g[k] for k in WINDOWS]],
                          axis=0).astype(int).tolist()

        plan = {
            "T": (w_train, box_t, True),   # True = load from labeled/*.jpg
            "Q": (w_q,     box_t, False),
            "A": (w_a,     box_t, False),
            "G": (w_g,     box_t, False),
            "L": (w_g,     box_l, False),
        }

        preds = {}
        bad = False
        for key, (win, box, from_labeled) in plan.items():
            frames = []
            for wk in WINDOWS:
                if from_labeled:
                    rel = m.get("frames", {}).get(wk)
                    img = cv2.imread(str(session_dir / rel)) if rel else None
                else:
                    img = load_color(win[wk])
                if img is None:
                    bad = True
                    break
                frames.append(crop_to_box(img, box))
            if bad:
                break
            probs, names = predict_probs(frames, args.classifier)
            preds[key] = names[int(np.argmax(probs))]
        if bad:
            skipped += 1
            continue

        for key, p in preds.items():
            tally[key][1] += 1
            tally[key][0] += (p == truth)

        iou, shift, scale = box_agreement(box_t, box_l)
        rows.append({
            "move": j,
            "truth": truth,
            "pred": preds,
            "offset": det_frame - int(gt_onset[j]),
            # How many of the 5 window frames the live path picks differently
            # from the trainer, at each rung.
            "frame_diff": {k: sum(w[wk] != w_train[wk] for wk in WINDOWS)
                           for k, w in (("Q", w_q), ("A", w_a), ("G", w_g))},
            # Squeeze disagreement: post-squeeze span, live vs BLE-derived.
            "span_ble":  ble_off["after"] - ble_off["before"],
            "span_live": (w_g["after"] - w_g["before"]) / fps,
            "iou": iou, "shift": shift, "scale": scale,
        })

    if not rows:
        print(f"    no auditable moves — skipped")
        return None
    if skipped:
        print(f"    {skipped} matched move(s) skipped (no label, no crop box, "
              f"or postprocess found no frame in window)")

    return {"session": session_dir.name, "tally": tally, "rows": rows,
            "n_moves": len(gt_onset), "n_matched": len(pairs), "fps": fps}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_ladder(tally: dict, title: str, vars_: list) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    print(f"  {'':2} {'variant':<22} {'adds':<34} {'acc':>7}  {'delta':>7}")
    print(f"  {'-'*70}")
    prev = None
    for key, label, desc in vars_:
        c, t = tally[key]
        acc = c / t * 100 if t else 0.0
        delta = "" if prev is None else f"{acc - prev:+6.1f}"
        print(f"  {key:<2} {label:<22} {desc:<34} {acc:>6.1f}%  {delta:>7}")
        prev = acc
    c_t, n = tally["T"]
    c_l, _ = tally["L"]
    print(f"  {'-'*70}")
    print(f"  total cost of detector-sourced windows: "
          f"{(c_l - c_t) / n * 100:+.1f} points over {n} moves "
          f"({c_t - c_l} moves lost)")


def print_offsets(rows: list[dict]) -> None:
    hist = Counter(r["offset"] for r in rows)
    print(f"\n  ANCHOR OFFSET (detected peak - BLE frame), over "
          f"{len(rows)} matched moves")
    print(f"    {'offset':>7} {'moves':>6}   {'T acc':>7} {'A acc':>7} "
          f"{'L acc':>7}")
    for off in sorted(hist):
        sub = [r for r in rows if r["offset"] == off]
        cells = []
        for key in ("T", "A", "L"):
            c = sum(r["pred"][key] == r["truth"] for r in sub)
            cells.append(f"{c/len(sub)*100:>6.1f}%")
        bar = "#" * int(len(sub) / max(hist.values()) * 22)
        print(f"    {off:>+7d} {len(sub):>6}   {'  '.join(cells)}  {bar}")
    errs = np.array([r["offset"] for r in rows])
    print(f"    mean {errs.mean():+.2f} frames ({errs.mean()/30*1000:+.0f}ms), "
          f"|median| {np.median(np.abs(errs)):.0f}, "
          f"|mean| {np.abs(errs).mean():.2f}")
    print(f"    A non-zero offset is still a true positive to the detector — "
          f"tolerance is +/-2\n    frames (67ms), while the window's own mid "
          f"frames are 30/60/100ms apart.")


def print_frame_diff(rows: list[dict], vars_: list) -> None:
    print(f"\n  WINDOW FRAMES DIFFERING from the trainer's own choice "
          f"(out of 5)")
    print(f"    {'rung':<22} {'mean':>6} {'0':>6} {'1-2':>6} {'3-4':>6} "
          f"{'5':>6}")
    for key, label, _ in vars_[1:4]:
        vals = [r["frame_diff"][key] for r in rows]
        b = Counter()
        for v in vals:
            b["0" if v == 0 else "1-2" if v <= 2 else "3-4" if v <= 4 else "5"] += 1
        n = len(vals)
        print(f"    {key} {label:<20} {np.mean(vals):>6.2f} "
              f"{b['0']/n*100:>5.0f}% {b['1-2']/n*100:>5.0f}% "
              f"{b['3-4']/n*100:>5.0f}% {b['5']/n*100:>5.0f}%")

    # The reading that actually explains rung Q. The T column is the control:
    # were it falling too, these would merely be harder moves.
    print(f"\n    Accuracy vs how many of the 5 slots differ (rung Q), the "
          f"same moves scored\n    on the trainer's own window as the control:")
    print(f"      {'differing':<10} {'moves':>6} {'T (control)':>12} {'Q':>8}")
    for k in range(6):
        sub = [r for r in rows if r["frame_diff"]["Q"] == k]
        if not sub:
            continue
        t = sum(r["pred"]["T"] == r["truth"] for r in sub) / len(sub) * 100
        q = sum(r["pred"]["Q"] == r["truth"] for r in sub) / len(sub) * 100
        print(f"      {k:<10} {len(sub):>6} {t:>11.1f}% {q:>7.1f}%")


def print_squeeze(rows: list[dict]) -> None:
    ratio = np.array([r["span_live"] / max(1e-6, r["span_ble"]) for r in rows])
    wide = int((ratio > 1.25).sum())
    narrow = int((ratio < 0.8).sum())
    print(f"\n  WINDOW SPAN, live vs training  (squeeze comes from DETECTED "
          f"neighbours live)")
    print(f"    median ratio {np.median(ratio):.2f}   "
          f"{wide} move(s) >25% WIDER ({wide/len(rows)*100:.0f}%)   "
          f"{narrow} move(s) >20% narrower ({narrow/len(rows)*100:.0f}%)")
    if wide:
        sub = [r for r in rows
               if r["span_live"] / max(1e-6, r["span_ble"]) > 1.25]
        c_g = sum(r["pred"]["G"] == r["truth"] for r in sub)
        c_a = sum(r["pred"]["A"] == r["truth"] for r in sub)
        print(f"    on those {len(sub)}: A (BLE gaps) {c_a/len(sub)*100:.1f}% "
              f"-> G (detected gaps) {c_g/len(sub)*100:.1f}%")
        print(f"    A wider-than-training window means a neighbouring move was "
              f"MISSED, so the\n    gap looks twice as long — the detector's "
              f"error lands on its neighbours' inputs.")


def print_crops(rows: list[dict]) -> None:
    iou   = np.array([r["iou"] for r in rows])
    shift = np.array([r["shift"] for r in rows])
    scale = np.array([r["scale"] for r in rows])
    print(f"\n  CROP BOX AGREEMENT  crops.json (training) vs per_frame_boxes "
          f"(live)")
    print(f"    IoU        median {np.median(iou):.3f}   p10 "
          f"{np.percentile(iou, 10):.3f}   <0.7 on "
          f"{(iou < 0.7).sum()/len(iou)*100:.0f}% of moves")
    print(f"    centre     median {np.median(shift)*100:.1f}% of the box side "
          f"  p90 {np.percentile(shift, 90)*100:.1f}%")
    print(f"    scale      median {np.median(scale):.3f}   p10 "
          f"{np.percentile(scale, 10):.3f}   p90 "
          f"{np.percentile(scale, 90):.3f}")
    print(f"    Training augments +/-8% translate and 0.9-1.1 scale, so shifts "
          f"inside that\n    band are covered; the tail is not.")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.detector, map_location=device)
    det_model = build_model(device)
    det_model.load_state_dict(ckpt["state_dict"])
    det_model.eval()
    if args.threshold is None:
        args.threshold = ckpt.get("threshold", 0.5)
    if args.min_sep is None:
        args.min_sep = ckpt.get("min_sep", MIN_SEP)

    from crop_utils import load_detector
    detector = load_detector()
    if detector is None:
        sys.exit("Cube detector unavailable (needs ultralytics + "
                 "cv/detection/detect_full_cube.pt). This audit is entirely "
                 "about the crop/window path, so a fallback crop would make "
                 "it meaningless.")

    dirs = [Path(p) for pattern in args.sessions
            for p in (Path(".").glob(pattern) if "*" in pattern
                      else [Path(pattern)])]
    dirs = sorted(d for d in dirs if d.is_dir())
    if not dirs:
        sys.exit("No session directories matched --sessions.")

    vars_ = variants(args.windowing)
    print(f"\nDetector:   {args.detector}  (threshold {args.threshold}, "
          f"min_sep {args.min_sep})")
    print(f"Classifier: {args.classifier}")
    print(f"Windowing:  {args.windowing}"
          f"{'' if args.windowing == 'timestamp' else '  (legacy, pre-2026-07-26)'}")
    print(f"Sessions:   {len(dirs)}")

    results = [r for d in dirs
               if (r := audit_session(d, detector, det_model, device, args))]
    if not results:
        sys.exit("\nNo sessions could be audited.")

    all_rows = [r for res in results for r in res["rows"]]
    total = {k: [0, 0] for k in VARIANT_KEYS}
    for res in results:
        for k, (c, t) in res["tally"].items():
            total[k][0] += c
            total[k][1] += t

    if len(results) > 1:
        print(f"\n{'='*72}")
        print(f"  PER SESSION — accuracy on matched onsets")
        print(f"{'='*72}")
        print(f"  {'session':<34} {'moves':>5} " +
              " ".join(f"{k:>6}" for k in VARIANT_KEYS) + f"  {'T->L':>6}")
        for res in results:
            cells, first, last = [], None, None
            for k in VARIANT_KEYS:
                c, t = res["tally"][k]
                acc = c / t * 100 if t else 0.0
                cells.append(f"{acc:>5.1f}%")
                first = acc if first is None else first
                last = acc
            print(f"  {res['session']:<34} {res['tally']['T'][1]:>5} " +
                  " ".join(cells) + f"  {last-first:>+5.1f}")

    print_ladder(total, f"LADDER — {len(all_rows)} moves across "
                        f"{len(results)} session(s)", vars_)
    print_offsets(all_rows)
    print_frame_diff(all_rows, vars_)
    print_squeeze(all_rows)
    print_crops(all_rows)

    # Which rung the errors that only exist live actually come from
    lost = [r for r in all_rows
            if r["pred"]["T"] == r["truth"] and r["pred"]["L"] != r["truth"]]
    if lost:
        blame = Counter()
        for r in lost:
            for k in ("Q", "A", "G", "L"):
                if r["pred"][k] != r["truth"]:
                    blame[k] += 1
                    break
        print(f"\n  {len(lost)} move(s) the trainer's window gets RIGHT and "
              f"the live window gets wrong.")
        print(f"  First rung at which each broke:")
        for k, label, _ in variants(args.windowing)[1:]:
            print(f"    {k} {label:<22} {blame[k]:>4}  "
                  f"({blame[k]/len(lost)*100:.0f}%)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"detector": args.detector, "classifier": args.classifier,
             "threshold": args.threshold, "min_sep": args.min_sep,
             "total": total,
             "sessions": [{"session": r["session"], "tally": r["tally"],
                           "n_moves": r["n_moves"], "n_matched": r["n_matched"],
                           "rows": r["rows"]} for r in results]},
            indent=1))
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Is the detector handing the classifier the kind of "
                    "window it trained on?")
    p.add_argument("--sessions", nargs="+", required=True,
                   help="Session folder(s), supports globs. Needs frames/, "
                        "detector_stream.npz, moves_labeled.jsonl and "
                        "crops.json")
    p.add_argument("--detector",   type=str, default=DETECTOR_PATH)
    p.add_argument("--classifier", type=str, default=CLASSIFIER_PATH)
    p.add_argument("--threshold",  type=float, default=None)
    p.add_argument("--min-sep",    type=int, default=None, dest="min_sep")
    p.add_argument("--windowing", choices=["timestamp", "index"],
                   default="timestamp",
                   help="How inference rebuilds the classifier window. "
                        "timestamp = nearest frame in time, the rule "
                        "postprocess_session.py used (default). index = the "
                        "legacy round(onset + offset*fps), kept so the two "
                        "can be compared on identical detections")
    p.add_argument("--tolerance",  type=int, default=TOLERANCE,
                   help="Frames of slack when matching detected onsets to "
                        "BLE ground truth")
    p.add_argument("--json", type=str, default=None,
                   help="Write the full per-move record here")
    main(p.parse_args())
