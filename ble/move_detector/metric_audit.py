"""
metric_audit.py

Scores one set of predictions TWO ways and reports where they disagree.

The two ways
------------
TIME  every predicted move is matched to the ground-truth onset nearest it
      IN TIME, within --tolerance frames. This is what live_detect.replay()
      does, and it is the number the recorded-session results are quoted
      from (~95% of found moves named right).

SEQ   Needleman-Wunsch alignment of the predicted WORD SEQUENCE against the
      truth word sequence, timestamps discarded. This is what
      live_detect.report_scramble() does, and therefore what every live
      take — `--scramble`, and verify_solve.py --ble — has ever reported.

They are not two views of one number. Alignment has to guess the
correspondence from the words alone, and unit-cost NW makes that guess
biased in a specific direction: a substitution costs 1 while a
miss+phantom pair costs 2, so whenever the detector drops one move and
invents another, the cheapest explanation is "detected, named wrong" —
which is charged to the CLASSIFIER. The traceback compounds it by testing
the diagonal first, so exact ties resolve to `sub` as well.

verify_solve.py already holds the information that settles it: BLE events
arrive with timestamps (`truth.moves_between()` keeps `ts`), and the take's
frames are timestamped too. Scoring throws that away by calling
`truth.words_between()` and comparing words.

What else this checks
---------------------
* the anchor-offset histogram on LIVE takes specifically. Every offset
  number in this repo was measured on record_training.py sessions; a live
  take goes through a different capture path, and a systematic BLE-to-frame
  timing shift would move every window without changing a single word.
* run lengths of consecutive wrong moves. Whole-cube rotations are invisible
  to BLE (no x/y/z events), and one rotation relabels every following move,
  so it shows up as ONE long run of errors where classifier noise shows up
  as isolated singles.

Usage:
    python metric_audit.py --sessions ../training_data/solve_20260726_*/
    python metric_audit.py --sessions ../training_data/solve_2026072[56]*/
"""

import argparse
import json
import sys
from collections import Counter
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

from model import build_model                                     # noqa: E402
from decode import (align_sequences, onset_collisions,           # noqa: E402
                    MIN_SEP, TOLERANCE)
from live_detect import analyse, DETECTOR_PATH, CLASSIFIER_PATH   # noqa: E402


def gt_onset_frames(move_ts: np.ndarray, frame_ts: np.ndarray) -> np.ndarray:
    """BLE timestamps -> nearest frame index. Identical to prepare_data.py."""
    n = len(frame_ts)
    idx = np.clip(np.searchsorted(frame_ts, move_ts), 0, n - 1)
    left = np.clip(idx - 1, 0, n - 1)
    take_left = np.abs(frame_ts[left] - move_ts) < np.abs(frame_ts[idx] - move_ts)
    return np.where(take_left, left, idx).astype(int)


def score_by_time(moves: list[dict], gt_onset: np.ndarray,
                  gt_labels: list[str], tolerance: int) -> dict:
    """
    Greedy one-to-one match on TIME, most-confident prediction first, then
    the same ok/sub/miss/phantom breakdown report_scramble prints.

    Confidence rather than onset score decides who claims a ground truth
    first, matching decode.match_onsets' policy of letting the strongest
    prediction win a contested match.
    """
    unmatched = list(range(len(gt_onset)))
    pairs: dict[int, int] = {}          # gt index -> prediction index
    for pi in sorted(range(len(moves)), key=lambda i: -moves[i]["conf"]):
        if not unmatched:
            break
        f = moves[pi]["frame"]
        best = min(unmatched, key=lambda j: abs(int(gt_onset[j]) - f))
        if abs(int(gt_onset[best]) - f) <= tolerance:
            unmatched.remove(best)
            pairs[best] = pi

    ok = sub = 0
    wrong, seq_err = [], []
    for j in range(len(gt_onset)):
        pi = pairs.get(j)
        if pi is None:
            seq_err.append(None)        # miss — breaks an error run
            continue
        if moves[pi]["move"] == gt_labels[j]:
            ok += 1
            seq_err.append(False)
        else:
            sub += 1
            seq_err.append(True)
            wrong.append((j, gt_labels[j], moves[pi]["move"],
                          moves[pi]["conf"],
                          moves[pi]["frame"] - int(gt_onset[j])))
    miss = len(gt_onset) - len(pairs)
    phantom = len(moves) - len(pairs)

    # Split the misses the model could have avoided from the ones no peak
    # picker can report at all. Without this the headline miss count is
    # inflated by the ground truth's own time resolution — on four of the
    # 2026-07-24/25 sessions EVERY miss was of the unresolvable kind, which
    # made a frame-rate limit read as a detector regression.
    unresolvable, crowded = onset_collisions(gt_onset)
    missed = [j for j in range(len(gt_onset)) if j not in pairs]
    miss_forced = sum(1 for j in missed if j in unresolvable)
    miss_crowded = sum(1 for j in missed if j in crowded)

    return {"ok": ok, "sub": sub, "miss": miss, "phantom": phantom,
            "miss_forced": miss_forced, "miss_crowded": miss_crowded,
            "miss_clean": miss - miss_forced - miss_crowded,
            "n_unresolvable": len(unresolvable), "n_crowded": len(crowded),
            "pairs": pairs, "wrong": wrong, "seq_err": seq_err}


def score_by_sequence(moves: list[dict], gt_labels: list[str]) -> dict:
    ops = align_sequences(gt_labels, [m["move"] for m in moves])
    c = {k: sum(1 for o, _, _ in ops if o == k)
         for k in ("ok", "sub", "miss", "phantom")}
    return {**c, "ops": ops}


def error_runs(seq_err: list) -> Counter:
    """Run lengths of consecutive WRONG moves (None = miss, ends a run)."""
    runs, cur = Counter(), 0
    for e in seq_err + [None]:
        if e is True:
            cur += 1
        else:
            if cur:
                runs[cur] += 1
            cur = 0
    return runs


def audit_session(session_dir: Path, detector, det_model, device,
                  args) -> dict | None:
    frames_dir = session_dir / "frames"
    fidx = session_dir / "frames.jsonl"
    mpath = session_dir / "moves.jsonl"
    if not (frames_dir.is_dir() and fidx.exists() and mpath.exists()):
        print(f"  {session_dir.name}: needs frames/, frames.jsonl and "
              f"moves.jsonl — skipped")
        return None

    recs = [json.loads(l) for l in open(fidx) if l.strip()]
    paths = [frames_dir / r["file"] for r in recs]
    keep = [i for i, p in enumerate(paths) if p.exists()]
    recs = [recs[i] for i in keep]
    paths = [paths[i] for i in keep]
    n = len(paths)
    if n < 2:
        print(f"  {session_dir.name}: fewer than 2 frames — skipped")
        return None

    ts = np.array([r["ts"] for r in recs], dtype=np.float64)
    fps = n / (ts[-1] - ts[0])
    gt = [json.loads(l) for l in open(mpath) if l.strip()]
    gt = [m for m in gt if m.get("wca_notation")]
    if not gt:
        print(f"  {session_dir.name}: no wca_notation in moves.jsonl — skipped")
        return None
    gt_ts = np.array([m["timestamp"] for m in gt], dtype=np.float64)
    gt_labels = [m["wca_notation"] for m in gt]
    gt_onset = gt_onset_frames(gt_ts, ts)

    print(f"\n  {session_dir.name}: {n} frames, {fps:.1f}fps, "
          f"{len(gt)} BLE moves")
    # A take whose BLE events sit outside the frame span is misattributed,
    # not misclassified — worth knowing before reading any accuracy below.
    out = int(((gt_ts < ts[0]) | (gt_ts > ts[-1])).sum())
    if out:
        print(f"    WARNING: {out} BLE move(s) fall OUTSIDE the captured "
              f"frame span — they can never be matched.")

    cache: dict[int, np.ndarray] = {}

    def load_color(i):
        if i not in cache:
            if len(cache) > 800:
                cache.clear()
            cache[i] = cv2.imread(str(paths[i]))
        return cache[i]

    res = analyse(load_color, n, fps, detector, det_model, device,
                  args.threshold, args.min_sep, args.classifier,
                  frame_times=ts)
    moves = res["moves"]
    if not moves:
        print(f"    no moves detected — skipped")
        return None

    t = score_by_time(moves, gt_onset, gt_labels, args.tolerance)
    s = score_by_sequence(moves, gt_labels)

    # Per-move records, so accuracy can be stratified by how CROWDED each
    # move is. Aggregate accuracy cannot compare two regimes that turn at
    # different speeds, and these regimes do: recorded sessions sit at a
    # 300-360ms median inter-move gap, live free solves at 210ms.
    boxes = res["boxes"]
    side = np.median(boxes[:, 2] - boxes[:, 0])
    box_frac = float(side / load_color(0).shape[0])
    box_jit = float(np.median(np.abs(np.diff(
        (boxes[:, 0] + boxes[:, 2]) / 2))) / max(1.0, side))

    rows = []
    for j in range(len(gt_ts)):
        prev = gt_ts[j] - gt_ts[j - 1] if j > 0 else np.inf
        nxt = gt_ts[j + 1] - gt_ts[j] if j < len(gt_ts) - 1 else np.inf
        pi = t["pairs"].get(j)
        # Was a NEIGHBOUR missed? A miss doubles the apparent gap either
        # side of it, so both neighbours get a window training never
        # squeezed — the detector's error landing on other moves' inputs.
        nbr_missed = ((j > 0 and j - 1 not in t["pairs"])
                      or (j < len(gt_ts) - 1 and j + 1 not in t["pairs"]))
        rows.append({
            "j": j, "truth": gt_labels[j],
            "gap": float(min(prev, nxt)),
            "nbr_missed": bool(nbr_missed),
            "found": pi is not None,
            "correct": bool(pi is not None
                            and moves[pi]["move"] == gt_labels[j]),
            "pred": moves[pi]["move"] if pi is not None else None,
            "conf": float(moves[pi]["conf"]) if pi is not None else None,
            "offset": (int(moves[pi]["frame"] - gt_onset[j])
                       if pi is not None else None),
        })

    def line(tag, d):
        det = d["ok"] + d["sub"]
        cls = d["ok"] / det * 100 if det else 0.0
        rec = det / len(gt) * 100
        print(f"    {tag:<5} ok {d['ok']:>4}  sub {d['sub']:>4}  "
              f"miss {d['miss']:>4}  phantom {d['phantom']:>4}   "
              f"recall {rec:>5.1f}%  CLASSIFIER {cls:>5.1f}%")

    print(f"    {'':5} {'':44}   {'detector':>13}  {'of found':>8}")
    line("TIME", t)
    line("SEQ", s)

    # A miss the peak picker was never allowed to avoid is not a model
    # result. Quote the adjusted recall next to the raw one, never instead
    # of it — the moves really were not reported, they just were not
    # reportable. See decode.onset_collisions.
    if t["miss_forced"] or t["n_unresolvable"]:
        det_adj = t["ok"] + t["sub"]
        denom = len(gt) - t["miss_forced"]
        print(f"    of the {t['miss']} TIME misses: {t['miss_forced']} "
              f"unresolvable (GT onsets < {MIN_SEP} frames apart — no peak "
              f"picker\n          can report both), {t['miss_crowded']} "
              f"crowded (exactly {MIN_SEP} apart), "
              f"{t['miss_clean']} clean")
        print(f"    recall excluding unresolvable: "
              f"{det_adj / denom * 100:.1f}%  (raw "
              f"{det_adj / len(gt) * 100:.1f}%)")

    print(f"    crop: cube fills {box_frac*100:.0f}% of frame height, "
          f"box drift {box_jit*100:.1f}% of its side per frame")

    return {"session": session_dir.name, "time": t, "seq": s, "rows": rows,
            "n_gt": len(gt), "moves": moves, "gt_onset": gt_onset,
            "gt_labels": gt_labels, "fps": fps, "n_frames": n,
            "box_frac": box_frac, "box_jit": box_jit}


CROWD_EDGES = [0.10, 0.15, 0.25, 0.40, np.inf]


def print_crowding(rows: list[dict]) -> None:
    """
    Recall and classifier accuracy against the gap to the nearest
    neighbouring move.

    This is the stratification that makes two regimes comparable. A number
    averaged over a session describes that session's turn speed as much as
    it describes the models, so two aggregate accuracies from differently
    paced footage are not measuring the same thing.
    """
    print(f"\n  BY CROWDING — gap to the nearest neighbouring move")
    print(f"    {'gap':<12} {'moves':>6} {'found':>7} {'recall':>7} "
          f"{'classifier':>11}")
    lo = 0.0
    for hi in CROWD_EDGES:
        sub = [r for r in rows if lo <= r["gap"] < hi]
        lo = hi
        if not sub:
            continue
        found = [r for r in sub if r["found"]]
        acc = (sum(r["correct"] for r in found) / len(found) * 100
               if found else 0.0)
        label = (f"<{hi*1000:.0f}ms" if hi != np.inf else ">=400ms")
        bar = "#" * int(acc / 5)
        print(f"    {label:<12} {len(sub):>6} {len(found):>7} "
              f"{len(found)/len(sub)*100:>6.1f}% {acc:>10.1f}%  {bar}")


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
                 "cv/detection/detect_full_cube.pt).")

    dirs = [Path(p) for pattern in args.sessions
            for p in (Path(".").glob(pattern) if "*" in pattern
                      else [Path(pattern)])]
    dirs = sorted(d for d in dirs if d.is_dir())
    if not dirs:
        sys.exit("No session directories matched --sessions.")

    print(f"\nDetector:   {args.detector}  (threshold {args.threshold}, "
          f"min_sep {args.min_sep})")
    print(f"Classifier: {args.classifier}")
    print(f"Sessions:   {len(dirs)}")

    results = [r for d in dirs
               if (r := audit_session(d, detector, det_model, device, args))]
    if not results:
        sys.exit("\nNothing audited.")

    agg = {k: {m: sum(r[k][m] for r in results)
               for m in ("ok", "sub", "miss", "phantom")}
           for k in ("time", "seq")}
    n_gt = sum(r["n_gt"] for r in results)

    print(f"\n{'='*72}")
    print(f"  SAME PREDICTIONS, TWO METRICS — {n_gt} truth moves, "
          f"{len(results)} session(s)")
    print(f"{'='*72}")
    print(f"  {'metric':<22} {'ok':>5} {'sub':>5} {'miss':>5} {'phantom':>7}"
          f"  {'recall':>7} {'classifier':>11}")
    for k, label in (("time", "TIME (frame match)"),
                     ("seq", "SEQ  (word align)")):
        d = agg[k]
        det = d["ok"] + d["sub"]
        print(f"  {label:<22} {d['ok']:>5} {d['sub']:>5} {d['miss']:>5} "
              f"{d['phantom']:>7}  {det/n_gt*100:>6.1f}% "
              f"{d['ok']/det*100 if det else 0:>10.1f}%")
    dt = agg["time"]["ok"] / max(1, agg["time"]["ok"] + agg["time"]["sub"])
    ds = agg["seq"]["ok"] / max(1, agg["seq"]["ok"] + agg["seq"]["sub"])
    print(f"\n  Word alignment reports the classifier "
          f"{(dt-ds)*100:+.1f} points {'lower' if dt > ds else 'higher'} "
          f"than timestamp matching\n  on identical predictions.")
    if agg["seq"]["sub"] > agg["time"]["sub"]:
        print(f"  It converts {agg['seq']['sub'] - agg['time']['sub']} "
              f"detector error(s) into classifier substitutions: "
              f"miss+phantom costs 2,\n  a sub costs 1, so NW always "
              f"prefers to call it a naming mistake.")

    all_rows = [r for res in results for r in res["rows"]]
    print_crowding(all_rows)

    # Does a detector MISS hurt the moves next to it?
    found = [r for r in all_rows if r["found"]]
    for flag, label in ((False, "no missed neighbour"),
                        (True,  "a neighbour was MISSED")):
        sub = [r for r in found if r["nbr_missed"] is flag]
        if sub:
            print(f"  {'NEIGHBOUR' if flag else '':<11} {label:<24} "
                  f"{len(sub):>5} moves  classifier "
                  f"{sum(r['correct'] for r in sub)/len(sub)*100:>5.1f}%")

    print(f"\n  CROP GEOMETRY")
    for res in results:
        print(f"    {res['session']:<34} cube {res['box_frac']*100:>4.0f}% "
              f"of frame height   drift {res['box_jit']*100:>4.1f}%/frame")

    # Anchor offsets, on live takes specifically
    offs = [m["frame"] - int(r["gt_onset"][j])
            for r in results for j, pi in r["time"]["pairs"].items()
            for m in [r["moves"][pi]]]
    if offs:
        o = np.array(offs)
        print(f"\n  ANCHOR OFFSET on these takes (detected - BLE frame), "
              f"n={len(o)}")
        print(f"    mean {o.mean():+.2f} frames ({o.mean()/30*1000:+.0f}ms)   "
              f"|mean| {np.abs(o).mean():.2f}   "
              f"median {np.median(o):+.0f}")
        h = Counter(int(x) for x in o)
        print(f"    " + "  ".join(f"{k:+d}:{h[k]}" for k in sorted(h)))
        print(f"    Recorded sessions measured mean +0.02, |mean| 0.29 — a "
              f"shift here would mean\n    the live capture path times BLE "
              f"events against frames differently.")

    # Rotation check
    runs = Counter()
    for r in results:
        runs += error_runs(r["time"]["seq_err"])
    if runs:
        total = sum(runs.values())
        print(f"\n  ERROR RUN LENGTHS (consecutive wrong, TIME matching)")
        print(f"    " + "  ".join(f"len{k}:{runs[k]}" for k in sorted(runs)))
        long_runs = sum(v for k, v in runs.items() if k >= 5)
        print(f"    {total} run(s); {long_runs} of length >=5.")
        print(f"    Isolated singles = classifier noise. One long run = a "
              f"whole-cube ROTATION,\n    which BLE cannot see and which "
              f"relabels every following move.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"sessions": [{"session": r["session"], "n_gt": r["n_gt"],
                           "time": {k: r["time"][k] for k in
                                    ("ok", "sub", "miss", "phantom")},
                           "seq": {k: r["seq"][k] for k in
                                   ("ok", "sub", "miss", "phantom")},
                           "rows": r["rows"],
                           "wrong": r["time"]["wrong"]}
                          for r in results]}, indent=1))
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare timestamp matching against word alignment on "
                    "identical predictions")
    p.add_argument("--sessions", nargs="+", required=True,
                   help="Session folder(s) with frames/, frames.jsonl and "
                        "moves.jsonl. Saved live takes qualify — no "
                        "postprocess_session.py needed")
    p.add_argument("--detector",   type=str, default=DETECTOR_PATH)
    p.add_argument("--classifier", type=str, default=CLASSIFIER_PATH)
    p.add_argument("--threshold",  type=float, default=None)
    p.add_argument("--min-sep",    type=int, default=None, dest="min_sep")
    p.add_argument("--tolerance",  type=int, default=TOLERANCE)
    p.add_argument("--json", type=str, default=None)
    main(p.parse_args())
