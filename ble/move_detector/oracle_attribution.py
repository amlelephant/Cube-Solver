"""
oracle_attribution.py

Gate G2 of MODEL_REWORK_PLAN.md: on the classifier-unseen ("honest")
sessions, which error stream actually binds the decoder's stuck-at-1/6
verified rate -- detector misses/phantoms, or classifier substitutions?
The decoder-lever sprint (decoder-sprint-exhausted memory) already proved
neither search width (D3) nor cost recalibration (D1/D2) can close the
gap; this measures which side of the detector/classifier boundary the
INFORMATION has to come from before any training is spent chasing it.

Four conditions per session, same decode, same cost model -- only the
INPUT changes:

  real                the deployed detector + deployed classifier,
                       unmodified (the baseline every other memory here
                       already reports).
  oracle-classifier    the SAME onsets as `real` (same misses, same
                       phantoms), but every onset TIME-MATCHED to a true
                       move (metric_audit.gt_onset_frames + score_by_time
                       -- NOT decode.align_sequences, which has no timing
                       information and would sometimes credit a
                       name-alignment "match" to the wrong physical
                       onset) gets its softmax replaced by a near-one-hot
                       on the truth label, run through the SAME
                       onset_costs() the real pipeline uses -- so the
                       cost model is identical, only the input softmax
                       differs. Phantom onsets (matched to nothing) keep
                       their real softmax; the truth has no label for
                       them.
  oracle-detector      onsets placed exactly at the true BLE-anchored
                       frame (perfect recall, zero misses, zero
                       phantoms), classified by RE-RUNNING the real
                       deployed classifier on a window anchored there --
                       this measures the model's actual accuracy under a
                       hypothetically perfect detector, not an assumed
                       one.
  both                 oracle onsets AND oracle softmax. Sanity check:
                       should verify on (almost) every session; if it
                       systematically doesn't, suspect the harness or the
                       endpoint states, not the models (see the
                       ble-truth-end-state-bug memory -- use the move
                       log for ground truth, never get_state()'s live
                       solved-check).

Whichever oracle condition recovers most of the true-verified gap toward
`both` names the binding error stream for Stage A's priority (recall /
background-class weighting vs. per-class discrimination).

    python oracle_attribution.py
    python oracle_attribution.py --sessions ../training_data/solve_2026072[67]*/

Session selection, unless --sessions is given: every training_data/solve_*
directory with moves.jsonl + frames/ that reconstruct.classifier_unseen()
currently reports as unseen by the deployed classifier -- computed live,
not read from a memory or a hardcoded list, per this project's own
"never cite a stale baseline" rule.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import reconstruct as RC
import live_detect as LD                              # noqa: F401 (sys.path bootstrap)
from metric_audit import gt_onset_frames, score_by_time  # noqa: E402
from decode import TOLERANCE                            # noqa: E402
from crop_utils import load_detector                    # noqa: E402

ONEHOT_EPS = 1e-4   # near-certain oracle softmax: mass (1 - 11*eps) on truth


# ---------------------------------------------------------------------------
# Frame loading -- identical filtering to reconstruct._load_replay, so onset
# frame indices computed here line up with the ones the real replay used.
# ---------------------------------------------------------------------------

def _load_frames_index(session_dir: Path):
    import cv2

    frames_dir = session_dir / "frames"
    recs = [json.loads(l) for l in open(session_dir / "frames.jsonl") if l.strip()]
    paths = [frames_dir / r["file"] for r in recs]
    keep = [i for i, p in enumerate(paths) if p.exists()]
    paths = [paths[i] for i in keep]
    ts = np.array([recs[i]["ts"] for i in keep])
    n = len(paths)
    fps = n / (ts[-1] - ts[0]) if n > 1 else 30.0

    cache: dict[int, np.ndarray] = {}

    def load_color(i):
        if i not in cache:
            if len(cache) > 600:
                cache.clear()
            cache[i] = cv2.imread(str(paths[i]))
        return cache[i]

    return load_color, n, fps, ts


def classify_at_onsets(load_color, onsets, n_frames, fps, frame_times,
                       detector, classifier):
    """
    The deployed classifier's REAL softmax at a caller-chosen onset frame
    list -- the same window/crop construction live_detect.analyse() uses
    for the detector's own peaks, just fed a different onset list (here,
    the perfect BLE-anchored frames instead of a peak-picked score curve).
    """
    if detector is not None:
        boxes, _ = LD.per_frame_boxes(detector, load_color, n_frames)
    else:
        probe = load_color(0)
        boxes = np.tile(LD.center_square(probe.shape),
                        (n_frames, 1)).astype(np.int32)

    onsets = np.asarray(onsets, dtype=int)
    windows = LD.onset_windows(onsets, fps, n_frames, frame_times)
    moves, class_names = [], None
    for o, w in zip(onsets, windows):
        idxs = [w[k] for k in LD.FRAME_ORDER]
        raw = [load_color(i) for i in idxs]
        if any(f is None for f in raw):
            moves.append(None)
            continue
        box = np.median(boxes[idxs], axis=0).astype(int)
        frames = [LD.crop_to_box(f, box) for f in raw]
        probs, class_names = LD.predict_probs(frames, classifier)
        cls = int(np.argmax(probs))
        # score=1.0: this onset is placed with perfect (oracle) confidence,
        # not detected -- costs_from_moves needs the field to price deletion.
        moves.append({"frame": int(o), "move": class_names[cls],
                      "conf": float(probs[cls]),
                      "probs": [float(p) for p in probs], "score": 1.0})
    return moves, class_names


def onehot_probs(label: str) -> np.ndarray:
    k = RC.WCA12.index(label)
    p = np.full(12, ONEHOT_EPS / 11, dtype=np.float64)
    p[k] = 1.0 - ONEHOT_EPS
    return p


# ---------------------------------------------------------------------------
# Session selection
# ---------------------------------------------------------------------------

def honest_sessions(root: Path, classifier_path: str) -> list[Path]:
    dirs = sorted(d for d in root.glob("solve_*") if d.is_dir())
    out = []
    for d in dirs:
        if not (d / "moves.jsonl").exists() or not (d / "frames").is_dir():
            continue
        if RC.classifier_unseen(d.name, classifier_path):
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Decode one condition, with the same beam/retry policy verify_solve uses
# ---------------------------------------------------------------------------

def decode_report(label, start, end, cost_rows, del_costs, gt, args, tables):
    kw = dict(c_del=args.del_cost, c_ins=args.ins_cost, c_rot=args.rot_cost,
              max_end_ins=args.max_end_ins, rel_weight=args.rel_weight,
              del_costs=del_costs, rotations=False, tables=tables)
    res = RC.decode_between(start, end, cost_rows, beam=args.beam, **kw)
    if not res["solved"] and args.retry_beam > args.beam:
        res = RC.decode_between(start, end, cost_rows, beam=args.retry_beam,
                                **kw)
        res["retried"] = True
    pred_names = [RC.WCA12[int(np.argmin(r))] for r in cost_rows]
    gtc = RC.gt_path_cost(gt, pred_names, cost_rows, del_costs, args.ins_cost)
    rec_acc = None
    if res["solved"]:
        rec_acc = RC.score_vs_gt(gt, res["moves"])["acc"]
    return {"label": label, "solved": res["solved"],
            "cost": res.get("cost"), "gt_path_cost": gtc,
            "gap": (None if not res["solved"] else res["cost"] - gtc),
            "n_moves": len(res["moves"]) if res.get("moves") else 0,
            "rec_acc": rec_acc}


# ---------------------------------------------------------------------------
# One session, four conditions
# ---------------------------------------------------------------------------

def run_session(d: Path, args, tables):
    replay = RC._load_replay(d, args)
    if replay is None or not replay["moves"]:
        print(f"  {d.name}: no replay onsets -- skipping")
        return None

    gt_records = [json.loads(l) for l in open(d / "moves.jsonl") if l.strip()]
    gt = [r.get("wca_notation") for r in gt_records]
    if not gt or any(g is None for g in gt):
        print(f"  {d.name}: unresolved wca_notation in moves.jsonl -- skipping")
        return None
    move_ts = np.array([r["timestamp"] for r in gt_records])

    threshold = replay["meta"].get("threshold", 0.5)
    start = RC.start_from_gt(gt)
    end = RC.SOLVED.copy()

    load_color, n, fps, frame_ts = _load_frames_index(d)
    gt_onset = gt_onset_frames(move_ts, frame_ts)

    # -- real --------------------------------------------------------------
    _real_names, real_rows, real_del = RC.costs_from_moves(replay["moves"],
                                                            threshold)

    # -- oracle-classifier: TIME-match real onsets to truth, one-hot the
    # matched ones' softmax; unmatched (phantom) onsets keep their real
    # softmax -- truth has no label to give them. ---------------------------
    sbt = score_by_time(replay["moves"], gt_onset, gt, TOLERANCE)
    oc_rows = [row.copy() for row in real_rows]
    for gt_idx, pred_idx in sbt["pairs"].items():
        oc_rows[pred_idx] = RC.onset_costs(onehot_probs(gt[gt_idx]))

    # -- oracle-detector: onsets AT the true frames, real classifier rerun
    # there. ------------------------------------------------------------
    od_moves_raw, _ = classify_at_onsets(load_color, gt_onset, n, fps,
                                         frame_ts, args._detector,
                                         args.classifier)
    dropped = sum(1 for m in od_moves_raw if m is None)
    od_moves = [m for m in od_moves_raw if m is not None]
    if dropped:
        print(f"    {d.name}: {dropped} oracle-detector onset(s) fell "
              f"outside the recorded stream -- dropped")
    od_names, od_rows, od_del = RC.costs_from_moves(od_moves, threshold)
    od_gt = gt[:len(od_moves)] if len(od_moves) < len(gt) else gt

    # -- both: oracle onsets AND oracle softmax -----------------------------
    both_rows = [RC.onset_costs(onehot_probs(g)) for g in gt]
    both_del = np.full(len(gt), RC.C_DEL, dtype=np.float32)

    out = {"session": d.name, "n_gt": len(gt),
           "n_real_onsets": len(replay["moves"]),
           "n_od_dropped": dropped}
    out["real"] = decode_report("real", start, end, real_rows, real_del,
                                gt, args, tables)
    out["oracle_classifier"] = decode_report(
        "oracle-classifier", start, end, oc_rows, real_del, gt, args, tables)
    out["oracle_detector"] = decode_report(
        "oracle-detector", start, end, od_rows, od_del, od_gt, args, tables)
    out["both"] = decode_report("both", start, end, both_rows, both_del,
                               gt, args, tables)
    return out


def print_session(out: dict):
    print(f"\n  {out['session']}  ({out['n_gt']} true moves, "
          f"{out['n_real_onsets']} real onsets"
          + (f", {out['n_od_dropped']} oracle-detector onset(s) dropped"
             if out['n_od_dropped'] else "") + ")")
    print(f"    {'condition':<20} {'verified':>8} {'cost':>9} "
          f"{'gt_path_cost':>13} {'gap':>8} {'rec_acc':>8}")
    for key in ("real", "oracle_classifier", "oracle_detector", "both"):
        r = out[key]
        cost_s = "-" if r["cost"] is None else f"{r['cost']:.2f}"
        gap_s = "-" if r["gap"] is None else f"{r['gap']:+.2f}"
        acc_s = "-" if r["rec_acc"] is None else f"{r['rec_acc']*100:5.1f}%"
        print(f"    {r['label']:<20} {'YES' if r['solved'] else 'no':>8} "
              f"{cost_s:>9} {r['gt_path_cost']:>13.2f} {gap_s:>8} {acc_s:>8}")


def print_summary(results: list[dict]):
    results = [r for r in results if r is not None]
    if not results:
        print("\n  No sessions produced a result.")
        return
    print(f"\n{'='*70}")
    print(f"  SUMMARY -- {len(results)} honest session(s)")
    print(f"{'='*70}")
    print(f"    {'condition':<20} {'verified':>10}")
    for key, label in (("real", "real"),
                       ("oracle_classifier", "oracle-classifier"),
                       ("oracle_detector", "oracle-detector"),
                       ("both", "both")):
        n_ver = sum(r[key]["solved"] for r in results)
        print(f"    {label:<20} {n_ver:>6}/{len(results)}")

    real_gap = sum(r["both"]["gt_path_cost"] and
                   (0 if r["real"]["solved"] else 1) for r in results)
    print(f"\n  Per-session gt_path_cost under each oracle (lower = closer "
          f"to verifiable):")
    print(f"    {'session':<32} {'real':>8} {'oracle-cls':>11} "
          f"{'oracle-det':>11} {'both':>8}")
    for r in results:
        print(f"    {r['session']:<32} "
              f"{r['real']['gt_path_cost']:>8.1f} "
              f"{r['oracle_classifier']['gt_path_cost']:>11.1f} "
              f"{r['oracle_detector']['gt_path_cost']:>11.1f} "
              f"{r['both']['gt_path_cost']:>8.1f}")

    def frac_closed(key):
        # How much of the real->both gt_path_cost gap does this oracle
        # close, averaged over sessions where real didn't already verify.
        vals = []
        for r in results:
            real_c, both_c = r["real"]["gt_path_cost"], r["both"]["gt_path_cost"]
            this_c = r[key]["gt_path_cost"]
            span = real_c - both_c
            if span > 1e-6:
                vals.append((real_c - this_c) / span)
        return float(np.mean(vals)) if vals else float("nan")

    fc, fd = frac_closed("oracle_classifier"), frac_closed("oracle_detector")
    print(f"\n  Fraction of the (real -> both) gt_path_cost gap each oracle "
          f"closes, averaged\n  over sessions:")
    print(f"    oracle-classifier: {fc*100:5.1f}%")
    print(f"    oracle-detector:   {fd*100:5.1f}%")
    if not np.isnan(fc) and not np.isnan(fd):
        bigger = "oracle-detector (detector misses/phantoms)" if fd > fc \
            else "oracle-classifier (classifier substitutions)"
        print(f"\n  {bigger} closes more of the gap -- MODEL_REWORK_PLAN.md "
              f"Stage A should\n  prioritize that error stream first.")
    print(f"{'='*70}")


def main():
    p = argparse.ArgumentParser(
        description="G2: oracle attribution on the honest sessions "
                    "(MODEL_REWORK_PLAN.md).")
    p.add_argument("--sessions", nargs="+", default=None,
                   help="Session dirs/globs. Default: every honest "
                        "session under ../training_data, computed live.")
    p.add_argument("--detector", type=str, default=LD.DETECTOR_PATH)
    p.add_argument("--classifier", type=str, default=LD.CLASSIFIER_PATH)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--beam", type=int, default=RC.BEAM)
    p.add_argument("--retry-beam", type=int, default=4 * RC.BEAM)
    p.add_argument("--del-cost", type=float, default=RC.C_DEL)
    p.add_argument("--ins-cost", type=float, default=RC.C_INS)
    p.add_argument("--rot-cost", type=float, default=RC.C_ROT)
    p.add_argument("--max-end-ins", type=int, default=RC.MAX_END_INS)
    p.add_argument("--rel-weight", type=float, default=RC.REL_WEIGHT)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    args.candidate_threshold = None   # RC._load_replay reads this attr

    if args.sessions:
        dirs = sorted(d for pattern in args.sessions
                     for d in (Path(".").glob(pattern) if "*" in pattern
                              else [Path(pattern)]) if d.is_dir())
    else:
        dirs = honest_sessions(Path("../training_data"), args.classifier)
        print(f"  {len(dirs)} honest session(s) (classifier_unseen, "
              f"computed live against {args.classifier}):")
        for d in dirs:
            print(f"    {d.name}")

    if not dirs:
        sys.exit("No session directories found.")

    args._detector = load_detector()
    tables = RC.build_tables()

    results = []
    for d in dirs:
        out = run_session(d, args, tables)
        if out is not None:
            print_session(out)
        results.append(out)

    print_summary(results)

    if args.out:
        Path(args.out).write_text(
            json.dumps([r for r in results if r is not None], indent=2,
                      default=str))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
