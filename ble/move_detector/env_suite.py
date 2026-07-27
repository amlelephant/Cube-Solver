"""
env_suite.py

The cross-environment scorecard: how well does this pipeline read moves
HERE, in this room, on this cube, under this light — and how does that
compare to where it was trained?

Why a separate file from verify_solve.py
----------------------------------------
verify_solve.py answers "is this one video a genuine solve". This answers
"is the system any good here", which needs several takes rather than one,
needs the takes to probe the axes that actually break, and needs the
result written down in a form two rooms apart can be compared with.

Cross-environment performance is the number that matters and the one most
easily faked. Moving to new lighting cost ~11 points of speed-matched
detector recall on 2026-07-22 until sessions from that environment were in
training. A same-room score says very little about a new room, and an
average over mixed takes says very little about anything — so this suite
keeps the takes separate, labelled, and stored.

What it measures, and why each column exists
--------------------------------------------
    capture fps     FIRST, and it gates everything else. The detector
                    trained on 30fps; at 20fps recall falls to ~90%, at
                    15fps to ~85%, and it MISSES moves rather than
                    inventing them. A darker room makes a webcam expose
                    longer and drop frames, so "the new environment is
                    worse" and "the new environment is dimmer so capture
                    got slower" look identical unless fps is on the
                    scorecard. Takes below 28fps are flagged, not averaged
                    in silently.
    luminance /     What "environment" means numerically, measured on the
    contrast /      cube crop itself rather than the room. A nickname like
    sharpness       "kitchen-evening" is not a measurement; these are, and
                    they are what a later regression gets correlated
                    against.
    detector recall  } the three-way split from align_sequences: a move
    classifier acc   } that never fired is a detector problem, a move
    end-to-end       } named wrong is a classifier problem, and conflating
                     } them sends you to retrain the wrong model.
    decoded acc     what the group-theoretic decode recovers on top, and
    verified        whether the cube-state constraint closed at all.

Ground truth comes from a PRESCRIBED SCRAMBLE, so no smart cube is needed
and the measurement describes whatever cube you are holding — the label is
the instruction, not the hardware. That is the whole point for a
cross-environment test: the new environment is wherever you happen to be,
and the BLE cube may not be there.

Usage
-----
    # measure here, save a scorecard
    python env_suite.py --name kitchen-evening

    # baseline from already-recorded BLE sessions, no camera needed
    python env_suite.py --name desk-0721 --session ../training_data/solve_20260721_*/

    # then compare any set of scorecards
    python env_suite.py --compare scorecards/*.json

Run from inside move_detector/, same convention as the rest of the repo.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import live_detect as LD
import reconstruct as RC
import verify_solve as VS

SCORECARD_DIR = Path("scorecards")

# The standard plan. Three takes, each probing an axis that has actually
# broken this pipeline before, rather than three repetitions of the same
# easy case (which would only measure variance).
TAKES = [
    ("steady", 20,
     "Perform the scramble at a comfortable, even pace.",
     "The baseline. Everything else is read relative to this take."),
    ("fast", 20,
     "Perform the scramble AS FAST AS YOU COMFORTABLY CAN.",
     "Crowded moves are the detector's known weak spot — peaks 1-2 frames "
     "apart\n     merge, and aggregate F1 hides it because fast pairs are "
     "a small fraction\n     of any recording. This take makes them the "
     "majority."),
    ("offset", 20,
     "Hold the cube well off-centre — a corner of the frame — and "
     "perform the scramble.",
     "The classifier reads a cube crop, so it depends on the YOLO "
     "detector finding\n     the cube. This take separates 'the room is "
     "dark' from 'the cube is not\n     where the detector expects it'."),
]


# ---------------------------------------------------------------------------
# Measuring the environment itself
# ---------------------------------------------------------------------------

def crop_stats(load_color, boxes, n_frames: int, samples: int = 24) -> dict:
    """
    Luminance, contrast and focus of the CUBE CROP, not the whole frame.

    The crop is what both models actually see, and it is not
    interchangeable with the room: a bright room with the cube in its own
    shadow scores dark here, correctly. Sharpness is the variance of a
    Laplacian — the standard cheap focus measure — and it is the one that
    catches motion blur from a long exposure, which is how dim light
    degrades a move classifier that reads temporal diffs.
    """
    import cv2
    lum, con, sharp = [], [], []
    for i in np.linspace(0, n_frames - 1, min(samples, n_frames)).astype(int):
        f = load_color(int(i))
        if f is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in boxes[int(i)]]
        x1, y1 = max(x1, 0), max(y1, 0)
        crop = f[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lum.append(float(g.mean()))
        con.append(float(g.std()))
        sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    if not lum:
        return {}
    return {"luminance": round(statistics.median(lum), 1),
            "contrast": round(statistics.median(con), 1),
            "sharpness": round(statistics.median(sharp), 1)}


# ---------------------------------------------------------------------------
# Scoring one take
# ---------------------------------------------------------------------------

def score_take(truth: list[str], moves: list[dict], threshold: float,
               args, tables, start_state, end_state) -> dict:
    """
    Every number for one take: the three-way stage split, then the decode.

    Detector recall counts a move as FOUND if it aligned to any prediction,
    right or wrong — that is precisely the split between "never fired" and
    "fired, named wrong", and it is why this is alignment rather than
    index-by-index comparison (one missed move shifts every later index and
    would read as ~25 errors instead of 1).
    """
    pred = [m["move"] for m in moves]
    ops = LD.align_sequences(truth, pred)
    c = {k: sum(1 for o, _, _ in ops if o == k)
         for k in ("ok", "sub", "miss", "phantom")}
    found = c["ok"] + c["sub"]
    out = {
        "n_truth": len(truth), "n_pred": len(pred), **c,
        "det_recall": found / len(truth) if truth else 0.0,
        "cls_acc": c["ok"] / found if found else 0.0,
        "e2e": c["ok"] / len(truth) if truth else 0.0,
        "mean_conf": float(np.mean([m["conf"] for m in moves])) if moves else 0.0,
    }

    _, cost_rows, del_costs = RC.costs_from_moves(
        moves, threshold, args.blend_inv, args.blend_unif, args.del_cost)
    res = VS.decode_claim(start_state, end_state, cost_rows, del_costs,
                          args, tables)
    out["verified"] = res["solved"]
    if res["solved"]:
        s = RC.score_vs_gt(truth, res["moves"])
        out.update({"decoded_acc": s["acc"], "decoded_exact": s["exact"],
                    "decode_cost": res["cost"]})
    else:
        # The system falls back to the raw sequence when the decode does
        # not close, so that is the accuracy actually delivered.
        out.update({"decoded_acc": out["e2e"], "decoded_exact": False,
                    "decode_cost": None})
    return out


def print_take(name: str, t: dict) -> None:
    print(f"\n  {'-'*66}")
    print(f"  TAKE '{name}'   {t['n_truth']} prescribed, {t['n_pred']} "
          f"predicted"
          + (f"   {t['fps']:.1f}fps" if t.get("fps") else ""))
    print(f"    detector recall     {t['det_recall']*100:5.1f}%   "
          f"({t['miss']} never fired, {t['phantom']} phantom)")
    print(f"    classifier accuracy {t['cls_acc']*100:5.1f}%   "
          f"({t['sub']} named wrong of {t['ok']+t['sub']} found)")
    print(f"    END-TO-END          {t['e2e']*100:5.1f}%   "
          f"mean confidence {t['mean_conf']*100:.0f}%")
    verdict = "verified" if t["verified"] else "NOT verified"
    print(f"    after decode        {t['decoded_acc']*100:5.1f}%   "
          f"({verdict}"
          f"{', exact' if t.get('decoded_exact') else ''})")
    if t.get("luminance") is not None:
        print(f"    crop: luminance {t['luminance']:.0f}  "
              f"contrast {t['contrast']:.0f}  sharpness {t['sharpness']:.0f}")
    if t.get("fps", 30) < 28:
        print(f"    WARNING: captured at {t['fps']:.1f}fps, below the 30fps "
              f"the detector trained on.\n             Missed moves here are "
              f"a capture problem first, a model problem second.")


# ---------------------------------------------------------------------------
# Live suite
# ---------------------------------------------------------------------------

def run_live_suite(args, tables) -> dict:
    detector, det_model, device, threshold, min_sep = VS.load_stack(args)
    plan = [t for t in TAKES if not args.only or t[0] in args.only]
    if not plan:
        sys.exit(f"--only matched no takes; available: "
                 f"{', '.join(t[0] for t in TAKES)}")

    print(f"\n{'='*70}")
    print(f"  ENVIRONMENT SUITE — '{args.name}'")
    print(f"{'='*70}")
    print(f"  {len(plan)} take(s). Each one: you are shown a scramble, you")
    print(f"  perform it on a solved cube, and every stage is scored against")
    print(f"  it. Solve the cube again between takes.")
    print(f"\n  Hold ONE orientation within a take. The classifier names")
    print(f"  camera-relative layers, so rotating the whole cube relabels")
    print(f"  every move after it and the comparison stops meaning anything.")

    takes = {}
    for i, (name, n_moves, instruction, rationale) in enumerate(plan, 1):
        n = args.scramble or n_moves
        scramble = LD.generate_scramble(
            n, seed=(args.seed + i) if args.seed is not None else None)
        print(f"\n{'='*70}")
        print(f"  TAKE {i}/{len(plan)}: '{name}' — {n} moves")
        print(f"{'='*70}")
        print(f"  {instruction}")
        print(f"     {rationale}")
        print(f"\n  Start this take from a SOLVED cube. The detector,")
        print(f"  classifier and end-to-end columns do not depend on that,")
        print(f"  but the decode column does — it verifies against "
              f"solved -> scramble,")
        print(f"  and a cube that started elsewhere makes that claim false "
              f"through no")
        print(f"  fault of the models.")
        for line in LD.format_scramble(scramble):
            print(f"    {line}")

        src = VS.record_phase(args, f"RECORDING take '{name}'",
                              ["SPACE to start, perform the scramble, "
                               "SPACE to stop.  Q aborts this take."],
                              overlay=LD.format_scramble(scramble))
        if src is None:
            print(f"  take '{name}' skipped (nothing recorded)")
            continue
        load_color, n_frames, fps, _window, ftimes = src
        print(f"\n  analysing {n_frames} frames...")
        res = LD.analyse(load_color, n_frames, fps, detector, det_model,
                         device, threshold, min_sep, args.classifier,
                         frame_times=ftimes)
        if not res["moves"]:
            print(f"  no moves detected — take '{name}' scored as a total "
                  f"detector failure")
            takes[name] = {"n_truth": n, "n_pred": 0, "ok": 0, "sub": 0,
                           "miss": n, "phantom": 0, "det_recall": 0.0,
                           "cls_acc": 0.0, "e2e": 0.0, "mean_conf": 0.0,
                           "verified": False, "decoded_acc": 0.0,
                           "decoded_exact": False, "decode_cost": None,
                           "fps": fps, "scramble": scramble}
            continue
        if res["class_names"] != RC.WCA12:
            sys.exit(f"classifier class order {res['class_names']} != "
                     f"{RC.WCA12} — refusing to score")

        t = score_take(scramble, res["moves"], threshold, args, tables,
                       RC.SOLVED.copy(), RC.seq_to_state(scramble))
        t["fps"] = fps
        t["scramble"] = scramble
        t["duration_s"] = round(n_frames / fps, 1)
        t["moves_per_s"] = round(len(res["moves"]) / max(n_frames / fps, 1e-6), 2)
        t.update(crop_stats(load_color, res["boxes"], n_frames))
        print_take(name, t)
        takes[name] = t

    return _finish(args, takes, threshold)


# ---------------------------------------------------------------------------
# Offline suite — a baseline from already-recorded BLE sessions
# ---------------------------------------------------------------------------

def run_session_suite(args, tables) -> dict:
    """
    The same scorecard, computed from recorded sessions with the BLE move
    list standing in for the prescribed scramble.

    This is how you get a reference row for the environment the models were
    trained in without re-recording it — and the columns are computed by
    the same code as the live path, so the two rows are comparable. What it
    cannot report is capture fps as an independent variable (the session
    was captured once, at whatever rate it was captured at).
    """
    dirs = sorted(d for pattern in args.session
                  for p in (Path(".").glob(pattern) if "*" in pattern
                            else [Path(pattern)])
                  for d in [Path(p)] if d.is_dir())
    if not dirs:
        sys.exit("No session directories matched --session.")

    takes = {}
    threshold = 0.5
    for d in dirs:
        gt = [m.get("wca_notation")
              for m in (json.loads(l) for l in open(d / "moves.jsonl")
                        if l.strip())]
        if not gt or any(g is None for g in gt):
            print(f"  {d.name}: no wca_notation — skipping")
            continue
        replay = RC._load_replay(d, args)
        if replay is None or not replay["moves"]:
            continue
        threshold = replay["meta"].get("threshold", 0.5)
        # The BLE moves define the state the cube must have started in for
        # this recording to end solved — the same construction session
        # replay uses everywhere else in this repo.
        t = score_take(gt, replay["moves"], threshold, args, tables,
                       RC.start_from_gt(gt), RC.SOLVED.copy())
        t["scramble"] = gt
        t.update(_session_crop_stats(d))
        print_take(d.name, t)
        takes[d.name] = t

    return _finish(args, takes, threshold)


def _session_crop_stats(session_dir: Path, samples: int = 12) -> dict:
    """Crop stats for a recorded session, so its row is comparable."""
    import cv2
    from crop_utils import load_detector, detect_box, square_with_margin
    frames_dir = session_dir / "frames"
    if not frames_dir.is_dir():
        return {}
    paths = sorted(frames_dir.glob("*.jpg"))
    if not paths:
        return {}
    model = load_detector()
    lum, con, sharp = [], [], []
    for i in np.linspace(0, len(paths) - 1, min(samples, len(paths))).astype(int):
        img = cv2.imread(str(paths[int(i)]))
        if img is None:
            continue
        box = detect_box(model, img) if model is not None else None
        if box is not None:
            x1, y1, x2, y2 = square_with_margin(box, img.shape)
            img = img[max(int(y1), 0):int(y2), max(int(x1), 0):int(x2)]
        if img.size == 0:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lum.append(float(g.mean()))
        con.append(float(g.std()))
        sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    if not lum:
        return {}
    return {"luminance": round(statistics.median(lum), 1),
            "contrast": round(statistics.median(con), 1),
            "sharpness": round(statistics.median(sharp), 1)}


# ---------------------------------------------------------------------------
# Scorecard I/O + comparison
# ---------------------------------------------------------------------------

def _finish(args, takes: dict, threshold: float) -> dict:
    if not takes:
        sys.exit("No takes scored — nothing to save.")
    card = {
        "name": args.name,
        "recorded": datetime.now().isoformat(timespec="seconds"),
        "detector": Path(args.detector).name,
        "classifier": Path(args.classifier).name,
        "threshold": threshold,
        "takes": takes,
        "overall": aggregate(takes),
    }
    print_overall(card)
    SCORECARD_DIR.mkdir(exist_ok=True)
    path = SCORECARD_DIR / f"{args.name}.json"
    path.write_text(json.dumps(card, indent=2))
    print(f"\n  scorecard written to {path}")
    print(f"  compare with:  python env_suite.py --compare "
          f"{SCORECARD_DIR}/*.json")
    return card


def aggregate(takes: dict) -> dict:
    """
    Move-weighted, not take-weighted: a 20-move take and a 60-move take are
    not equal evidence, and averaging the percentages would treat them as
    if they were.
    """
    tot = sum(t["n_truth"] for t in takes.values())
    if not tot:
        return {}
    w = lambda key: sum(t[key] * t["n_truth"] for t in takes.values()) / tot
    fps = [t["fps"] for t in takes.values() if "fps" in t]
    lum = [t["luminance"] for t in takes.values() if t.get("luminance")]
    return {
        "n_takes": len(takes), "n_moves": tot,
        "det_recall": round(w("det_recall"), 4),
        "cls_acc": round(w("cls_acc"), 4),
        "e2e": round(w("e2e"), 4),
        "decoded_acc": round(w("decoded_acc"), 4),
        "verified": sum(t["verified"] for t in takes.values()),
        "fps": round(statistics.median(fps), 1) if fps else None,
        "luminance": round(statistics.median(lum), 1) if lum else None,
    }


def print_overall(card: dict) -> None:
    o = card["overall"]
    print(f"\n{'='*70}")
    print(f"  SCORECARD — '{card['name']}'   "
          f"{o['n_takes']} takes, {o['n_moves']} moves")
    print(f"{'='*70}")
    print(f"    detector recall      {o['det_recall']*100:5.1f}%")
    print(f"    classifier accuracy  {o['cls_acc']*100:5.1f}%")
    print(f"    end-to-end           {o['e2e']*100:5.1f}%")
    print(f"    after decode         {o['decoded_acc']*100:5.1f}%   "
          f"({o['verified']}/{o['n_takes']} takes verified)")
    if o.get("fps"):
        print(f"    median capture       {o['fps']:.1f}fps"
              + ("   <-- below 30fps, read every miss with this in mind"
                 if o["fps"] < 28 else ""))
    if o.get("luminance"):
        print(f"    median crop luminance {o['luminance']:.0f}")

    worst = min(card["takes"].items(), key=lambda kv: kv[1]["e2e"])
    best = max(card["takes"].items(), key=lambda kv: kv[1]["e2e"])
    if worst[0] != best[0]:
        print(f"\n    Best take '{best[0]}' {best[1]['e2e']*100:.1f}%, "
              f"worst '{worst[0]}' {worst[1]['e2e']*100:.1f}% — "
              f"a {(best[1]['e2e']-worst[1]['e2e'])*100:.1f} point spread "
              f"WITHIN\n    this environment. Any cross-environment "
              f"difference smaller than that\n    is not yet a difference.")


def compare(paths: list[str]) -> None:
    cards = []
    for pattern in paths:
        for p in (sorted(Path(".").glob(pattern)) if "*" in pattern
                  else [Path(pattern)]):
            if p.is_file():
                cards.append(json.loads(p.read_text()))
    if not cards:
        sys.exit("No scorecards matched --compare.")
    cards.sort(key=lambda c: c["recorded"])

    models = {(c["detector"], c["classifier"]) for c in cards}
    print(f"\n{'='*94}")
    print(f"  CROSS-ENVIRONMENT COMPARISON — {len(cards)} scorecard(s)")
    print(f"{'='*94}")
    if len(models) > 1:
        print(f"  WARNING: these scorecards were not all produced by the same "
              f"models. A row-to-row\n  difference here mixes 'different "
              f"room' with 'different weights' and cannot\n  separate them:")
        for c in cards:
            print(f"    {c['name']:<22} {c['detector']} + {c['classifier']}")
        print()

    hdr = (f"  {'environment':<22} {'takes':>5} {'moves':>6} {'fps':>5} "
           f"{'lum':>5} {'det':>7} {'cls':>7} {'e2e':>7} {'decoded':>8}")
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")
    for c in cards:
        o = c["overall"]
        print(f"  {c['name']:<22} {o['n_takes']:>5} {o['n_moves']:>6} "
              f"{(o.get('fps') or 0):>5.1f} "
              f"{(o.get('luminance') or 0):>5.0f} "
              f"{o['det_recall']*100:>6.1f}% {o['cls_acc']*100:>6.1f}% "
              f"{o['e2e']*100:>6.1f}% {o['decoded_acc']*100:>7.1f}%")

    base = cards[0]
    if len(cards) < 2:
        return
    print(f"\n  Read against the baseline row '{base['name']}':")
    for c in cards[1:]:
        d_det = (c["overall"]["det_recall"] - base["overall"]["det_recall"]) * 100
        d_cls = (c["overall"]["cls_acc"] - base["overall"]["cls_acc"]) * 100
        d_e2e = (c["overall"]["e2e"] - base["overall"]["e2e"]) * 100
        d_fps = (c["overall"].get("fps") or 0) - (base["overall"].get("fps") or 0)
        print(f"\n    {c['name']}: end-to-end {d_e2e:+.1f} pts "
              f"(detector {d_det:+.1f}, classifier {d_cls:+.1f})")
        # Attribute the change, in the order that decides what to do next.
        if d_fps < -2 and d_det < -2:
            print(f"      Capture ran {abs(d_fps):.1f}fps slower here and "
                  f"recall fell with it. Fix the\n      capture rate before "
                  f"concluding anything about the models — below 30fps\n"
                  f"      the detector misses moves rather than inventing "
                  f"them, which is exactly\n      what this row looks like.")
        elif d_det < -3 and d_det < d_cls:
            print(f"      Dominated by the DETECTOR. More sessions from this "
                  f"environment in\n      training is the fix "
                  f"(move_detector/train.py --holdout session\n"
                  f"      --val-session-names, one per environment).")
        elif d_cls < -3:
            print(f"      Dominated by the CLASSIFIER. Record sessions here "
                  f"and retrain it\n      (train_move_classifier.py "
                  f"--val-session-names, one per environment) —\n"
                  f"      the decode only has ~one insertion of headroom "
                  f"past classifier error.")
        elif abs(d_e2e) <= 3:
            print(f"      Within the spread a single environment shows "
                  f"across its own takes.\n      Not yet a difference.")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Cross-environment accuracy scorecard for the "
                    "detector + classifier + decode pipeline")
    p.add_argument("--name", type=str, default=None,
                   help="Short label for this environment, e.g. "
                        "kitchen-evening. Becomes the scorecard filename.")
    p.add_argument("--compare", nargs="+", default=None,
                   help="Print a comparison table of saved scorecards "
                        "(globs ok) instead of measuring")
    p.add_argument("--session", nargs="+", default=None,
                   help="Score recorded session(s) with BLE truth instead "
                        "of recording live — how to get a baseline row for "
                        "the training environment")
    p.add_argument("--only", nargs="+", default=None,
                   help=f"Run only these takes: "
                        f"{', '.join(t[0] for t in TAKES)}")
    p.add_argument("--scramble", type=int, default=None,
                   help="Override the per-take scramble length")
    p.add_argument("--detector", type=str, default=LD.DETECTOR_PATH)
    p.add_argument("--classifier", type=str, default=LD.CLASSIFIER_PATH)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seed", type=int, default=None,
                   help="Seed the scrambles so two environments can be "
                        "measured on the SAME move sequences")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--min-sep", type=int, default=None, dest="min_sep")
    p.add_argument("--no-resample", action="store_true")
    p.add_argument("--refresh-cache", action="store_true")
    # Decoder knobs, forwarded to reconstruct.py.
    p.add_argument("--beam", type=int, default=RC.BEAM)
    p.add_argument("--retry-beam", type=int, default=RC.BEAM)
    p.add_argument("--del-cost", type=float, default=RC.C_DEL)
    p.add_argument("--ins-cost", type=float, default=RC.C_INS)
    p.add_argument("--rot-cost", type=float, default=RC.C_ROT)
    p.add_argument("--max-end-ins", type=int, default=RC.MAX_END_INS)
    p.add_argument("--blend-inv", type=float, default=RC.BLEND_INV)
    p.add_argument("--blend-unif", type=float, default=RC.BLEND_UNIF)
    p.add_argument("--rel-weight", type=float, default=RC.REL_WEIGHT)
    p.add_argument("--rotations", action="store_true")
    args = p.parse_args()

    if args.compare:
        compare(args.compare)
        return
    if not args.name:
        sys.exit("--name is required: scorecards are only useful if you can "
                 "tell them apart later.\nTry --name kitchen-evening, or "
                 "--compare scorecards/*.json to read existing ones.")

    tables = RC.build_tables()
    if args.session:
        run_session_suite(args, tables)
    else:
        run_live_suite(args, tables)


if __name__ == "__main__":
    main()
