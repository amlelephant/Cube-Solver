"""
phantom_confidence.py

Asks one question: when the detector fires on nothing, does the classifier
already tell us?

Why it matters
--------------
Every classifier in this repo is trained on windows anchored on BLE move
TIMESTAMPS — i.e. on windows that are guaranteed to contain a real move.
At inference the anchor comes from the detector instead, and the detector
fires on things that are not moves. On the 10 live takes that is ~69
phantom detections against 577 matched ones: roughly **one classifier call
in ten is on a window containing no move at all**.

The classifier has 12 classes and a softmax. It cannot answer "nothing
happened here" — it has to spread mass over twelve real moves, none of
which is correct. `--anchor-jitter` does not cover this: its offset PMF was
measured over MATCHED moves only (metric_audit's `time.pairs`), so it is a
distribution over good detections, censored of exactly the failure mode in
question.

This also connects to the decoder. Phantoms are INSERTIONS, and insertions
are what the 2026-07-27 decoder sprint measured as blowing the budget on
the stubborn sessions (9-30 insertion-units over). A cheap, reliable
"this is not a move" signal would let the decoder delete them.

The measurement decides which fix is warranted
----------------------------------------------
  * If phantom windows already come out at clearly lower confidence, the
    information exists and the fix is decoder-side and cheap: feed
    confidence into the deletion cost. No retrain.
  * If phantom confidence looks like matched confidence, the classifier is
    confidently naming noise, the information is absent, and the fix is a
    13th null class — which is a retrain, but trainable for free since BLE
    truth says exactly when moves did NOT happen.

Usage:
  cd ble/move_detector
  python phantom_confidence.py --sessions ../training_data/solve_2026072[56]*/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_BLE_DIR = Path(__file__).resolve().parents[1]
if str(_BLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLE_DIR))

import cv2                                                        # noqa: E402
import torch                                                      # noqa: E402

from decode import TOLERANCE, MIN_SEP                             # noqa: E402
from model import build_model                                     # noqa: E402
from live_detect import analyse, DETECTOR_PATH, CLASSIFIER_PATH   # noqa: E402
from metric_audit import gt_onset_frames, score_by_time           # noqa: E402


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--detector", default=DETECTOR_PATH)
    ap.add_argument("--classifier", default=CLASSIFIER_PATH)
    ap.add_argument("--tolerance", type=int, default=TOLERANCE)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-sep", type=int, default=None, dest="min_sep")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dirs = [Path(p) for pat in args.sessions
            for p in (Path(".").glob(pat) if "*" in pat else [Path(pat)])]
    dirs = [d for d in dirs if d.is_dir()]

    # Detector set up exactly as metric_audit.main does it, including taking
    # the operating point from the checkpoint rather than a local default —
    # a different threshold would change the phantom count itself.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dckpt = torch.load(args.detector, map_location=device)
    det_model = build_model(device)
    det_model.load_state_dict(dckpt["state_dict"])
    det_model.eval()
    if args.threshold is None:
        args.threshold = dckpt.get("threshold", 0.5)
    if args.min_sep is None:
        args.min_sep = dckpt.get("min_sep", MIN_SEP)

    from crop_utils import load_detector
    cube_det = load_detector()
    if cube_det is None:
        sys.exit("Cube detector unavailable (needs ultralytics + "
                 "cv/detection/detect_full_cube.pt).")

    print(f"\nDetector:   {args.detector}  (threshold {args.threshold}, "
          f"min_sep {args.min_sep})")
    print(f"Classifier: {args.classifier}")
    print(f"Sessions:   {len(dirs)}\n")

    conf = {"ok": [], "sub": [], "phantom": []}
    score = {"ok": [], "sub": [], "phantom": []}
    per_session = []

    for d in dirs:
        # Truth loaded exactly the way metric_audit.audit_session does it,
        # so the ok/sub/miss/phantom split here is the same split those
        # numbers came from.
        fidx, mpath = d / "frames.jsonl", d / "moves.jsonl"
        if not ((d / "frames").is_dir() and fidx.exists() and mpath.exists()):
            print(f"  {d.name}: needs frames/, frames.jsonl, moves.jsonl "
                  f"— skipped")
            continue
        recs = [json.loads(l) for l in open(fidx) if l.strip()]
        recs = [r for r in recs if (d / "frames" / r["file"]).exists()]
        if len(recs) < 2:
            continue
        frame_ts = np.array([r["ts"] for r in recs], dtype=np.float64)

        gtm = [json.loads(l) for l in open(mpath) if l.strip()]
        gtm = [m for m in gtm if m.get("wca_notation")]
        if not gtm:
            print(f"  {d.name}: no wca_notation in moves.jsonl — skipped")
            continue
        labels = [m["wca_notation"] for m in gtm]
        gt = gt_onset_frames(np.array([m["timestamp"] for m in gtm],
                                      dtype=np.float64), frame_ts)

        paths = [d / "frames" / r["file"] for r in recs]
        n = len(paths)
        fps = n / (frame_ts[-1] - frame_ts[0])
        cache: dict[int, np.ndarray] = {}

        def load_color(i, _paths=paths, _cache=cache):
            if i not in _cache:
                if len(_cache) > 600:
                    _cache.clear()
                _cache[i] = cv2.imread(str(_paths[i]))
            return _cache[i]

        res = analyse(load_color, n, fps, cube_det, det_model, device,
                      args.threshold, args.min_sep, args.classifier,
                      verbose=False, frame_times=frame_ts)
        moves = res["moves"]
        if not moves:
            continue
        sc = score_by_time(moves, gt, labels, args.tolerance)

        matched_pi = set(sc["pairs"].values())
        wrong_pi = {sc["pairs"][w[0]] for w in sc["wrong"]}
        for pi, m in enumerate(moves):
            if pi not in matched_pi:
                k = "phantom"
            elif pi in wrong_pi:
                k = "sub"
            else:
                k = "ok"
            conf[k].append(m["conf"])
            score[k].append(m["score"])

        per_session.append({"session": d.name, **{k: sc[k] for k in
                            ("ok", "sub", "miss", "phantom")}})
        print(f"  {d.name:<38} ok {sc['ok']:>3}  sub {sc['sub']:>3}  "
              f"miss {sc['miss']:>3}  phantom {sc['phantom']:>3}")

    n_calls = sum(len(v) for v in conf.values())
    if not n_calls:
        sys.exit("No classifier calls scored.")

    print(f"\n{'='*74}")
    print(f"  CLASSIFIER CONFIDENCE by what the detection actually was")
    print(f"{'='*74}")
    print(f"  {'':<10}{'n':>6}{'share':>8}{'mean':>8}{'median':>8}"
          f"{'p10':>8}{'p25':>8}{'<0.5':>8}")
    for k in ("ok", "sub", "phantom"):
        v = np.array(conf[k])
        if not len(v):
            continue
        print(f"  {k:<10}{len(v):>6}{len(v)/n_calls*100:>7.1f}%"
              f"{v.mean():>8.3f}{np.median(v):>8.3f}{pct(v,10):>8.3f}"
              f"{pct(v,25):>8.3f}{(v<0.5).mean()*100:>7.0f}%")

    print(f"\n  DETECTOR onset score, same split (is the detector itself "
          f"less sure?)")
    for k in ("ok", "sub", "phantom"):
        v = np.array(score[k])
        if len(v):
            print(f"  {k:<10}{len(v):>6}{'':>8}{v.mean():>8.3f}"
                  f"{np.median(v):>8.3f}{pct(v,10):>8.3f}{pct(v,25):>8.3f}")

    # Separability: could a threshold actually remove phantoms without
    # throwing away real moves? That is the whole question — and it is
    # asked of BOTH available signals, because they are not equivalent.
    def auc_of(real, ph):
        allv = np.concatenate([real, ph])
        lab = np.concatenate([np.ones(len(real)), np.zeros(len(ph))])
        order = np.argsort(allv)
        ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv)+1)
        return (ranks[lab == 1].sum() - len(real)*(len(real)+1)/2) / \
               (len(real)*len(ph))

    for tag, src, grid in (
            ("CLASSIFIER CONFIDENCE", conf, (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)),
            ("DETECTOR ONSET SCORE",  score, (0.3, 0.35, 0.4, 0.45, 0.5, 0.6))):
        real = np.array(src["ok"] + src["sub"])
        ph = np.array(src["phantom"])
        if not (len(ph) and len(real)):
            continue
        a = auc_of(real, ph)
        print(f"\n  DROPPING DETECTIONS BELOW A {tag} THRESHOLD")
        print(f"  {'thresh':>8}{'phantoms cut':>16}{'real moves lost':>18}"
              f"{'net':>8}")
        for t in grid:
            cut, lost = int((ph < t).sum()), int((real < t).sum())
            print(f"  {t:>8.2f}{cut:>9} /{len(ph):<5}{lost:>12} /{len(real):<5}"
                  f"{cut-lost:>+8}")
        print(f"  AUC as a real-vs-phantom discriminator: {a:.3f}"
              f"   (0.5 = useless)")

    # The two signals are only worth combining if they disagree. If the
    # detector score already dominates, a null class buys nothing.
    rs = np.array(score["ok"] + score["sub"]); ps = np.array(score["phantom"])
    rc = np.array(conf["ok"] + conf["sub"]);   pc = np.array(conf["phantom"])
    if len(ps) and len(rs):
        both = auc_of(rs * rc, ps * pc)
        print(f"\n  AUC of score x confidence combined: {both:.3f}")
        print(f"  (compare against each alone above — if combining does not "
              f"beat the\n   better single signal, the classifier adds "
              f"nothing the detector lacks)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"sessions": per_session,
             "conf": {k: list(map(float, v)) for k, v in conf.items()}},
            indent=1))
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
