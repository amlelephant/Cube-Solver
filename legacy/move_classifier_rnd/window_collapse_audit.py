"""
window_collapse_audit.py

Measures what the collapsed move window actually costs, before anyone
spends time fixing it.

The defect (see PATH_TO_VERIFICATION.md 2h): move_window() sandwiches the
five window slots into the gaps between neighbouring moves, scaling the
three mid offsets by post/BASE_POST. The offsets shrink continuously; the
frames do not. At 30fps two scaled offsets less than ~17ms apart resolve to
the SAME frame, and the diff across that pair is exactly the neutral image
— an input channel carrying nothing.

Why the obvious measurement is the wrong one
--------------------------------------------
Comparing accuracy on fast moves against slow moves does NOT measure this.
Fast moves are harder for reasons that have nothing to do with the window:
more motion blur, more detector crowding, sloppier turns. That comparison
would confound the window defect with move difficulty and would report a
large effect no matter what.

What this does instead: compare, WITHIN a gap bucket, the windows that
happened to collapse against the ones that did not. Whether two slots round
onto the same frame depends on the sub-frame phase of the anchor against
the camera clock, which is independent of how hard the move is. Inside a
narrow bucket that is close to a natural experiment, and the difference is
attributable to the dead channels rather than to difficulty.

Reported separately, because they have different fixes:
  * collapse   — two slots resolving to one frame. Ours, fixable in
                 window selection.
  * camera dup — the webcam returning the same frame twice (4-6% of
                 consecutive frames). Not fixable here; sets the floor.
Both are counted the same way (pixel-identical consecutive window frames),
so they are separated by gap: at gaps > COLLAPSE_FREE_GAP the geometry
cannot collide, so anything dead there is the camera.

Usage:
  cd ble
  python window_collapse_audit.py --sessions training_data/solve_*/
  python window_collapse_audit.py --sessions training_data/solve_*/ \
      --model move_classifier_all39_jitter.pt --held-out-only
"""

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Subset
except ImportError:
    sys.exit("PyTorch not installed.")

# Archived 2026-08-03 from ble/ into legacy/move_classifier_rnd/ — see
# viz_encodings.py's identical bootstrap comment for why this is needed.
_BLE_DIR = Path(__file__).resolve().parents[2] / "ble"
if str(_BLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLE_DIR))

from train_move_classifier import (MoveDiffDataset, build_model, _load_state,
                                   ckpt_encoding, LABEL_MODES)
from encodings_move import get as get_encoding

# Above this neighbour gap the scaled offsets cannot collide on a 30fps
# grid (30ms and 40ms nominal spacing, scale ~0.83). Derived in the doc;
# re-derived by --explain here so it cannot silently drift from the code.
COLLAPSE_FREE_GAP = 0.42

DUP_TOL = 0.5      # mean |Δ| below this = the same frame


def dup_pairs(paths: list[Path], box) -> int:
    """How many consecutive window frames are pixel-identical."""
    ims = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    if any(i is None for i in ims):
        return -1
    return sum(1 for a, b in zip(ims, ims[1:])
               if float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
               < DUP_TOL)


def move_gaps(session_dir: Path) -> dict[int, float]:
    """move_num -> gap to the NEAREST neighbouring move, in seconds."""
    f = session_dir / "moves_labeled.jsonl"
    if not f.exists():
        return {}
    ms = [json.loads(l) for l in open(f) if l.strip()]
    out = {}
    for i, m in enumerate(ms):
        g = []
        if i > 0:
            g.append(m["timestamp"] - ms[i - 1]["timestamp"])
        if i < len(ms) - 1:
            g.append(ms[i + 1]["timestamp"] - m["timestamp"])
        out[m["move_num"]] = min(g) if g else 9.9
    return out


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """
    Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]].
    Independent groups (clean vs collapsed windows are different moves),
    so this rather than McNemar.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c
    def p(k):
        return (comb(row1, k) * comb(n - row1, col1 - k)) / comb(n, col1)
    obs = p(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(p(k) for k in range(lo, hi + 1)
                        if p(k) <= obs * (1 + 1e-9)))


def bucket(g: float) -> str:
    if g < 0.25:
        return "<250ms"
    if g < 0.40:
        return "250-400ms"
    if g < 0.60:
        return "400-600ms"
    return ">600ms"


BUCKETS = ["<250ms", "250-400ms", "400-600ms", ">600ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--model", default="move_classifier_all39_jitter.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--held-out-only", action="store_true", default=True,
                    help="Score only the sessions the checkpoint held out "
                         "(default). Training sessions are memorised and "
                         "would hide the effect.")
    ap.add_argument("--all-sessions", dest="held_out_only",
                    action="store_false")
    args = ap.parse_args()

    session_dirs = [Path(p) for pat in args.sessions
                    for p in (Path(".").glob(pat) if "*" in pat else [Path(pat)])]
    session_dirs = [d for d in session_dirs if d.is_dir()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    enc_name = ckpt_encoding(ckpt)
    label_mode = ckpt.get("label_mode", "wca")
    held = ckpt.get("val_session_names") or []

    if args.held_out_only:
        if not held:
            sys.exit("Checkpoint records no held-out sessions; use "
                     "--all-sessions and read the numbers as leaky.")
        session_dirs = [d for d in session_dirs if d.name in set(held)]

    ds = MoveDiffDataset(session_dirs, augment=False, label_mode=label_mode,
                         encoding=enc_name)
    if not len(ds):
        sys.exit("No samples.")

    enc = get_encoding(enc_name)
    model = build_model(device, in_channels=enc.channels)
    _load_state(model, ckpt["state_dict"])
    model.eval()

    print(f"\n{'='*70}")
    print(f"  Window-collapse audit")
    print(f"{'='*70}")
    print(f"  Model:     {args.model}  ({enc_name}, {enc.channels}ch)")
    print(f"  Sessions:  {len(session_dirs)} "
          f"({'HELD OUT' if args.held_out_only else 'ALL — leaky'})")
    print(f"  Moves:     {len(ds)}")

    # Predictions
    correct = np.zeros(len(ds), dtype=bool)
    preds   = np.zeros(len(ds), dtype=int)
    truths  = np.zeros(len(ds), dtype=int)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=8)
    i = 0
    with torch.no_grad():
        for imgs, labels in loader:
            pred = model(imgs.to(device)).argmax(1).cpu().numpy()
            n = len(pred)
            correct[i:i + n] = pred == labels.numpy()
            preds[i:i + n]   = pred
            truths[i:i + n]  = labels.numpy()
            i += n

    # Per-sample gap + dead-pair count
    gaps_by_session = {d.name: move_gaps(d) for d in session_dirs}
    rows = []
    for j, (paths, label, box) in enumerate(ds.samples):
        sess = ds.sample_session[j]
        mn = int(paths[0].name.split("_")[1])
        g = gaps_by_session.get(sess, {}).get(mn)
        if g is None:
            continue
        d = dup_pairs(paths, box)
        if d < 0:
            continue
        rows.append((g, d, bool(correct[j]), int(truths[j]), int(preds[j])))

    print(f"  Scored:    {len(rows)} moves with a known neighbour gap\n")

    # ---- 1. the raw (confounded) view, shown so it can be dismissed -----
    print(f"  1. Accuracy by gap — CONFOUNDED, fast moves are harder for "
          f"many reasons")
    print(f"     {'gap':<12}{'n':>5}{'acc':>9}{'mean dead':>11}")
    for b in BUCKETS:
        r = [x for x in rows if bucket(x[0]) == b]
        if not r:
            continue
        print(f"     {b:<12}{len(r):>5}{np.mean([x[2] for x in r])*100:>8.1f}%"
              f"{np.mean([x[1] for x in r]):>11.2f}")

    # ---- 2. the controlled view ----------------------------------------
    print(f"\n  2. WITHIN each gap bucket: collapsed vs clean windows")
    print(f"     (whether slots collide depends on sub-frame anchor phase, "
          f"not on\n      how hard the move is — so this isolates the dead "
          f"channels)")
    print(f"     {'gap':<12}{'clean n':>9}{'acc':>8}{'dead n':>9}{'acc':>8}"
          f"{'delta':>9}{'p':>9}")
    tot = [0, 0, 0, 0]      # clean_ok, clean_n, dead_ok, dead_n
    for b in BUCKETS:
        r = [x for x in rows if bucket(x[0]) == b]
        clean = [x[2] for x in r if x[1] == 0]
        dead  = [x[2] for x in r if x[1] > 0]
        if len(clean) < 5 or len(dead) < 5:
            print(f"     {b:<12}{len(clean):>9}{'':>8}{len(dead):>9}"
                  f"{'':>8}{'too few':>9}")
            continue
        ca, cn = sum(clean), len(clean)
        da, dn = sum(dead), len(dead)
        p = fisher_exact_greater(ca, cn - ca, da, dn - da)
        tot = [tot[0] + ca, tot[1] + cn, tot[2] + da, tot[3] + dn]
        print(f"     {b:<12}{cn:>9}{ca/cn*100:>7.1f}%{dn:>9}{da/dn*100:>7.1f}%"
              f"{(da/dn - ca/cn)*100:>+8.1f}{p:>9.3f}")

    if tot[1] and tot[3]:
        ca, cn, da, dn = tot
        p = fisher_exact_greater(ca, cn - ca, da, dn - da)
        print(f"     {'-'*62}")
        print(f"     {'pooled':<12}{cn:>9}{ca/cn*100:>7.1f}%{dn:>9}"
              f"{da/dn*100:>7.1f}%{(da/dn - ca/cn)*100:>+8.1f}{p:>9.3f}")

        # ---- 3. what fixing it could buy -------------------------------
        # Only the COLLAPSE share is addressable; camera duplication is not.
        short = [x for x in rows if x[0] < COLLAPSE_FREE_GAP]
        long_ = [x for x in rows if x[0] >= COLLAPSE_FREE_GAP]
        dead_rate_short = np.mean([x[1] > 0 for x in short]) if short else 0
        dead_rate_long  = np.mean([x[1] > 0 for x in long_]) if long_ else 0
        addressable = max(0.0, dead_rate_short - dead_rate_long) * \
            (len(short) / len(rows))
        delta = ca / cn - da / dn
        print(f"\n  3. Ceiling on a fix (upper bound, assumes a perfect "
              f"window selector)")
        print(f"     windows with a dead pair: {dead_rate_short*100:.0f}% at "
              f"gap<{COLLAPSE_FREE_GAP*1000:.0f}ms vs "
              f"{dead_rate_long*100:.0f}% at gap>={COLLAPSE_FREE_GAP*1000:.0f}ms")
        print(f"     the excess is collapse (ours); the rest is camera "
              f"duplication (not fixable here)")
        print(f"     addressable share of all moves : {addressable*100:.1f}%")
        print(f"     measured cost per dead window  : {delta*100:.1f} points")
        print(f"     => optimistic ceiling          : "
              f"{addressable*delta*100:.2f} points of move accuracy")

        # ---- 4. is the proposed MECHANISM even operating? --------------
        # A dead channel destroys temporal ORDER, and order is what
        # separates X from X' — the face is visible in any single diff.
        # So if collapse is doing what the story says, its errors should be
        # disproportionately DIRECTION errors (same face, wrong way).
        # If the error mix is unchanged, the mechanism is not operating and
        # the small aggregate effect is something else entirely.
        print(f"\n  4. Error TYPE — the mechanism check")
        print(f"     dead channels destroy temporal order, and order is "
              f"what separates X from X'.\n     If collapse works the way "
              f"the story says, its errors skew to direction.")
        print(f"     {'window':<12}{'errors':>8}{'direction':>11}"
              f"{'wrong face':>12}{'dir share':>11}")
        for tag, sel in (("clean", lambda r: r[1] == 0),
                         ("collapsed", lambda r: r[1] > 0)):
            errs = [r for r in rows if sel(r) and not r[2]]
            if not errs:
                continue
            # classes are [face*2 + direction], so //2 is the face
            dir_err = sum(1 for r in errs if r[3] // 2 == r[4] // 2)
            face_err = len(errs) - dir_err
            print(f"     {tag:<12}{len(errs):>8}{dir_err:>11}{face_err:>12}"
                  f"{dir_err/len(errs)*100:>10.0f}%")


if __name__ == "__main__":
    main()
