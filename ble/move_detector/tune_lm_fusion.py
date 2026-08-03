"""
tune_lm_fusion.py

Picks the shallow-fusion weights (alpha, beta) for ctc_decode.
prefix_beam_decode + move_lm.MoveLM, on a dev split, so that the honest
holdout is never used to choose them.

The split discipline, which is the whole point of this file
------------------------------------------------------------
There are three roles and they must not be collapsed:

  LM-fit sessions    (default: 30)  build the n-gram
  dev sessions       (default:  8)  choose alpha/beta
  holdout sessions   (        4)    NEVER touched here

Both the LM-fit and dev sets come from the sessions the ACOUSTIC model
already trained on, so absolute MER printed here is optimistic and should
never be quoted. That is fine for its purpose: alpha/beta is a two-
parameter balance between two scores, and that balance transfers even when
the level does not.

The LM is deliberately fit WITHOUT the dev sessions. An LM that had seen
the dev solves would score them far too well, and the sweep would then
choose an alpha that is much too high — tuning the fusion weight on
leakage rather than on prior quality.

Usage:
    python tune_lm_fusion.py --model checkpoints/move_ctc_s0.pt
    python tune_lm_fusion.py --model checkpoints/move_ctc_s0.pt --order 4 --beam 16
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from model import build_joint_from_ckpt, score_stream_joint
from dataset import JointSessionStream
from ctc_decode import prefix_beam_decode, move_error_rate
from move_lm import MoveLM, load_truth, SESSION_ROOT


def cache_posteriors(names, model, device):
    """Score each session once; the sweep then only re-runs the decoder."""
    out = []
    for n in names:
        d = SESSION_ROOT / n
        p = d / "detector_stream_color.npz"
        gt = load_truth(d)
        if not p.exists() or not gt:
            continue
        s = JointSessionStream(p)
        _, class_prob, _ = score_stream_joint(model, s, device)
        out.append((n, np.log(np.maximum(class_prob, 1e-12)), gt))
    return out


def sweep_point(cached, lm, alpha, beta, beam):
    s = i = d = n_true = 0
    for _, lp, gt in cached:
        labels, _ = prefix_beam_decode(lp, beam=beam,
                                       lm=lm if alpha else None,
                                       alpha=alpha, beta=beta)
        _, parts = move_error_rate(labels, gt)
        s += parts["sub"]; i += parts["ins"]; d += parts["del"]
        n_true += parts["n_true"]
    return {"mer": (s + i + d) / max(n_true, 1), "sub": s, "ins": i,
            "del": d, "n_true": n_true}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="checkpoints/move_ctc_s0.pt")
    p.add_argument("--order", type=int, default=4,
                   help="n-gram order; 4 and 5 tie on dev perplexity "
                        "(6.635 vs 6.634), so 4 is taken for half the "
                        "contexts — see move_lm.py")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--dev", type=int, default=8,
                   help="fallback: carve this many dev sessions off the end "
                        "of train. Only useful if --dev-sessions is unset, "
                        "and largely useless in practice — the acoustic "
                        "model scores its own training sessions at ~0%% MER, "
                        "leaving nothing for fusion to improve.")
    p.add_argument("--dev-sessions", nargs="+", default=None,
                   help="explicit dev session names that NO model has seen "
                        "(preferred). These carry real errors, so alpha/beta "
                        "is chosen against the regime it will face.")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location=device)
    model = build_joint_from_ckpt(ck, device); model.eval()

    train = list(ck["train_session_names"])
    if args.dev_sessions:
        # Sessions the acoustic model has never seen: the LM may then be
        # fit on ALL of train with no leakage onto dev.
        dev_names, fit_names = list(args.dev_sessions), train
    else:
        dev_names, fit_names = train[-args.dev:], train[:-args.dev]

    lm = MoveLM.from_sessions(fit_names, order=args.order)
    print(f"\n  model {args.model}  (epoch {ck['epoch']}, seed "
          f"{ck.get('seed','?')})")
    print(f"  LM     order {args.order}, fit on {lm.n_sequences} sessions / "
          f"{lm.n_moves} moves  (dev sessions EXCLUDED)")

    t0 = time.time()
    cached = cache_posteriors(dev_names, model, device)
    print(f"  dev    {len(cached)} sessions, "
          f"{sum(len(g) for _, _, g in cached)} moves "
          f"({time.time()-t0:.0f}s to score)")
    dev_seqs = [g for _, _, g in cached]
    print(f"  dev perplexity under this LM: {lm.perplexity(dev_seqs):.2f} "
          f"(uniform = 12.0)\n")

    base = sweep_point(cached, lm, 0.0, 0.0, args.beam)
    print(f"  unfused baseline: MER {base['mer']*100:.2f}%  "
          f"(sub {base['sub']} / ins {base['ins']} / del {base['del']})\n")
    print(f"  {'alpha':<7} {'beta':<7} {'MER':<9} {'sub':<6} {'ins':<6} "
          f"{'del':<6} {'vs base'}")
    print(f"  {'-'*58}")

    rows, best = [], None
    for a in args.alphas:
        for b in args.betas:
            if a == 0.0 and b != 0.0:
                continue          # beta without the LM is not fusion
            r = sweep_point(cached, lm, a, b, args.beam)
            r.update(alpha=a, beta=b)
            rows.append(r)
            delta = (r["mer"] - base["mer"]) * 100
            mark = ""
            if best is None or r["mer"] < best["mer"]:
                best = r
                mark = "  <- best"
            print(f"  {a:<7.2f} {b:<7.2f} {r['mer']*100:<8.2f}% {r['sub']:<6} "
                  f"{r['ins']:<6} {r['del']:<6} {delta:+.2f}pt{mark}")

    print(f"\n  {'='*58}")
    print(f"  CHOSEN  alpha={best['alpha']:.2f}  beta={best['beta']:.2f}")
    print(f"  dev MER {base['mer']*100:.2f}% -> {best['mer']*100:.2f}%   "
          f"ins {base['ins']} -> {best['ins']}   del {base['del']} -> "
          f"{best['del']}   sub {base['sub']} -> {best['sub']}")
    if args.dev_sessions:
        print(f"\n  Dev sessions were unseen by the acoustic model, so the")
        print(f"  level here is meaningful — but it is still a TUNING set:")
        print(f"  quote the holdout, not this.")
    else:
        print(f"\n  These are DEV numbers on sessions the acoustic model")
        print(f"  trained on — optimistic in level, and not to be quoted.")
        print(f"  Only the chosen (alpha, beta) carries over to the holdout.")

    edge = []
    if best["alpha"] in (min(args.alphas), max(args.alphas)):
        edge.append("alpha")
    if best["beta"] in (min(args.betas), max(args.betas)):
        edge.append("beta")
    if edge:
        print(f"\n  WARNING: the optimum sits at the grid edge in "
              f"{' and '.join(edge)}. Widen the grid before trusting it — "
              f"the true optimum may be outside what was searched.")

    if args.out:
        import json
        Path(args.out).write_text(json.dumps(
            {"model": args.model, "order": args.order, "beam": args.beam,
             "dev_sessions": [n for n, _, _ in cached],
             "lm_fit_sessions": fit_names,
             "baseline": base, "best": best, "grid": rows}, indent=2,
            default=float))
        print(f"\n  Written to {args.out}")


if __name__ == "__main__":
    main()
