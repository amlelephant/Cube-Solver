"""
move_lm.py

An n-gram language model over WCA quarter-turn sequences, built from BLE
ground truth — the statistical form of the "algorithm prior", intended to
be fused into ctc_decode.prefix_beam_decode rather than applied as a
post-hoc repair.

Why an n-gram and not algorithms.py's matcher
---------------------------------------------
ALGORITHM_PRIOR.md §6b measured the curated-library matcher and found it
**reaches only 21% of errors**, for a structural reason: recognising a
15-move algorithm requires its neighbourhood to be read correctly, and an
error is precisely where the neighbourhood is not. The matcher therefore
self-selects onto the parts of the solve that were already right.

Beam-search fusion changes what is being asked of the prior, and an
n-gram is the only shape that can answer it. Prefix beam search needs a
score for EVERY candidate extension at EVERY frame; a library matcher only
fires on complete recognised windows, which is far too sparse to steer a
beam. More importantly the reach argument dissolves: the LM is consulted
while many competing prefixes are still alive, so it does not have to
recognise an algorithm inside a corrupted sequence — it has to make the
corruption lose in the first place.

The evidence for the n-gram was already in that document's §1, measured on
35 sessions and never acted on:

    4-move windows belonging to a 4-gram seen >= 3x     65.6%
    the same measurement with the moves SHUFFLED         0.0%

The 0.0% null is what makes this a real signal rather than a description
of the move alphabet. This module also avoids §6d's failure mode entirely
— library ambiguity across 4704 rotation/mirror variants — because it
never has to decide which orientation it is looking at.

Smoothing
---------
Witten-Bell interpolated backoff. With only ~3000 training moves a
5-gram is mostly unseen contexts, so the backoff weight has to depend on
how much evidence a context actually has:

    P(w|c) = lam(c) * P_ml(w|c) + (1 - lam(c)) * P(w|c[1:])
    lam(c) = count(c) / (count(c) + n_unique_continuations(c))

A context seen many times with few distinct continuations (`R U R' ...`)
keeps most of its mass; a context seen once backs off almost entirely.
Chosen over Kneser-Ney deliberately: KN's discount is estimated from
count-of-counts statistics that are themselves unreliable at this data
size.

CRITICAL: build only from sessions the acoustic model trained on. An LM
built over the held-out sessions would leak their move sequences into
their own decode, and the decode would look excellent for a reason that
has nothing to do with the model.

Usage:
    from move_lm import MoveLM
    lm = MoveLM.from_sessions(train_names, order=4)
    lm.logp(next_class, context_tuple)      # natural log
"""

import json
import math
from collections import defaultdict
from pathlib import Path

WCA12 = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]
V = len(WCA12)
BOS = -1          # sentence-start symbol, never a real class
DEFAULT_ORDER = 4
SESSION_ROOT = Path("../training_data")


class MoveLM:
    """Interpolated Witten-Bell n-gram over the 12 quarter-turn classes."""

    def __init__(self, order: int = DEFAULT_ORDER):
        self.order = order
        # counts[k] maps a k-length context tuple -> {class: count}
        self.counts = [defaultdict(lambda: defaultdict(int))
                       for _ in range(order)]
        self.n_sequences = 0
        self.n_moves = 0
        self._cache: dict = {}

    # -- construction --------------------------------------------------
    def add_sequence(self, classes: list[int]) -> None:
        self.n_sequences += 1
        self.n_moves += len(classes)
        padded = [BOS] * (self.order - 1) + list(classes)
        for i in range(self.order - 1, len(padded)):
            w = padded[i]
            for k in range(self.order):
                ctx = tuple(padded[i - k:i]) if k else ()
                self.counts[k][ctx][w] += 1

    @classmethod
    def from_sessions(cls, session_names: list[str], order: int = DEFAULT_ORDER,
                      root: Path = SESSION_ROOT) -> "MoveLM":
        lm = cls(order)
        for name in session_names:
            seq = load_truth(root / name)
            if seq:
                lm.add_sequence(seq)
        return lm

    # -- scoring -------------------------------------------------------
    def logp(self, w: int, context: tuple) -> float:
        """log P(w | context). `context` may be any length; only the last
        order-1 symbols matter, and it is padded with BOS when short."""
        ctx = tuple(context[-(self.order - 1):]) if self.order > 1 else ()
        if len(ctx) < self.order - 1:
            ctx = (BOS,) * (self.order - 1 - len(ctx)) + ctx
        key = (w, ctx)
        hit = self._cache.get(key)
        if hit is None:
            hit = math.log(max(self._p(w, ctx, len(ctx)), 1e-12))
            self._cache[key] = hit
        return hit

    def _p(self, w: int, ctx: tuple, k: int) -> float:
        if k == 0:
            table = self.counts[0][()]
            total = sum(table.values())
            # uniform floor so an unseen class is improbable, not impossible
            return (table.get(w, 0) + 1.0) / (total + V)
        table = self.counts[k].get(ctx)
        lower = self._p(w, ctx[1:], k - 1)
        if not table:
            return lower
        total = sum(table.values())
        uniq = len(table)
        lam = total / (total + uniq)          # Witten-Bell
        return lam * (table.get(w, 0) / total) + (1 - lam) * lower

    def perplexity(self, sequences: list[list[int]]) -> float:
        """Per-move perplexity. The model-selection metric for `order`:
        lower means the LM predicts held-out solves better, and a model
        that cannot beat uniform (12.0) is worth nothing to the beam."""
        total_lp, n = 0.0, 0
        for seq in sequences:
            ctx = ()
            for w in seq:
                total_lp += self.logp(w, ctx)
                ctx = ctx + (w,)
                n += 1
        return math.exp(-total_lp / max(n, 1))


def load_truth(session_dir: Path) -> list[int] | None:
    """BLE ground-truth move classes for one session, or None."""
    path = session_dir / "moves.jsonl"
    if not path.exists():
        return None
    out = []
    for line in open(path):
        if not line.strip():
            continue
        w = json.loads(line).get("wca_notation")
        if w not in WCA12:
            return None            # a non-quarter-turn makes the row unusable
        out.append(WCA12.index(w))
    return out or None


def _main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="checkpoints/move_ctc_s0.pt",
                   help="checkpoint whose train/val split defines the LM's "
                        "training set (never build over the holdout)")
    p.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--dev", type=int, default=8,
                   help="training sessions held out of the LM to pick order")
    args = p.parse_args()

    import torch
    ck = torch.load(args.model, map_location="cpu")
    train = list(ck["train_session_names"])
    held = list(ck["val_session_names"])

    dev = train[-args.dev:]
    fit = train[:-args.dev]
    dev_seqs = [s for s in (load_truth(SESSION_ROOT / n) for n in dev) if s]
    held_seqs = [s for s in (load_truth(SESSION_ROOT / n) for n in held) if s]

    print(f"\n  LM fit on {len(fit)} sessions, order chosen on {len(dev_seqs)} "
          f"dev sessions ({sum(len(s) for s in dev_seqs)} moves).")
    print(f"  Held-out perplexity is reported for information only — it must "
          f"NOT pick the order.\n")
    print(f"  {'order':<7} {'dev ppl':<10} {'holdout ppl':<13} {'contexts'}")
    print(f"  {'-'*46}")
    best = (None, float('inf'))
    for o in args.orders:
        lm = MoveLM.from_sessions(fit, order=o)
        d = lm.perplexity(dev_seqs)
        h = lm.perplexity(held_seqs)
        n_ctx = len(lm.counts[o - 1]) if o >= 1 else 0
        star = ""
        if d < best[1]:
            best = (o, d)
            star = "  <- best on dev"
        print(f"  {o:<7} {d:<10.3f} {h:<13.3f} {n_ctx}{star}")
    print(f"\n  uniform baseline perplexity = {float(V):.1f}")
    print(f"  chosen order = {best[0]} (by dev perplexity)")


if __name__ == "__main__":
    _main()
