"""
f3_decode.py — figures for the group decode and the falsifiability sweep.
Run after m2_decode has produced m2_decode_s0.json / m2_decode_s1.json.

  fig_decode.pdf   raw vs post-decode per session, and the cost the true
                   story pays against the beam's measured envelope
  fig_cliff.pdf    that cost against raw accuracy — why the decode is
                   all-or-nothing
  fig_drift.pdf    corpus speed drift, training vs held-out
"""

from __future__ import annotations

import json

import numpy as np

import common as C
import figstyle as F
import matplotlib.pyplot as plt


def load_m2():
    rows = []
    for s in ("s0", "s1"):
        p = C.DATA / f"m2_decode_{s}.json"
        if p.exists():
            rows += json.loads(p.read_text())
    if not rows:
        p = C.DATA / "m2_decode.json"
        if p.exists():
            rows = json.loads(p.read_text())
    return rows


def fig_decode(rows):
    """Left: what the decode does to accuracy. Right: why it does so little
    — the cost the TRUE story pays, against the beam's measured envelope.

    The falsifiability sweep is deliberately NOT plotted as a bar chart of
    acceptance rates: on this holdout every claim is rejected, including
    every true one, so a chart of zeros would read as a security result
    when it is really a statement about the recogniser's error rate."""
    meta = {m["session"]: m for m in C.load("holdout_meta.json")}
    sess = sorted({r["session"] for r in rows},
                  key=lambda s: np.mean([r["raw_acc"] for r in rows
                                         if r["session"] == s]))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.1),
                                  gridspec_kw=dict(wspace=0.34))
    # --- panel 1: raw -> post ------------------------------------------
    F.grid(ax, axis="x")
    y = np.arange(len(sess))[::-1]
    for i, s in enumerate(sess):
        g = [r for r in rows if r["session"] == s]
        raw = np.mean([r["raw_acc"] for r in g]) * 100
        post = np.mean([r["post_acc"] for r in g]) * 100
        col = F.AQUA if post > raw + 0.05 else (F.RED if post < raw - 0.05
                                                else F.MUTED)
        ax.plot([raw, post], [y[i], y[i]], color=col, lw=2.2, zorder=3,
                solid_capstyle="round")
        ax.plot([raw], [y[i]], "o", color=F.MUTED, ms=3.6, zorder=4)
        ax.plot([post], [y[i]], "o", color=col, ms=4.6, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{s[10:12]}-{s[12:14]} {s[15:17]}:{s[17:19]}"
         + ("*" if [r for r in rows if r["session"] == s][0]["slice_session"]
            else "")
         for s in sess], fontsize=6.4)
    ax.set_xlabel("per-move accuracy (%)")
    ax.set_title("Raw $\\rightarrow$ post-decode")
    ax.text(0.0, -0.17, "grey = raw, coloured = post-decode; * = slice",
            transform=ax.transAxes, fontsize=6.6, color=F.MUTED)

    # --- panel 2: the cost of the true story vs the beam envelope --------
    F.grid(ax2, axis="x")
    costs = [np.mean([r["gt_path_cost"] for r in rows if r["session"] == s])
             for s in sess]
    for i, (s, c) in enumerate(zip(sess, costs)):
        ev = meta[s]["evening"]
        ax2.barh(y[i], c, height=0.62,
                 color=F.ORANGE if ev else F.BLUE, zorder=3)
    ax2.axvline(10, color=F.INK, lw=1.3, ls=(0, (4, 2)), zorder=5)
    ax2.text(11.5, y[-1] - 0.15, "beam envelope $\\approx$ 10",
             fontsize=6.6, color=F.INK, va="center")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("cost of the TRUE story under the decoder's model")
    ax2.set_title("Why nothing verifies")
    ax2.text(0.0, -0.17, "blue = daytime, orange = evening; every session "
             "is far outside the envelope", transform=ax2.transAxes,
             fontsize=6.6, color=F.MUTED)
    F.save(fig, "fig_decode.pdf")


def fig_cliff(rows):
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    F.grid(ax)
    ver = [r for r in rows if r["verified"]]
    non = [r for r in rows if not r["verified"]]
    for g, col, lab, mk in ((non, F.MUTED, "did not verify", "o"),
                            (ver, F.AQUA, "verified", "D")):
        if not g:
            continue
        ax.scatter([r["raw_acc"] * 100 for r in g],
                   [r["gt_path_cost"] for r in g],
                   s=26, color=col, marker=mk, zorder=3, label=lab)
    ax.set_xlabel("raw per-move accuracy (%)")
    ax.set_ylabel("cost of the true story")
    ax.set_title("Cost, not error count, is what binds")
    ax.axhline(10.0, color=F.INK, lw=1.1, ls=(0, (4, 2)), zorder=2)
    ax.text(ax.get_xlim()[0] + 1, 11, "measured beam envelope $\\approx$ 10",
            fontsize=6.6, color=F.INK2)
    ax.legend(loc="upper right", fontsize=6.8)
    F.save(fig, "fig_cliff.pdf")


def fig_drift():
    import datetime as dt
    import torch
    paths = [C.MD / "checkpoints" / c
             for c in ("move_ctc_spd_s0.pt", "move_ctc_spd_s1.pt")]
    seen = C.ckpt_seen(paths)
    pts = []
    for npz in sorted(C.SESSIONS.glob("solve_*/detector_stream_color.npz")):
        d = npz.parent
        if d.name.endswith("_scramble"):
            continue
        m = C.session_meta(d)
        if not m["tps"] or m["n_moves"] < 40:
            continue
        s = d.name.split("_")[1]
        day = dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        pts.append((day, m["tps"], m["crowded_frac"], d.name in seen))
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    F.grid(ax)
    d0 = min(p[0] for p in pts)
    for tr, col, lab, mk in ((True, F.MUTED, "in training", "o"),
                             (False, F.BLUE, "held out", "D")):
        g = [p for p in pts if p[3] == tr]
        ax.scatter([(p[0] - d0).days for p in g], [p[1] for p in g],
                   s=28, color=col, marker=mk, zorder=3, label=lab)
    xs = np.array([(p[0] - d0).days for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts])
    k, b = np.polyfit(xs, ys, 1)
    ax.plot([xs.min(), xs.max()], [k * xs.min() + b, k * xs.max() + b],
            color=F.ORANGE, lw=1.6, zorder=2)
    ax.set_xlabel(f"days since first recording ({d0.isoformat()})")
    ax.set_ylabel("turns per second")
    ax.set_title("The solver got faster than the corpus")
    ax.legend(loc="lower right", fontsize=6.8)
    ax.text(0.0, -0.34, f"trend {k:+.3f} turns/s per day; the held-out "
            f"sessions sit at the fast end, so day-distance and speed are "
            f"confounded.", transform=ax.transAxes, fontsize=6.6,
            color=F.MUTED)
    F.save(fig, "fig_drift.pdf")


def placeholder(name, title):
    """Keep the document compilable while the slow sweep is still running."""
    fig, ax = plt.subplots(figsize=(6.9, 1.6))
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=10,
            color=F.INK, fontweight="bold")
    ax.text(0.5, 0.22, "measurement still running — regenerate with "
            "f3_decode.py once m2_decode_s{0,1}.json exist",
            ha="center", va="center", fontsize=8, color=F.MUTED)
    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, fill=False,
                               edgecolor=F.GRID, lw=1.0,
                               transform=ax.transAxes))
    F.save(fig, name)


if __name__ == "__main__":
    fig_drift()
    rows = load_m2()
    if rows and len({r["session"] for r in rows}) >= 3:
        fig_decode(rows)
        fig_cliff(rows)
    else:
        print("  m2 incomplete — writing placeholders")
        placeholder("fig_decode.pdf", "Group decode + falsifiability sweep")
        placeholder("fig_cliff.pdf", "Decode cost envelope")
