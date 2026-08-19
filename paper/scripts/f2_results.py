"""
f2_results.py — result figures that depend only on m1/m3/m4/m5.

  fig_ladder.pdf      holdout proximity ladder
  fig_ablation.pdf    the five-rung checkpoint ladder, one holdout
  fig_persession.pdf  per-session accuracy and error channels
  fig_confusion.pdf   substitution confusion matrix
  fig_anticheat.pdf   observed move count, legit solves vs proxy attacks
  fig_errpos.pdf      where errors fall within a solve
"""

from __future__ import annotations

import numpy as np

import common as C
import figstyle as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#ffffff", "#cfe0f5", "#7fb0e8", "#2a78d6", "#123f78"])


def fig_ladder():
    rows = C.load("m5_ladder.json")
    labels = ["trained on\n(memorisation)", "validation\n(same-day)",
              "held out,\nbracketed days", "held out,\nlater days"]
    vals = [r["mean"] * 100 for r in rows]
    ns = [r["n"] for r in rows]
    s0 = [r["seed0"] * 100 for r in rows]
    s1 = [r["seed1"] * 100 for r in rows]
    colors = [F.MUTED, F.YELLOW, F.BLUE, F.BLUE]

    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    F.grid(ax)
    x = np.arange(4)
    ax.bar(x, vals, width=0.62, color=colors, zorder=3)
    for i in range(4):
        ax.plot([x[i], x[i]], [s0[i], s1[i]], color=F.INK, lw=1.1, zorder=4,
                solid_capstyle="butt")
        ax.text(x[i], vals[i] + 1.1, f"{vals[i]:.1f}", ha="center",
                fontsize=7.5, color=F.INK, fontweight="bold")
        ax.text(x[i], 82.0, f"n={ns[i]}", ha="center", fontsize=6.8,
                color="#ffffff")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(80, 102)
    ax.set_ylabel("per-move accuracy (%)")
    ax.set_title("Accuracy falls with distance from training")
    ax.text(0.0, -0.31, "black bar spans the two training seeds",
            transform=ax.transAxes, fontsize=6.8, color=F.MUTED)
    F.save(fig, "fig_ladder.pdf")


def fig_ablation():
    rows = C.load("m4_summary.json")
    short = ["peak-pick\n(joint head)", "CTC", "+ photometric\naug",
             "+ 6 faster\nsessions", "+ speed\naug"]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    F.grid(ax)
    for k, (key, col, lab) in enumerate((("daytime", F.BLUE, "daytime (9 takes)"),
                                         ("evening", F.ORANGE, "evening (5 takes)"))):
        v = [r[key] * 100 for r in rows]
        e = [r[key + "_spread"] * 100 / 2 for r in rows]
        ax.bar(x + (k - 0.5) * w, v, width=w - 0.04, color=col, zorder=3,
               label=lab)
        ax.errorbar(x + (k - 0.5) * w, v, yerr=e, fmt="none", ecolor=F.INK,
                    elinewidth=1.0, capsize=2.2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7)
    ax.set_ylim(55, 97)
    ax.set_ylabel("per-move accuracy (%)")
    ax.set_title("Every training change, scored on one never-seen holdout")
    ax.legend(loc="upper left", ncols=2, fontsize=7)
    ax.text(0.0, -0.30, "bars are the two-seed mean; whiskers half the "
            "seed spread. 14 held-out solves throughout.",
            transform=ax.transAxes, fontsize=6.8, color=F.MUTED)
    F.save(fig, "fig_ablation.pdf")


def fig_persession():
    rows = [r for r in C.load("m1_recognition.json")
            if r["session"].endswith("_solve")]
    names = sorted({r["session"] for r in rows})
    # order by mean accuracy
    acc = {n: np.mean([r["acc"] for r in rows if r["session"] == n])
           for n in names}
    names.sort(key=lambda n: -acc[n])
    meta = {m["session"]: m for m in C.load("holdout_meta.json")}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.0),
                                  gridspec_kw=dict(wspace=0.30))
    y = np.arange(len(names))[::-1]
    F.grid(ax, axis="x")
    for i, n in enumerate(names):
        rs = [r for r in rows if r["session"] == n]
        col = F.ORANGE if meta[n]["evening"] else F.BLUE
        ax.barh(y[i], acc[n] * 100, height=0.62, color=col, zorder=3)
        lo = min(r["acc"] for r in rs) * 100
        hi = max(r["acc"] for r in rs) * 100
        ax.plot([lo, hi], [y[i], y[i]], color=F.INK, lw=1.1, zorder=4)
        ax.text(max(hi, acc[n] * 100) + 1.0, y[i], f"{acc[n]*100:.1f}",
                va="center", fontsize=6.6, color=F.INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{n[10:12]}-{n[12:14]}  {n[15:17]}:{n[17:19]}" for n in names],
        fontsize=6.6)
    ax.set_xlim(60, 106)
    ax.set_xlabel("per-move accuracy (%)")
    ax.set_title("Per-session, both seeds")
    ax.text(0.0, -0.16, "blue = daytime, orange = evening",
            transform=ax.transAxes, fontsize=6.8, color=F.MUTED)

    # error channels, pooled by regime
    F.grid(ax2, axis="x")
    groups = [("daytime", False), ("evening", True)]
    chans = [("miss", F.BLUE), ("sub", F.ORANGE), ("phantom", F.AQUA)]
    yy = np.arange(len(groups))[::-1]
    for gi, (lab, ev) in enumerate(groups):
        g = [r for r in rows if bool(meta[r["session"]]["evening"]) == ev]
        n_gt = sum(r["n_gt"] for r in g)
        left = 0.0
        for ci, (ch, col) in enumerate(chans):
            v = sum(r[ch] for r in g) / n_gt * 100
            ax2.barh(yy[gi], v, left=left, height=0.5, color=col, zorder=3)
            if v > 1.4:
                ax2.text(left + v / 2, yy[gi], f"{v:.1f}", ha="center",
                         va="center", fontsize=6.8, color="white",
                         fontweight="bold")
            left += v + 0.16
    ax2.set_yticks(yy)
    ax2.set_yticklabels([g[0] for g in groups], fontsize=7.5)
    ax2.set_xlabel("errors per 100 ground-truth moves")
    ax2.set_title("Error mass by channel")
    ax2.set_ylim(-0.6, 1.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in chans]
    ax2.legend(handles, ["miss (onset never emitted)",
                         "substitution (named wrong)",
                         "phantom (spurious emission)"],
               loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=6.8,
               ncols=1)
    F.save(fig, "fig_persession.pdf")


def fig_confusion():
    d = C.load("m5_confusion.json")
    M = np.array(d["matrix"], dtype=float)
    labels = d["labels"]
    fig, ax = plt.subplots(figsize=(3.35, 3.05))
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=max(M.max(), 1))
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(f"Substitutions ({d['n_sub']} held out, 2 seeds)")
    for i in range(12):
        for j in range(12):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        fontsize=6.2,
                        color="white" if M[i, j] > M.max() * 0.55 else F.INK)
    for s in ax.spines.values():
        s.set_visible(False)
    k = d["kinds"]
    tot = sum(k.values())
    ax.text(0.0, -0.24, f"same face, wrong direction {k['inverse']/tot*100:.0f}%"
            f"   ·   adjacent face {k['adjacent']/tot*100:.0f}%"
            f"   ·   opposite face {k['opposite']/tot*100:.0f}%",
            transform=ax.transAxes, fontsize=6.8, color=F.MUTED)
    F.save(fig, "fig_confusion.pdf")


def fig_anticheat():
    import json
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    F.grid(ax)
    paths = [C.MD / "checkpoints" / c
             for c in ("move_ctc_spd_s0.pt", "move_ctc_spd_s1.pt")]
    seen = C.ckpt_seen(paths)
    rows = json.loads((C.DATA / "anticheat_gate_move_ctc_spd_s0.json").read_text())
    held = [r for r in rows if r["session"] not in seen]
    legit = sorted(r["observed_moves"] for r in held if r["role"] == "solve")
    proxy = sorted(r["observed_moves"] for r in held
                   if r["role"] == "scramble_proxy")
    ax.scatter(proxy, np.full(len(proxy), 1) + np.random.default_rng(0)
               .normal(0, 0.045, len(proxy)), s=26, color=F.ORANGE, zorder=3,
               label=f"scramble-as-solver-following proxy (n={len(proxy)})")
    ax.scatter(legit, np.full(len(legit), 2) + np.random.default_rng(1)
               .normal(0, 0.045, len(legit)), s=26, color=F.BLUE, zorder=3,
               label=f"genuine solves (n={len(legit)})")
    ax.axvline(32, color=F.INK, lw=1.4, ls=(0, (4, 2)), zorder=4)
    ax.text(32.8, 2.42, "floor = 32", fontsize=7, color=F.INK)
    ax.axvspan(26, 32, color="#f2f1ec", zorder=1)
    ax.text(24.0, 0.60, "God's number (QTM) = 26", fontsize=6.6,
            color=F.MUTED, va="bottom", ha="left")
    ax.axvline(26, color=F.MUTED, lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["proxy attack", "genuine"], fontsize=7.5)
    ax.set_ylim(0.5, 2.9)
    ax.set_xlim(0, 200)
    ax.set_xlabel("moves the model decoded over the timed window")
    ax.set_title("The count gate separates by a margin, not at a boundary")
    ax.legend(loc="upper right", fontsize=6.8)
    ax.text(0.0, -0.34, f"held-out sessions only, seed 0. "
            f"Lowest genuine {min(legit)}; highest proxy {max(proxy)}; "
            f"gap {min(legit)-max(proxy)} moves.",
            transform=ax.transAxes, fontsize=6.8, color=F.MUTED)
    F.save(fig, "fig_anticheat.pdf")


def fig_errpos():
    d = C.load("m5_positions.json")
    pos = d["positions"]
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    F.grid(ax)
    nb = 5
    bins = np.linspace(0, 1, nb + 1)
    w = 0.26 / nb * nb / 1.0
    centres = bins[:-1] + 0.5 / nb
    series = (("miss", F.BLUE, "miss"), ("sub", F.ORANGE, "substitution"),
              ("phantom", F.AQUA, "phantom"))
    for j, (k, col, lab) in enumerate(series):
        v = np.array(pos[k])
        h, _ = np.histogram(v, bins=bins)
        ax.bar(centres + (j - 1) * 0.058, h / max(len(v), 1) * 100,
               width=0.054, color=col, zorder=3,
               label=f"{lab} ($n$={len(v)})")
    ax.axhline(20, color=F.MUTED, lw=1.0, ls=(0, (3, 3)), zorder=2)
    ax.text(0.985, 20.6, "uniform", ha="right", fontsize=6.4, color=F.MUTED)
    ax.set_xticks(centres)
    ax.set_xticklabels(["first\nfifth", "", "middle", "", "last\nfifth"],
                       fontsize=6.8)
    ax.set_xlabel("position within the solve", labelpad=1)
    ax.set_ylabel("% of that channel's errors")
    ax.set_title("Where the errors fall")
    ax.legend(loc="upper right", fontsize=6.6)
    F.save(fig, "fig_errpos.pdf")


if __name__ == "__main__":
    fig_ladder()
    fig_ablation()
    fig_persession()
    fig_confusion()
    fig_anticheat()
    fig_errpos()
