"""
f1_data_and_model.py — figures showing what the model reads and what it emits.

  fig_input.pdf         the four input channels on real held-out frames,
                        with the BLE onset that supervises them
  fig_posteriorgram.pdf the 13-way per-frame posteriorgram over a real
                        held-out span, with truth onsets and CTC emissions
  fig_ctc_failure.pdf   the same, on a span the model gets wrong
"""

from __future__ import annotations

import numpy as np

import common as C
import figstyle as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

GOOD = "solve_20260730_113054_solve"    # daytime, 97.0% held out
HARD = "solve_20260803_095533_solve"    # daytime, 81.4% held out — miss-heavy
TAG = "move_ctc_spd_s0"


def load(session):
    d = C.SESSIONS / session
    z = np.load(d / "detector_stream_color.npz", allow_pickle=True)
    post = np.load(C.DATA / "post" / f"{TAG}__{session}.npz")["class_prob"]
    return z, post


def fig_input():
    z, _ = load(GOOD)
    frames = z["frames"]          # (N, H, W, 3) BGR uint8
    onsets = z["onset_idx"].astype(int)
    fps = float(z["fps"])
    from reconstruct import WCA12
    cls = z["onset_class"].astype(int)

    # a window centred on a mid-solve onset
    k = len(onsets) // 2
    c = onsets[k]
    idx = list(range(c - 4, c + 5, 2))

    fig, axes = plt.subplots(2, len(idx), figsize=(6.9, 2.35),
                             gridspec_kw=dict(hspace=0.12, wspace=0.06))
    lum = (frames[..., 2] * 0.299 + frames[..., 1] * 0.587
           + frames[..., 0] * 0.114) / 255.0
    dmap = LinearSegmentedColormap.from_list(
        "d", ["#1c3f6e", "#f4f3ee", "#8c3410"])
    for j, i in enumerate(idx):
        axes[0, j].imshow(frames[i][..., ::-1])
        d = lum[i] - lum[i - 1]
        axes[1, j].imshow(d, cmap=dmap, vmin=-0.25, vmax=0.25)
        for r in range(2):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            for s in axes[r, j].spines.values():
                s.set_visible(True)
                s.set_color(F.GRID)
        off = (i - c) / fps * 1000
        axes[0, j].set_title(f"{off:+.0f} ms",
                             fontsize=7, color=F.INK2 if i != c else F.ORANGE,
                             fontweight="bold" if i == c else "normal",
                             pad=3, loc="center")
    axes[0, 0].set_ylabel("RGB\n(3 ch)", fontsize=7.5, color=F.INK2)
    axes[1, 0].set_ylabel("diff-luma\n(1 ch)", fontsize=7.5, color=F.INK2)
    fig.suptitle(
        f"Input to the encoder: 96$\\times$96 cube-cropped RGB plus a signed "
        f"luma difference.\nCentre column is the BLE-timestamped onset of "
        f"move “{WCA12[cls[k]]}” in {GOOD[6:].replace('_',' ')}.",
        fontsize=8, color=F.INK, y=1.14, x=0.02, ha="left")
    F.save(fig, "fig_input.pdf")


def _posteriorgram_panel(ax, post, lo, hi, onsets, cls, title):
    from reconstruct import WCA12
    labels = WCA12 + ["blank"]
    sub = post[lo:hi].T                      # (13, T)
    cmap = LinearSegmentedColormap.from_list(
        "seq", ["#ffffff", "#cfe0f5", "#7fb0e8", "#2a78d6", "#123f78"])
    ax.imshow(sub, aspect="auto", cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest",
              extent=[lo, hi, 12.5, -0.5])
    ax.set_yticks(range(13))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("frame index (30 fps)")
    ax.set_title(title, fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    # truth onsets
    for o, k in zip(onsets, cls):
        if lo <= o < hi:
            ax.plot([o], [k], marker="v", color=F.ORANGE, markersize=5,
                    markeredgewidth=0, zorder=5, clip_on=False)
    ax.axhline(11.5, color=F.GRID, lw=0.8)


def fig_posteriorgram():
    from ctc_decode import prefix_beam_decode
    for session, name, span in ((GOOD, "fig_posteriorgram.pdf", 3.2),
                                (HARD, "fig_ctc_failure.pdf", 3.2)):
        z, post = load(session)
        onsets = z["onset_idx"].astype(int)
        cls = z["onset_class"].astype(int)
        fps = float(z["fps"])
        lab, fr = prefix_beam_decode(np.log(np.maximum(post, 1e-12)), beam=16)

        # densest span of this length — a pause shows nothing interesting
        w = int(span * fps)
        best, lo = -1, 0
        for s in range(0, len(post) - w, 5):
            n = int(((onsets >= s) & (onsets < s + w)).sum())
            if n > best:
                best, lo = n, s
        hi = lo + w

        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(6.9, 3.5), height_ratios=[4, 1.05],
            gridspec_kw=dict(hspace=0.55))
        acc = {r["session"]: r["acc"] for r in C.load("m1_recognition.json")
               if r["model"] == TAG}
        _posteriorgram_panel(
            ax, post, lo, hi, onsets, cls,
            f"{session[6:].replace('_', ' ')} — held out, "
            f"{acc[session]*100:.1f}% per-move accuracy")
        ax.text(1.0, 1.20, "▼ BLE ground-truth onset", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=F.ORANGE)

        # emission lane
        from reconstruct import WCA12
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        for s in ax2.spines.values():
            s.set_visible(False)
        ax2.set_xticks([])
        pmax = 1 - post[:, 12]
        ax2.fill_between(range(lo, hi), 0, pmax[lo:hi], color="#dfe9f7",
                         lw=0, zorder=1)
        ax2.plot(range(lo, hi), pmax[lo:hi], color=F.BLUE, lw=1.4, zorder=2)
        ax2.set_ylim(0, 1.5)
        prev = -99
        row = 0
        for c, f_ in zip(lab, fr):
            if lo <= f_ < hi:
                row = (row + 1) % 2 if f_ - prev < (hi - lo) * 0.045 else 0
                prev = f_
                ax2.axvline(f_, color=F.VIOLET, lw=1.0, ymax=0.62, zorder=3)
                ax2.text(f_, 1.0 + 0.24 * row, WCA12[c], fontsize=6.5,
                         ha="center", color=F.VIOLET, zorder=4)
        ax2.set_xlabel("CTC beam emissions (violet ticks) over "
                       "$1-P(\\mathrm{blank})$ (blue)",
                       fontsize=7.5, labelpad=3)
        F.save(fig, name)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(C.MD))
    fig_input()
    fig_posteriorgram()
