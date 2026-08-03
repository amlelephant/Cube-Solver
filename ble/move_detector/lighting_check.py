"""
lighting_check.py

Tell the user their lighting is unlike anything the model trained on —
BEFORE they solve, not after.

Why this exists
---------------
Measured 2026-07-31, CTC arm, on takes the checkpoint had never seen:

    11:20 morning   92.5% end-to-end
    11:31 morning   90.2%
    22:19 evening   69.1%
    21:11 evening   56.0%
    21:37 evening   76.9%   <- same room and cube as 21:11, plus a lamp

That last pair is the whole argument. Same evening, same sitting, same
cube, one added lamp: 56.0% -> 76.9%. Lighting is causal and the remedy is
free, but only if the user finds out before spending a solve on it. The
training corpus is 54 of 62 sessions between 09:00 and 18:00, so evening
is roughly 6% of what the model has ever seen.

This module is a DESCRIPTION, not a gate — measured, not assumed
----------------------------------------------------------------
The obvious design was a preflight that flags a take as out-of-distribution
before recording. It was built and it does not work. Both forms were tried
against the four takes whose end-to-end accuracy is known:

  * per-statistic z-score against the corpus's session-to-session spread.
    The 56.0% take scores |z| <= 1.4 on every statistic and is reported
    "within range". Its diff_floor of 1.30 sits inside a corpus range of
    0.00..6.11 — evening-like statistics are not outside the corpus, they
    are in a sparse part of it.
  * local density: mean distance to the 5 nearest corpus sessions in the
    z-normalised statistic space. This is ANTI-correlated with accuracy —
    the 56.0% take lands at the 54th percentile of corpus looseness (i.e.
    perfectly typical) while the 76.9% take lands at the 79th:

        56.0%  5-NN 1.47   looser than 54% of corpus
        76.9%  5-NN 1.87   looser than 79% of corpus
        90.2%  5-NN 1.46   looser than 52% of corpus
        92.5%  5-NN 1.40   looser than 46% of corpus

So these six statistics do not carry whatever separates a 56% take from a
90% one, and no verdict is printed here. A gate that says "nothing to do"
on a take that then scores 56% is worse than no gate at all, because it
converts an unknown into a false assurance.

The causal channel was never isolated either. The statistic that most
obviously differs between morning and evening — the diff-luma baseline,
5.5-5.8 by day against 1.18-1.30 at night — was tested by restoring it at
inference and came out flat (56.0/57.1/56.0/54.8% at noise sigma
0/4/8/12).

What IS predictive, on the evidence available, is the clock: four daytime
takes at 90.2-92.5% against three evening takes at 56.0-76.9%. That is
what `time_of_day_note` reports, and it is a heuristic backed by seven
takes, not a model.

Use this module to CHARACTERISE sessions — particularly when deliberately
recording new lighting, to confirm the new sessions actually differ from
the corpus rather than just feeling different.

Usage:
    python lighting_check.py --build          # from the prepared corpus
    python lighting_check.py --check ../training_data/solve_20260731_211018_solve/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REFERENCE_PATH = Path(__file__).parent / "results" / "2026-07-31" / "lighting_reference.json"

# |z| beyond this against the corpus's session-to-session spread is called
# out. 2.5 is chosen so the four morning takes in the corpus stay silent
# and the 21:11 take does not — deliberately loose, because a preflight
# that cries wolf gets ignored and this one has no power to be right.
Z_WARN = 2.5

# The hours the corpus actually covers: 54 of 62 sessions fall inside them.
DAY_START, DAY_END = 9, 18

HINTS = {
    "luma_mean": ("dimmer than", "brighter than", "add or remove light"),
    "luma_std": ("flatter than", "harsher than",
                 "diffuse the light, or add a second source"),
    "rb": ("cooler/bluer than", "warmer/oranger than",
           "incandescent and LED read warm; daylight reads neutral"),
    "gb": ("less green than", "greener than", "fluorescent light does this"),
    "saturation": ("washed out vs", "more saturated than",
                   "low-CRI bulbs flatten sticker colours"),
    "diff_floor": ("stiller between frames than", "noisier between frames than",
                   "a low value means long exposure or in-camera denoising "
                   "— more light shortens the exposure"),
}


def frame_stats(frames: np.ndarray, stride: int = 5) -> dict:
    """
    Statistics of one already-cropped BGR stream, (N, H, W, 3) uint8.

    Deliberately cheap and deliberately several. `diff_floor` is the
    10th percentile of the per-frame mean absolute luma difference — the
    QUIET frames, when the cube is still between turns — which is what
    exposes long exposure and in-camera temporal denoising. Taking a
    percentile rather than a mean keeps it from being dominated by the
    turns themselves.
    """
    f = frames[::stride].astype(np.float32)
    luma = f[..., 2] * 0.299 + f[..., 1] * 0.587 + f[..., 0] * 0.114
    mx = f.max(axis=-1)
    sat = np.where(mx > 1e-6, (mx - f.min(axis=-1)) / np.maximum(mx, 1e-6), 0.0)

    # diff_floor uses CONSECUTIVE frames, so it is computed on the full
    # stream rather than the strided subsample — a difference across a
    # stride-5 gap is a different quantity entirely.
    fl = frames.astype(np.float32)
    lum_full = fl[..., 2] * 0.299 + fl[..., 1] * 0.587 + fl[..., 0] * 0.114
    d = np.abs(np.diff(lum_full, axis=0)).mean(axis=(1, 2))

    return {
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "rb": float(f[..., 2].mean() - f[..., 0].mean()),
        "gb": float(f[..., 1].mean() - f[..., 0].mean()),
        "saturation": float(sat.mean() * 255.0),
        "diff_floor": float(np.percentile(d, 10)),
    }


def build_reference(patterns: list[str], out_path: Path = REFERENCE_PATH
                    ) -> dict:
    """
    Aggregate per-session statistics over the prepared colour corpus.

    Reads detector_stream_color.npz, which is ALREADY cropped by the same
    code the live path uses — so the reference and the thing checked
    against it are the same measurement, and no cube detector has to run
    here at all.
    """
    dirs = sorted({Path(p) for pat in patterns
                   for p in (Path(".").glob(pat) if "*" in pat else [pat])})
    rows = []
    for d in dirs:
        npz = Path(d) / "detector_stream_color.npz"
        if not npz.exists():
            continue
        with np.load(npz) as z:
            s = frame_stats(z["frames"])
        s["session"] = Path(d).name
        rows.append(s)
        print(f"  {Path(d).name[-34:]:<36} "
              + "  ".join(f"{k} {s[k]:7.2f}" for k in
                          ("luma_mean", "rb", "diff_floor")))
    if not rows:
        sys.exit("No prepared colour sessions found. Run "
                 "`prepare_data.py --color` first.")

    keys = [k for k in rows[0] if k != "session"]
    ref = {"n_sessions": len(rows),
           "sessions": [r["session"] for r in rows],
           "stats": {k: {"mean": float(np.mean([r[k] for r in rows])),
                         "std": float(np.std([r[k] for r in rows])),
                         "min": float(np.min([r[k] for r in rows])),
                         "max": float(np.max([r[k] for r in rows]))}
                     for k in keys}}
    out_path.write_text(json.dumps(ref, indent=2))
    print(f"\n  Reference over {len(rows)} sessions -> {out_path}")
    return ref


def load_reference(path: Path = REFERENCE_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def compare(stats: dict, ref: dict) -> list[dict]:
    """Per-statistic z-score against the corpus's session-to-session spread."""
    out = []
    for k, v in stats.items():
        r = ref["stats"].get(k)
        if r is None:
            continue
        sd = max(r["std"], 1e-6)
        out.append({"name": k, "value": v, "ref_mean": r["mean"],
                    "ref_min": r["min"], "ref_max": r["max"],
                    "z": (v - r["mean"]) / sd})
    return sorted(out, key=lambda d: -abs(d["z"]))


def report(stats: dict, ref: dict, label: str = "this take") -> list[dict]:
    """
    Print how one session compares to the corpus, per statistic.

    Deliberately returns no verdict — see the module docstring for the two
    verdicts that were built, measured, and found to have no power. The
    `z` column is a position within the corpus, not a warning.
    """
    rows = compare(stats, ref)
    print(f"\n  Lighting vs the {ref['n_sessions']}-session training corpus "
          f"({label}):")
    print(f"    {'statistic':<12} {'this':>8} {'corpus':>9} "
          f"{'range':>17} {'z':>7}")
    for r in rows:
        print(f"    {r['name']:<12} {r['value']:>8.2f} {r['ref_mean']:>9.2f} "
              f"{r['ref_min']:>7.2f}..{r['ref_max']:<8.2f} {r['z']:>+7.1f}")
    far = [r for r in rows if abs(r["z"]) >= Z_WARN]
    for r in far:
        lo, hi, hint = HINTS[r["name"]]
        print(f"    note: {r['name']} is {lo if r['z'] < 0 else hi} the "
              f"corpus mean ({hint})")
    print(f"    These statistics do NOT predict accuracy — measured, see "
          f"the module docstring.")
    return rows


def time_of_day_note(ts: float | None = None) -> str | None:
    """
    The one predictor with demonstrated separation: the clock.

    Returns a warning string when the local hour falls outside the window
    the corpus actually covers, or None. Crude on purpose — it is a
    reminder carrying a measured number, not a detector, and unlike the
    statistical gates it cannot issue a false all-clear (an unusual room in
    daylight simply gets no message, which is what "no information" should
    look like).
    """
    import time as _time
    hour = _time.localtime(ts).tm_hour if ts else _time.localtime().tm_hour
    if DAY_START <= hour < DAY_END:
        return None
    return (
        f"  LIGHTING: it is {hour:02d}:xx. The training corpus is 54 of 62\n"
        f"  sessions between {DAY_START:02d}:00 and {DAY_END:02d}:00, so this "
        f"is a regime the model has\n"
        f"  barely seen. Measured 2026-07-31, same room and cube, CTC arm:\n"
        f"        21:11, no extra light   56.0% end to end\n"
        f"        21:37, one lamp added   76.9% end to end\n"
        f"        11:20 / 11:31 daylight  92.5% / 90.2%\n"
        f"  If you can add light, do it before recording — it was worth 21\n"
        f"  points on the one controlled comparison we have."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true",
                   help="Rebuild results/2026-07-31/lighting_reference.json from the corpus")
    p.add_argument("--sessions", nargs="+",
                   default=["../training_data/solve_*/"])
    p.add_argument("--check", nargs="+", default=None,
                   help="Check prepared session(s) against the reference")
    args = p.parse_args()

    if args.build:
        build_reference(args.sessions)
        return

    ref = load_reference()
    if ref is None:
        sys.exit(f"No {REFERENCE_PATH.name}. Build it with:\n"
                 f"    python lighting_check.py --build")
    if not args.check:
        sys.exit("Nothing to do: pass --build or --check <session>")

    for pat in args.check:
        for d in (Path(".").glob(pat) if "*" in pat else [Path(pat)]):
            npz = Path(d) / "detector_stream_color.npz"
            if not npz.exists():
                print(f"\n  {Path(d).name}: not prepared (--color) — skipping")
                continue
            with np.load(npz) as z:
                report(frame_stats(z["frames"]), ref, label=Path(d).name)


if __name__ == "__main__":
    main()
