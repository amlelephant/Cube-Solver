"""
make_session_list.py

Writes the explicit --sessions list for a retrain, with the permanent
held-out set excluded by NAME.

Why a script and not a glob. `../training_data/solve_*/` would sweep in the
07-29/30/31 takes, and those are the only sessions no checkpoint has ever
trained on — they are the cross-day/evening evaluation set that every
comparison in ALGORITHM_PRIOR, GAMEPLAN and the algo*_s*.json sweeps is
denominated in. Training on them would silently convert every "held out"
number in this repo into a memorisation number, which is the exact failure
[[named-holdouts-cross-env]] records. Excluding them has to be explicit and
reviewable, so it lives here rather than in a shell glob someone edits.

The 4 validation sessions are NOT excluded here — they are passed to
train_ctc.py via --val-session-names, which holds them out of training
while still using them for early stopping. Keeping the same 4 as
move_ctc_aug_s0/s1.pt is what makes the retrain a controlled comparison:
same augmentation, same val split, only the training data differs.

Usage:
    python make_session_list.py            # writes train_sessions.txt
    python make_session_list.py --print
"""

import argparse
from pathlib import Path

# Never trained on, by name. The 07-29/30/31 takes: 6 daytime + 2 evening,
# cross-day relative to every training session.
HELD_OUT = {
    "solve_20260729_221809_scramble", "solve_20260729_221809_solve",
    "solve_20260730_111941_scramble", "solve_20260730_111941_solve",
    "solve_20260730_113054_scramble", "solve_20260730_113054_solve",
    "solve_20260731_211018_solve", "solve_20260731_213559_solve",
}

# Held out of TRAINING but used for early stopping. Same four as
# move_ctc_s0/s1 and move_ctc_aug_s0/s1 — do not change them without
# retraining the baselines too, or the comparison stops being controlled.
VAL_NAMES = [
    "solve_20260721_102711", "solve_20260722_101225",
    "solve_20260723_105530_solve", "solve_20260724_100120_solve",
]


def session_list(root: Path) -> list[Path]:
    return sorted(d for d in root.glob("solve_*")
                  if (d / "detector_stream_color.npz").exists()
                  and d.name not in HELD_OUT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="../training_data")
    p.add_argument("--out", default="train_sessions.txt")
    p.add_argument("--print", action="store_true")
    args = p.parse_args()

    dirs = session_list(Path(args.root))
    text = "\n".join(d.as_posix() for d in dirs)
    if args.print:
        print(text)
    else:
        # newline="" so Python does not translate to CRLF. The file is
        # consumed as `--sessions $(cat train_sessions.txt | tr "\n" " ")`,
        # and a trailing \r rides along into each argument, where
        # Path("...\r").is_dir() is False and train_ctc.py reports the
        # thoroughly unhelpful "No session directories found".
        with open(args.out, "w", newline="") as fh:
            fh.write(text + "\n")
        print(f"  wrote {args.out}: {len(dirs)} sessions "
              f"({len(dirs) - len(VAL_NAMES)} train + {len(VAL_NAMES)} val)")
        print(f"  held out entirely: {len(HELD_OUT)}")
        for n in sorted(d.name for d in dirs if "20260802" in d.name):
            print(f"    new: {n}")


if __name__ == "__main__":
    main()
