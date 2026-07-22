"""
train.py

Trains the move-onset detector.

Pipeline:
    record_training.py  (with --keep-frames on postprocess)
      -> prepare_data.py      builds detector_stream.npz per session
      -> train.py             this file
      -> decode.py            peak-picking + windows for the classifier

Loss is plain BCE against the Gaussian onset targets from dataset.py. No
positive reweighting: at sigma=2 with a ~13-frame median move gap the
target already averages ~0.4.

Model selection is by ONSET F1 on held-out sessions, not by validation
loss. Loss is measured per frame and is dominated by the easy negatives
between moves; a model can improve its loss while getting worse at the
thing that matters, which is placing exactly one peak per turn. The
selection metric is therefore computed the way the detector is actually
used: score the whole session, peak-pick it, match against BLE truth.

The reported F1 uses a fixed threshold during training (cheap, comparable
across epochs). After training, the best checkpoint gets a threshold sweep
so the operating point is tuned on held-out data rather than guessed —
that tuned threshold is saved into the checkpoint.

Usage:
    python prepare_data.py --sessions ../training_data/solve_*/
    python train.py --sessions ../training_data/solve_*/
    python train.py --sessions ../training_data/solve_*/ --epochs 60 --batch 8
    python train.py --sessions ../training_data/solve_*/ --eval \\
        --model move_detector.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Progress output uses em-dashes, ± and a '<-' best-epoch marker. When stdout
# is a console Windows hands us UTF-8 and those survive, but redirect to a
# file (`> training_run_*.log`, the convention here) and it falls back to
# cp1252, where the first such character raises UnicodeEncodeError and kills
# the run mid-training. Reconfiguring costs nothing and keeps a logged run
# from dying on its own progress line.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
except ImportError:
    sys.exit("PyTorch not installed. Run: pip install torch torchvision")

from dataset import (SessionStream, OnsetClipDataset, load_streams,
                     split_streams, split_clips_pooled, overlap_frac,
                     CLIP_LEN, SIGMA)
from model import build_model, score_stream
from decode import (peak_pick, match_onsets, sweep_threshold, format_metrics,
                    MIN_SEP, THRESHOLD, TOLERANCE, BETA)

MODEL_PATH = "move_detector.pt"


def resolve_sessions(patterns: list[str]) -> list[Path]:
    dirs = [Path(p) for pattern in patterns
            for p in (Path(".").glob(pattern) if "*" in pattern
                      else [Path(pattern)])]
    return sorted(d for d in dirs if d.is_dir())


def evaluate_streams(model, streams: list[SessionStream], device,
                     threshold: float = THRESHOLD, min_sep: int = MIN_SEP,
                     tolerance: int = TOLERANCE
                     ) -> tuple[dict, list[tuple[str, dict]], dict]:
    """
    Score every session end to end, peak-pick, and match against BLE truth.

    Aggregates by summing tp/fp/fn across sessions and recomputing the
    rates from those totals — averaging per-session F1 would let a short
    session count as much as a long one.
    """
    per_session, scores_by_name = [], {}
    tp = fp = fn = 0
    errs = []

    for s in streams:
        scores = score_stream(model, s, device)
        scores_by_name[s.name] = scores
        pred = peak_pick(scores, threshold=threshold, min_sep=min_sep)
        m = match_onsets(pred, s.onset_idx, scores, tolerance)
        per_session.append((s.name, m))
        tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
        if not np.isnan(m["median_err"]):
            errs.append(m["median_err"])

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0

    agg = {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
           "recall": recall, "f1": f1,
           "median_err": float(np.median(errs)) if errs else float("nan"),
           "bias": float(np.mean([m["bias"] for _, m in per_session
                                  if not np.isnan(m["bias"])]))
           if per_session else float("nan")}
    return agg, per_session, scores_by_name


def clip_leakage(dataset: OnsetClipDataset, train_idx: list[int],
                 val_idx: list[int], clip_len: int) -> float:
    """
    Fraction of validation-clip frames that ALSO appear inside some
    training clip. 0.0 means the two sets are disjoint in pixels; 1.0 means
    every validation frame was trained on and the val metric is measuring
    recall of memorised data.
    """
    from collections import defaultdict
    covered = defaultdict(set)
    for i in train_idx:
        si, st = dataset.index[i]
        covered[si].update(range(st, st + clip_len))

    shared = total = 0
    for i in val_idx:
        si, st = dataset.index[i]
        frames = set(range(st, st + clip_len))
        total  += len(frames)
        shared += len(frames & covered[si])
    return shared / total if total else 0.0


def report_data(train_s, val_s, clip_len):
    print(f"\n{'='*66}")
    print(f"  Move Onset Detector — Training")
    print(f"{'='*66}")
    print(f"  Train: {len(train_s)} session(s), "
          f"{sum(len(s) for s in train_s)} frames, "
          f"{sum(len(s.onset_idx) for s in train_s)} onsets")
    print(f"  Val:   {len(val_s)} session(s) fully held out, "
          f"{sum(len(s) for s in val_s)} frames, "
          f"{sum(len(s.onset_idx) for s in val_s)} onsets")
    for s in val_s:
        print(f"           {s.name}  ({len(s)} frames, "
              f"{len(s.onset_idx)} onsets, {s.fps:.1f}fps)")


def train(args):
    session_dirs = resolve_sessions(args.sessions)
    if not session_dirs:
        sys.exit("No session directories found. Check --sessions.")

    streams = load_streams(session_dirs, sigma=args.sigma)
    if not streams:
        sys.exit("No prepared sessions. Run prepare_data.py first.")

    if args.holdout == "none":
        # Final fit: no validation at all. Every session trains, for a FIXED
        # epoch budget and at a FIXED threshold, both carried over from a
        # prior --holdout session run. There is nothing held out to select
        # against here, so early stopping and the threshold sweep are both
        # disabled — leaving them on would be selecting against training
        # data, which is how a model ends up tuned to its own noise.
        print(f"\n{'='*66}")
        print(f"  Move Onset Detector — FINAL FIT (no validation)")
        print(f"{'='*66}")
        print(f"  Sessions: {len(streams)} (all training), "
              f"{sum(len(s) for s in streams)} frames, "
              f"{sum(len(s.onset_idx) for s in streams)} onsets")
        print(f"  Epochs:    {args.epochs} (fixed — carry from the session run)")
        print(f"  Threshold: {args.threshold} (fixed — carry from the session run)")
        print(f"\n  The F1 printed each epoch is measured ON TRAINING DATA.")
        print(f"  It is a progress indicator, NOT a performance estimate.")
        print(f"  The honest number is whatever --holdout session reported.")
        train_ds = OnsetClipDataset(streams, clip_len=args.clip_len,
                                    stride=args.stride, augment=True)
        val_s = streams
    elif args.holdout == "session":
        train_s, val_s = split_streams(streams, args.val_sessions,
                                       val_names=args.val_session_names)
        report_data(train_s, val_s, args.clip_len)
        train_ds = OnsetClipDataset(train_s, clip_len=args.clip_len,
                                    stride=args.stride, augment=True)
        train_sampler = None
    else:
        # Pooled: every session trains, clips split randomly 80/20.
        full_ds = OnsetClipDataset(streams, clip_len=args.clip_len,
                                   stride=args.stride, augment=True)
        train_idx, val_idx = split_clips_pooled(full_ds, args.val_frac)
        leak = clip_leakage(full_ds, train_idx, val_idx, args.clip_len)

        print(f"\n{'='*66}")
        print(f"  Move Onset Detector — Training")
        print(f"{'='*66}")
        print(f"  Sessions: {len(streams)} (all used for training), "
              f"{sum(len(s) for s in streams)} frames, "
              f"{sum(len(s.onset_idx) for s in streams)} onsets")
        print(f"  Split:    pooled clips "
              f"{100*(1-args.val_frac):.0f}/{100*args.val_frac:.0f} — "
              f"{len(train_idx)} train / {len(val_idx)} val")
        print(f"\n  !! Clip overlap at clip_len={args.clip_len} "
              f"stride={args.stride}: "
              f"{overlap_frac(args.clip_len, args.stride)*100:.0f}% between "
              f"neighbours")
        print(f"  !! {leak*100:.1f}% of validation frames also appear inside "
              f"a training clip.")
        if leak > 0.5:
            print(f"  !! The val F1 below is close to training-set "
                  f"performance and should NOT")
            print(f"  !! be read as generalisation. Use --holdout session "
                  f"for that number.")
        print(f"  !! (--stride >= --clip-len drives this to 0%.)")

        train_ds = torch.utils.data.Subset(full_ds, train_idx)
        # Validation still scores whole sessions — the model needs real
        # temporal context either way, and with this much overlap the val
        # clips already span nearly every frame.
        val_s = streams

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(device, dropout=args.dropout)
    n_par  = sum(p.numel() for p in model.parameters())

    print(f"\n  Clips: {len(train_ds)} of {args.clip_len} frames "
          f"(stride {args.stride})")
    print(f"  Model: {n_par/1e6:.2f}M params, "
          f"receptive field {model.receptive_field} frames "
          f"(~{model.receptive_field/30:.1f}s at 30fps)")
    print(f"  Device: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")
    print(f"  Target: Gaussian sigma={args.sigma} frames  |  "
          f"decode: thr={args.threshold} min_sep={args.min_sep} "
          f"tol=±{args.tolerance}f")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1, best_epoch = 0.0, 0
    out_path = Path(args.output)

    print(f"\n  {'Epoch':<6} {'Loss':<10} {'Val F1':<9} {'P':<8} {'R':<8} {'|err|'}")
    print(f"  {'-'*54}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        scheduler.step()

        agg, _, _ = evaluate_streams(model, val_s, device, args.threshold,
                                     args.min_sep, args.tolerance)
        is_final_fit = args.holdout == "none"
        marker = "" if is_final_fit else (" ← best" if agg["f1"] > best_f1 else "")
        print(f"  {epoch:<6} {total/max(count,1):<10.4f} "
              f"{agg['f1']*100:<8.1f}% {agg['precision']*100:<7.1f}% "
              f"{agg['recall']*100:<7.1f}% {agg['median_err']:.1f}f{marker}")

        if is_final_fit:
            # Save every epoch so the run ends on the LAST epoch, not the
            # best-scoring one (which would be selection against train data).
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_f1": None,
                "holdout": "none",
                "val_session_names": [],
                "train_session_names": [s.name for s in streams],
                "sigma": args.sigma,
                "clip_len": args.clip_len,
                "threshold": args.threshold,
                "min_sep": args.min_sep,
                "tolerance": args.tolerance,
            }, out_path)
            continue

        if agg["f1"] > best_f1:
            best_f1, best_epoch = agg["f1"], epoch
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_f1": agg["f1"],
                "holdout": args.holdout,
                "val_session_names": [s.name for s in val_s],
                "sigma": args.sigma,
                "clip_len": args.clip_len,
                "threshold": args.threshold,
                "min_sep": args.min_sep,
                "tolerance": args.tolerance,
            }, out_path)

        if epoch - best_epoch >= args.patience:
            print(f"  Early stop: no F1 improvement in {args.patience} epochs")
            break

    if args.holdout == "none":
        print(f"\n  Final fit complete — {args.epochs} epochs on all "
              f"{len(streams)} sessions.")
        print(f"  Saved to: {out_path}  (threshold {args.threshold}, "
              f"carried from the session run)")
        print(f"\n  No performance number is reported here by design; this "
              f"model has no held-out")
        print(f"  data. Cite the --holdout session result when describing "
              f"how well it works.")
        return

    print(f"\n  Best onset F1: {best_f1*100:.1f}% (epoch {best_epoch})")

    # Tune the operating point on the best checkpoint
    ckpt = torch.load(out_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    _, _, scores = evaluate_streams(model, val_s, device, args.threshold,
                                    args.min_sep, args.tolerance)

    all_scores = np.concatenate([scores[s.name] for s in val_s])
    offsets, shift = [], 0
    for s in val_s:
        offsets.append(s.onset_idx + shift)
        shift += len(s)
    best_t, _ = sweep_threshold(all_scores, np.concatenate(offsets),
                                min_sep=args.min_sep, tolerance=args.tolerance,
                                beta=args.beta)

    agg, per_session, _ = evaluate_streams(model, val_s, device, best_t,
                                           args.min_sep, args.tolerance)
    ckpt["threshold"] = best_t
    ckpt["val_f1"]    = agg["f1"]
    ckpt["beta"]      = args.beta
    torch.save(ckpt, out_path)

    print(f"  Tuned threshold: {best_t:.2f} (was {args.threshold}), "
          f"selected by F{args.beta:g} — recall is worth more than "
          f"precision here (see decode.sweep_threshold)")
    report_final(agg, per_session, args)
    print(f"\n  Model saved to: {out_path}")


def report_final(agg, per_session, args):
    print(f"\n  Held-out performance at the tuned threshold:")
    print(f"  {'-'*72}")
    for name, m in per_session:
        print(f"  {format_metrics(m, name[-13:])}")
    print(f"  {'-'*72}")
    print(f"  {format_metrics(agg, 'AGGREGATE')}")

    print(f"\n  Reading this:")
    print(f"    fn = missed turns. Some are irreducible — two-handed "
          f"simultaneous moves")
    print(f"         (~1.8% of pairs) overlap in time and cannot produce "
          f"two peaks.")
    print(f"    fp = phantom turns, usually cube rotations or regrips. "
          f"Raise the")
    print(f"         threshold to trade these against fn.")
    print(f"    |err| = median frames between a detected peak and the BLE "
          f"timestamp;")
    print(f"         above ~{args.tolerance}f the classifier windows start "
          f"drifting off-centre.")


def evaluate(args):
    session_dirs = resolve_sessions(args.sessions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)

    streams = load_streams(session_dirs, sigma=ckpt.get("sigma", SIGMA))
    if not streams:
        sys.exit("No prepared sessions found. Run prepare_data.py first.")

    names = set(ckpt.get("val_session_names", []))
    val_s = [s for s in streams if s.name in names]
    if not val_s:
        sys.exit(f"None of this checkpoint's held-out sessions are present in "
                 f"--sessions.\nIt was validated on: {', '.join(sorted(names))}")
    if len(val_s) < len(names):
        print(f"WARNING: {len(names) - len(val_s)} held-out session(s) not "
              f"found in --sessions")

    model = build_model(device)
    model.load_state_dict(ckpt["state_dict"])

    threshold = args.threshold if args.threshold != THRESHOLD \
        else ckpt.get("threshold", THRESHOLD)
    min_sep   = ckpt.get("min_sep", args.min_sep)

    print(f"\nLoaded {args.model} (epoch {ckpt['epoch']}, "
          f"val F1 {ckpt['val_f1']*100:.1f}% at save time)")
    print(f"Evaluating on {len(val_s)} fully held-out session(s) at "
          f"threshold {threshold:.2f}, min_sep {min_sep}")

    agg, per_session, _ = evaluate_streams(model, val_s, device, threshold,
                                           min_sep, args.tolerance)
    report_final(agg, per_session, args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train the move-onset detector (CNN encoder + TCN)")
    p.add_argument("--sessions", nargs="+", required=True,
                   help="Session folder(s) — supports globs: "
                        "../training_data/solve_*/")
    p.add_argument("--epochs",  type=int,   default=60)
    p.add_argument("--batch",   type=int,   default=8,
                   help="Clips per batch (each clip is --clip-len frames)")
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip-len", type=int,  default=CLIP_LEN,
                   help="Frames per training clip; must exceed the 61-frame "
                        "receptive field")
    p.add_argument("--stride",  type=int,   default=24,
                   help="Frame stride between training clips")
    p.add_argument("--sigma",   type=float, default=SIGMA,
                   help="Gaussian onset target width, in frames")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="Peak threshold used during training; the best "
                        "checkpoint is re-tuned by sweep afterwards")
    p.add_argument("--beta",    type=float, default=BETA,
                   help="Recall:precision weight when re-tuning the "
                        "threshold. >1 favours recall; 1.0 restores the old "
                        "F1 selection")
    p.add_argument("--min-sep", type=int,   default=MIN_SEP,
                   help="Refractory period between peaks, in frames")
    p.add_argument("--tolerance", type=int, default=TOLERANCE,
                   help="Frames of slack when matching a peak to BLE truth")
    p.add_argument("--holdout", type=str, default="session",
                   choices=["session", "pooled", "none"],
                   help="session = hold out whole solves (honest "
                        "generalisation); pooled = every solve trains, clips "
                        "split randomly (val frames overlap train frames — "
                        "see the leakage warning it prints); none = final "
                        "fit on everything for a fixed --epochs and "
                        "--threshold carried from a session run")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Validation fraction for --holdout pooled")
    p.add_argument("--val-sessions", type=int, default=None,
                   help="How many whole sessions to hold out (default: 20%%)")
    p.add_argument("--val-session-names", nargs="+", default=None,
                   help="Hold out these sessions by name instead of picking "
                        "at random. Use this when the sessions span several "
                        "recording environments, so the holdout can be made "
                        "to span them too")
    p.add_argument("--patience", type=int,  default=15)
    p.add_argument("--output",  type=str,   default=MODEL_PATH)
    p.add_argument("--eval",    action="store_true",
                   help="Evaluate an existing checkpoint instead of training")
    p.add_argument("--model",   type=str,   default=MODEL_PATH)
    args = p.parse_args()

    if args.clip_len <= 61:
        sys.exit(f"--clip-len {args.clip_len} is not longer than the model's "
                 f"61-frame receptive field; most of each clip would be "
                 f"padding. Use 96 or more.")

    evaluate(args) if args.eval else train(args)
