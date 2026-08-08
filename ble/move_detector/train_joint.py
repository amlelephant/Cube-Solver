"""
train_joint.py

Stage A (MODEL_REWORK_PLAN.md): trains the joint onset+class detector — a
single frame-synchronous model emitting, per frame, both "is a move
happening" (unchanged from the shipped detector) and "which of the 12 WCA
quarter turns is it, or background" (new). This dissolves the peak-pick
-> fixed-window -> separate-classifier seam that G2's oracle attribution
(oracle_attribution.py, 2026-07-28) found responsible for ~81% of the
honest verification gap.

Completely parallel to train.py, not a modification of it: reads
detector_stream_color.npz (prepare_data.py --color) instead of the
deployed detector_stream.npz, builds model.build_joint_model() instead of
build_model(), and writes its own checkpoint file. Nothing here can affect
the shipped checkpoints/move_detector_all28.pt / move_classifier_all39_jitter.pt
pair.

Loss: per-frame BCE on the onset head (unchanged) + soft-label cross-
entropy on the class head (dataset.build_dense_targets already produces a
valid 13-way probability simplex per frame, so this is literally
-sum(target * log_softmax(pred)), no one-hot conversion needed).
--class-weight balances the two terms; background dominates ~85-90% of
frames by frame count, but NOT by loss magnitude once weighted this
way — measure before assuming an additional per-class reweighting is
needed (MODEL_REWORK_PLAN.md Stage A explicitly flags this as something
to measure, not assume, citing the onset head's own --pos-weight history
where the "obviously needed" reweighting turned out not to matter).

Model selection is by onset F1 (same criterion train.py uses, so a run is
comparable to the deployed detector's own selection), with class accuracy
reported alongside every epoch for visibility — see evaluate_streams.

Usage:
    python prepare_data.py --sessions ../training_data/solve_*/ --color
    python train_joint.py --sessions ../training_data/solve_*/ \\
        --val-session-names solve_20260721_102711 solve_20260722_101225 \\
        solve_20260723_105530_solve solve_20260724_100120_solve \\
        --seed 0 --output checkpoints/move_joint_seed0.pt
    python train_joint.py --sessions ../training_data/solve_*/ --eval \\
        --model checkpoints/move_joint_seed0.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

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

from dataset import (JointSessionStream, JointClipDataset, load_joint_streams,
                     split_streams, check_crop_regime, check_label_regime,
                     seed_worker,
                     CLIP_LEN, SIGMA, COUNT_RADIUS, COUNT_CLASSES)
from model import build_joint_model, build_joint_from_ckpt, score_stream_joint
from decode import peak_pick, match_onsets, sweep_threshold, format_metrics, \
    MIN_SEP, THRESHOLD, TOLERANCE, BETA

MODEL_PATH = "move_joint.pt"


def resolve_sessions(patterns: list[str]) -> list[Path]:
    dirs = [Path(p) for pattern in patterns
            for p in (Path(".").glob(pattern) if "*" in pattern
                      else [Path(pattern)])]
    return sorted(d for d in dirs if d.is_dir())


def class_accuracy(class_prob: np.ndarray, stream: JointSessionStream
                   ) -> dict:
    """
    Two class-head metrics, deliberately kept separate:

    at_onset   argmax over the 12 move classes (columns 0-11, background
               EXCLUDED) at each TRUE onset frame, vs the true class — "if
               the model knows a move is here, does it name it right?"
               Comparable in spirit to the deployed classifier's own
               accuracy number, but scored on the detector's own frame
               rather than a hand-built window.
    frame_bg   argmax over the FULL 13 columns at every frame in the
               session vs the dense target's own argmax — "does the model
               separate move-frames from background at all?" This is the
               metric MODEL_REWORK_PLAN.md's Stage A says to prioritise
               (G2: detector-side recall/background separation closes
               ~81% of the honest gap, vs ~12% from classifier accuracy).
    """
    onset_idx, onset_class = stream.onset_idx, stream.onset_class
    if len(onset_idx):
        pred = class_prob[onset_idx, :12].argmax(axis=1)
        at_onset = float((pred == onset_class).mean())
    else:
        at_onset = float("nan")

    frame_pred = class_prob.argmax(axis=1)
    frame_true = stream.class_target.argmax(axis=1)
    frame_bg = float((frame_pred == frame_true).mean())
    return {"at_onset": at_onset, "frame_bg": frame_bg}


def evaluate_streams(model, streams: list[JointSessionStream], device,
                     threshold: float = THRESHOLD, min_sep: int = MIN_SEP,
                     tolerance: int = TOLERANCE):
    """Joint counterpart of train.evaluate_streams: onset F1 (identical
    construction) plus the two class-accuracy numbers above, aggregated by
    onset-count-weighted mean (not per-session average) so a long session
    does not count the same as a short one."""
    per_session, tp = [], 0
    fp = fn = 0
    errs = []
    n_onset_w, at_onset_sum, frame_bg_w, frame_bg_sum = 0, 0.0, 0, 0.0
    cnt_acc = []

    for s in streams:
        onset_prob, class_prob, count_prob = score_stream_joint(model, s, device)
        pred = peak_pick(onset_prob, threshold=threshold, min_sep=min_sep)
        m = match_onsets(pred, s.onset_idx, onset_prob, tolerance)
        acc = class_accuracy(class_prob, s)
        acc.update(count_accuracy(count_prob, s))
        cnt_acc.append(acc)
        per_session.append((s.name, m, acc))
        tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
        if not np.isnan(m["median_err"]):
            errs.append(m["median_err"])
        if len(s.onset_idx) and not np.isnan(acc["at_onset"]):
            n_onset_w += len(s.onset_idx)
            at_onset_sum += acc["at_onset"] * len(s.onset_idx)
        frame_bg_w += len(s)
        frame_bg_sum += acc["frame_bg"] * len(s)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0
    agg = {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
          "recall": recall, "f1": f1,
          "median_err": float(np.median(errs)) if errs else float("nan"),
          "bias": float("nan"),
          "at_onset": at_onset_sum / n_onset_w if n_onset_w else float("nan"),
          "frame_bg": frame_bg_sum / frame_bg_w if frame_bg_w else float("nan")}
    for k in ("count2_recall", "count2_prec", "pair_recall"):
        vals = [a[k] for a in cnt_acc if not np.isnan(a[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    return agg, per_session


def joint_loss(onset_logits, class_logits, y_onset, y_class,
              class_weight: float, onset_criterion,
              count_logits=None, y_count=None, count_weight: float = 0.0,
              count_criterion=None):
    """BCE(onset) + class_weight * soft-label CE(class), per-frame mean,
    plus count_weight * hard-label CE(count) when the count head is on."""
    onset_l = onset_criterion(onset_logits, y_onset)
    log_p = torch.log_softmax(class_logits, dim=-1)     # (B, T, 13)
    class_l = -(y_class * log_p).sum(dim=-1).mean()
    total = onset_l + class_weight * class_l
    count_v = 0.0
    if count_logits is not None:
        count_l = count_criterion(count_logits.reshape(-1, COUNT_CLASSES),
                                  y_count.reshape(-1))
        total = total + count_weight * count_l
        count_v = count_l.item()
    return total, onset_l.item(), class_l.item(), count_v


def count_class_weights(streams, device) -> torch.Tensor:
    """
    Inverse-frequency weights for the count head's 3-way CE.

    Necessary, not cosmetic: at COUNT_RADIUS=2 the target is 76.2% / 21.7%
    / 2.0% across the prepared sessions, and unweighted CE on a 2% class
    collapses to never predicting it — which would make the head a no-op
    dressed up as a measurement. Weights are normalised to mean 1 so
    count_weight stays comparable to class_weight.
    """
    counts = np.zeros(COUNT_CLASSES, dtype=np.float64)
    for s in streams:
        counts += np.bincount(s.count_target, minlength=COUNT_CLASSES)
    freq = counts / max(counts.sum(), 1)
    w = 1.0 / np.maximum(freq, 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device), counts


def count_accuracy(count_prob: np.ndarray, stream) -> dict:
    """
    Per-frame count metrics, reported separately for the rare class
    because overall accuracy is ~98% for a head that never predicts 2.

    count2_recall   of frames whose TRUE count is 2, how many are called 2
    count2_prec     of frames CALLED 2, how many truly are
    pair_recall     of true onset PAIRS within COUNT_RADIUS of each other,
                    how many have at least one frame called 2 between them
                    — the number that actually bounds what the decoder can
                    recover, since one flagged frame is enough to split a
                    peak (see joint_decode.posteriorgram_to_moves)
    """
    if count_prob is None:
        return {"count2_recall": float("nan"), "count2_prec": float("nan"),
                "pair_recall": float("nan")}
    pred = count_prob.argmax(axis=1)
    true = stream.count_target
    t2, p2 = (true >= 2), (pred >= 2)
    rec = float(p2[t2].mean()) if t2.any() else float("nan")
    prec = float(t2[p2].mean()) if p2.any() else float("nan")

    idx = np.sort(stream.onset_idx)
    pairs = [(a, b) for a, b in zip(idx[:-1], idx[1:])
             if b - a <= 2 * stream.count_radius]
    hit = sum(1 for a, b in pairs if p2[a:b + 1].any())
    return {"count2_recall": rec, "count2_prec": prec,
            "pair_recall": hit / len(pairs) if pairs else float("nan")}


def report_data(train_s, val_s):
    print(f"\n{'='*70}")
    print(f"  Joint Onset+Class Detector — Training (Stage A)")
    print(f"{'='*70}")
    print(f"  Train: {len(train_s)} session(s), "
          f"{sum(len(s) for s in train_s)} frames, "
          f"{sum(len(s.onset_idx) for s in train_s)} onsets")
    print(f"  Val:   {len(val_s)} session(s) fully held out, "
          f"{sum(len(s) for s in val_s)} frames, "
          f"{sum(len(s.onset_idx) for s in val_s)} onsets")
    for s in val_s:
        print(f"           {s.name}  ({len(s)} frames, "
              f"{len(s.onset_idx)} onsets, {s.fps:.1f}fps)")


def report_final(agg, per_session):
    print(f"\n  Held-out performance at the tuned threshold:")
    print(f"  {'-'*86}")
    for name, m, acc in per_session:
        print(f"  {format_metrics(m, name[-13:])}   "
              f"at_onset {acc['at_onset']*100:5.1f}%  "
              f"frame_bg {acc['frame_bg']*100:5.1f}%")
    print(f"  {'-'*86}")
    print(f"  {format_metrics(agg, 'AGGREGATE')}   "
          f"at_onset {agg['at_onset']*100:5.1f}%  "
          f"frame_bg {agg['frame_bg']*100:5.1f}%")
    print(f"\n  Reading this:")
    print(f"    onset F1/P/R  same construction as the shipped detector — "
          f"comparable directly.")
    print(f"    at_onset      of the frames a move WAS found on, is the "
          f"class right? (12-way,")
    print(f"                  background excluded) — closest analogue to "
          f"the deployed classifier's")
    print(f"                  own accuracy number, but on this model's own "
          f"onset frames.")
    print(f"    frame_bg      13-way argmax vs target argmax over EVERY "
          f"frame in the session —")
    print(f"                  the metric MODEL_REWORK_PLAN.md's Stage A "
          f"says to prioritise (G2:")
    print(f"                  detector-side recall/background separation "
          f"closes ~81%% of the honest")
    print(f"                  verification gap, classifier accuracy only "
          f"~12%%).")


def train(args):
    session_dirs = resolve_sessions(args.sessions)
    if not session_dirs:
        sys.exit("No session directories found. Check --sessions.")

    torch.manual_seed(args.seed)

    streams = load_joint_streams(session_dirs, mmap=args.workers > 0,
                                 sigma=args.sigma,
                                 count_radius=args.count_radius)
    if not streams:
        sys.exit("No prepared colour sessions. Run "
                 "`prepare_data.py --color` first.")

    crop_regime = check_crop_regime(streams, allow_mixed=args.allow_uncropped)
    label_regime = check_label_regime(streams)

    train_s, val_s = split_streams(streams, args.val_sessions,
                                   val_names=args.val_session_names,
                                   seed=args.seed)
    report_data(train_s, val_s)

    train_ds = JointClipDataset(train_s, clip_len=args.clip_len,
                                stride=args.stride, augment=True,
                                seed=args.seed,
                                aug_strength=args.aug_strength,
                                speed_aug=args.speed_aug)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=True,
                              worker_init_fn=seed_worker if args.workers else None,
                              persistent_workers=bool(args.workers),
                              pin_memory=True,
                              generator=torch.Generator().manual_seed(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_counts = COUNT_CLASSES if args.count_head else None
    model  = build_joint_model(device, dropout=args.dropout,
                               n_counts=n_counts)
    n_par  = sum(p.numel() for p in model.parameters())

    print(f"\n  Clips: {len(train_ds)} of {args.clip_len} frames "
          f"(stride {args.stride}), seed {args.seed}")
    print(f"  Model: {n_par/1e6:.2f}M params (joint, in_channels=4, "
          f"n_classes=13), receptive field {model.receptive_field} frames")
    print(f"  Device: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")
    print(f"  Target: Gaussian sigma={args.sigma} frames  |  class_weight="
          f"{args.class_weight:g}  |  decode: thr={args.threshold} "
          f"min_sep={args.min_sep} tol=+/-{args.tolerance}f")

    onset_criterion = nn.BCEWithLogitsLoss()
    count_criterion = None
    if args.count_head:
        cw, cdist = count_class_weights(train_s, device)
        count_criterion = nn.CrossEntropyLoss(weight=cw)
        share = 100 * cdist / max(cdist.sum(), 1)
        print(f"  Count head: radius={args.count_radius} "
              f"(+/-{args.count_radius}f = "
              f"{(2*args.count_radius+1)*1000/30:.0f}ms bin), "
              f"count_weight={args.count_weight:g}")
        print(f"              train target 0/1/2+ = {share[0]:.1f}% / "
              f"{share[1]:.1f}% / {share[2]:.2f}%, "
              f"CE weights {cw[0]:.2f}/{cw[1]:.2f}/{cw[2]:.2f}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision, measured 1.7x on this model (119 -> 71 ms/clip). The
    # encoder is memory-bandwidth bound on 768 frames of 96x96 activations
    # per batch, not FLOP bound, which is why fp16 helps and channels_last
    # does not (it is measurably WORSE here — the (B,T,C,H,W) flatten
    # defeats it).
    amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_f1, best_epoch = 0.0, 0
    out_path = Path(args.output)

    print(f"\n  {'Epoch':<6} {'Loss':<9} {'(ons/cls)':<12} {'Val F1':<9} "
          f"{'P':<8} {'R':<8} {'at_onset':<10} {'frame_bg'}")
    print(f"  {'-'*80}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, total_o, total_c, total_n, count = 0.0, 0.0, 0.0, 0.0, 0
        for x, y_onset, y_class, y_count in train_loader:
            x, y_onset, y_class = (x.to(device), y_onset.to(device),
                                   y_class.to(device))
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(x)
            # Losses in fp32: the encoder is the expensive part and the only
            # part autocast is for, while soft-label CE over a 13-way
            # softmax is exactly the kind of small reduction fp16 rounds
            # badly.
            onset_logits, class_logits = out[0].float(), out[1].float()
            count_logits = out[2].float() if args.count_head else None
            loss, lo, lc, ln = joint_loss(
                onset_logits, class_logits, y_onset, y_class,
                args.class_weight, onset_criterion,
                count_logits=count_logits,
                y_count=y_count.to(device) if args.count_head else None,
                count_weight=args.count_weight,
                count_criterion=count_criterion)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = x.size(0)
            total += loss.item() * bs
            total_o += lo * bs
            total_c += lc * bs
            total_n += ln * bs
            count += bs
        scheduler.step()

        agg, _ = evaluate_streams(model, val_s, device, args.threshold,
                                  args.min_sep, args.tolerance)
        marker = " <- best" if agg["f1"] > best_f1 else ""
        cnt_col = (f" pair_R {agg['pair_recall']*100:5.1f}%"
                   if args.count_head else "")
        print(f"  {epoch:<6} {total/max(count,1):<9.4f} "
              f"({total_o/max(count,1):.3f}/{total_c/max(count,1):.3f}"
              f"{'/'+format(total_n/max(count,1),'.3f') if args.count_head else ''}) "
              f"{agg['f1']*100:<8.1f}% {agg['precision']*100:<7.1f}% "
              f"{agg['recall']*100:<7.1f}% {agg['at_onset']*100:<9.1f}% "
              f"{agg['frame_bg']*100:.1f}%{cnt_col}{marker}")

        if agg["f1"] > best_f1:
            best_f1, best_epoch = agg["f1"], epoch
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_f1": agg["f1"],
                "val_at_onset": agg["at_onset"],
                "val_frame_bg": agg["frame_bg"],
                "holdout": "session",
                "val_session_names": [s.name for s in val_s],
                "train_session_names": [s.name for s in train_s],
                "crop_regime": crop_regime,
                "label_regime": label_regime,
                "aug_strength": args.aug_strength,
                "sigma": args.sigma,
                "class_weight": args.class_weight,
                "clip_len": args.clip_len,
                "threshold": args.threshold,
                "min_sep": args.min_sep,
                "tolerance": args.tolerance,
                "seed": args.seed,
                "model_type": "joint",
                "in_channels": model.encoder.net[0].in_channels,
                "n_classes": model.n_classes,
                "n_counts": model.n_counts,
                "count_radius": args.count_radius,
                "count_weight": args.count_weight,
                "val_pair_recall": agg["pair_recall"],
            }, out_path)

        if epoch - best_epoch >= args.patience:
            print(f"  Early stop: no F1 improvement in {args.patience} epochs")
            break

    print(f"\n  Best onset F1: {best_f1*100:.1f}% (epoch {best_epoch})")

    ckpt = torch.load(out_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    agg, _ = evaluate_streams(model, val_s, device, args.threshold,
                              args.min_sep, args.tolerance)
    all_onset = np.concatenate([
        score_stream_joint(model, s, device)[0] for s in val_s])
    offsets, shift = [], 0
    for s in val_s:
        offsets.append(s.onset_idx + shift)
        shift += len(s)
    best_t, _ = sweep_threshold(all_onset, np.concatenate(offsets),
                                min_sep=args.min_sep, tolerance=args.tolerance,
                                beta=args.beta)

    agg, per_session = evaluate_streams(model, val_s, device, best_t,
                                        args.min_sep, args.tolerance)
    ckpt["threshold"] = best_t
    ckpt["val_f1"] = agg["f1"]
    ckpt["val_at_onset"] = agg["at_onset"]
    ckpt["val_frame_bg"] = agg["frame_bg"]
    ckpt["val_pair_recall"] = agg["pair_recall"]
    ckpt["beta"] = args.beta
    torch.save(ckpt, out_path)

    print(f"  Tuned threshold: {best_t:.2f} (was {args.threshold}), "
          f"selected by F{args.beta:g}")
    report_final(agg, per_session)
    print(f"\n  Model saved to: {out_path}")


def evaluate(args):
    session_dirs = resolve_sessions(args.sessions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)

    streams = load_joint_streams(session_dirs, sigma=ckpt.get("sigma", SIGMA))
    if not streams:
        sys.exit("No prepared colour sessions. Run "
                 "`prepare_data.py --color` first.")

    trained = set(ckpt.get("train_session_names") or [])
    held = set(ckpt.get("val_session_names") or [])
    unseen_s = [s for s in streams if s.name not in trained and s.name not in held]
    held_s = [s for s in streams if s.name in held]
    train_s = [s for s in streams if s.name in trained]

    eval_s = unseen_s + held_s
    if train_s and args.allow_train_sessions:
        eval_s += train_s
    if not eval_s:
        sys.exit("Every requested session is training data for this "
                 "checkpoint; pass --allow-train-sessions to score anyway.")
    if train_s and not args.allow_train_sessions:
        print(f"\nSkipping {len(train_s)} training session(s) "
              f"(pass --allow-train-sessions to include them)")

    model = build_joint_from_ckpt(ckpt, device)
    threshold = ckpt.get("threshold", THRESHOLD)

    print(f"\nLoaded {args.model} (epoch {ckpt['epoch']}, seed "
          f"{ckpt.get('seed', '?')}, val F1 {ckpt.get('val_f1', 0)*100:.1f}%)")
    print(f"Evaluating {len(eval_s)} session(s) at threshold {threshold:.2f}")
    if unseen_s:
        print(f"  {len(unseen_s)} never seen in any form  <- the number to cite")
        for s in unseen_s:
            print(f"      {s.name}")
    if held_s:
        print(f"  {len(held_s)} held out during training")

    agg, per_session = evaluate_streams(model, eval_s, device, threshold,
                                        ckpt.get("min_sep", MIN_SEP),
                                        args.tolerance)
    report_final(agg, per_session)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train the joint onset+class detector (Stage A, "
                    "MODEL_REWORK_PLAN.md)")
    p.add_argument("--sessions", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-amp", action="store_true",
                   help="disable mixed precision (1.7x slower)")
    p.add_argument("--count-head", action="store_true",
                   help="add the 0/1/2 local-onset-count head "
                        "(dataset.build_count_target)")
    p.add_argument("--count-radius", type=int, default=COUNT_RADIUS,
                   help="count-target bin radius in frames")
    p.add_argument("--count-weight", type=float, default=1.0,
                   help="weight on the count head's CE loss")
    p.add_argument("--class-weight", type=float, default=1.0,
                   help="Weight on the class-head soft-CE term relative "
                        "to the onset BCE term")
    p.add_argument("--clip-len", type=int, default=CLIP_LEN)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--aug-strength", type=float, default=1.0,
                   help="Scales every photometric augmentation toward "
                        "identity (dataset.AUG_*); 1.0 = the widened "
                        "ranges added 2026-07-31, 0 = geometry only")
    p.add_argument("--speed-aug", type=float, default=0.0,
                   help="P(a clip is time-warped to look like a faster "
                        "solve). CAPPED at dataset.MAX_DENSE_SPEED here, "
                        "unlike train_ctc.py: the dense sigma=1 targets "
                        "cannot represent two onsets closer than a few "
                        "frames, so an uncapped warp would merge two peaks "
                        "into one and train the model to UNDER-count.")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--min-sep", type=int, default=MIN_SEP)
    p.add_argument("--tolerance", type=int, default=TOLERANCE)
    p.add_argument("--val-sessions", type=int, default=None)
    p.add_argument("--val-session-names", nargs="+", default=None,
                   help="Hold out these sessions by name. Pass the SAME "
                        "names the deployed detector/classifier use for a "
                        "directly comparable number.")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--output", type=str, default=MODEL_PATH)
    p.add_argument("--seed", type=int, default=0,
                   help="Weight init + data shuffling seed — run at least "
                        "two to quote a result inside the seed envelope "
                        "(see encoding-rework-flat memory: ~2.3pt spread "
                        "observed elsewhere in this project)")
    p.add_argument("--workers", type=int, default=3,
                   help="dataloader workers; each costs ~1.8GB (Windows "
                        "spawn copies the streams). 0 starves the GPU.")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--model", type=str, default=MODEL_PATH)
    p.add_argument("--allow-train-sessions", action="store_true")
    p.add_argument("--allow-uncropped", action="store_true")
    args = p.parse_args()

    if args.clip_len <= 61:
        sys.exit(f"--clip-len {args.clip_len} is not longer than the "
                 f"model's 61-frame receptive field.")

    evaluate(args) if args.eval else train(args)
