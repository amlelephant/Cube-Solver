"""
train_ctc.py

Trains the Stage A trunk with a CTC objective instead of per-frame
alignment supervision — the second of the two onset-representation arms
(the first is train_joint.py --count-head).

What is and is not being changed
--------------------------------
The architecture is IDENTICAL to train_joint.py's: same FrameEncoder, same
TCN, same 13-way class head. Only two things differ, and they are the
things under test:

  * the LOSS. train_joint.py supervises where every move is, frame by
    frame (a Gaussian bump per onset). CTC supervises only WHICH MOVES, IN
    WHAT ORDER, and marginalises over every alignment that could have
    produced them. The 13th column stops meaning "background" and starts
    meaning CTC's blank.
  * the DECODER. peak-picking is gone; ctc_decode.prefix_beam_decode
    searches over frames, deciding move existence inside the search.

Keeping the trunk fixed is what makes the comparison readable. If this
loses, it loses because of the objective and the decoder, not because it
was also a different network.

The obvious objection, stated up front: we HAVE alignment labels, free,
from BLE, and CTC throws them away. That is a real cost and the reason
this is an experiment rather than a migration. The reason to run it anyway
is that the alignment labels are partly wrong exactly where the pipeline
fails — 10.1% of BLE moves share a 30ms tick and cannot be assigned
distinct frames, and those crowded pairs are the same population
peak-picking loses to merged humps. CTC is indifferent to both.

--aux-onset-weight adds the old BCE onset loss back as an auxiliary (the
onset head is present either way). Default 0.0 = pure CTC, which is the
clean test. A nonzero value is the hybrid, worth trying only if pure CTC
shows signs of life but trains unstably.

Model selection is by MOVE ERROR RATE on the held-out sessions, not onset
F1: a CTC model produces no aligned onsets, so F1 is not computable for
it. MER is the move-level analogue of word error rate and, unlike F1, is
computable for BOTH arms — which makes it the one number the three models
in this experiment can be ranked on directly.

Usage:
    python train_ctc.py --sessions "../training_data/*" \\
        --val-session-names solve_20260721_102711 solve_20260722_101225 \\
        solve_20260723_105530_solve solve_20260724_100120_solve \\
        --seed 0 --output move_ctc_seed0.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
except ImportError:
    sys.exit("PyTorch not installed. Run: pip install torch torchvision")

from dataset import (CTCClipDataset, ctc_collate, seed_worker,
                     load_joint_streams, split_streams, check_crop_regime,
                     check_label_regime,
                     CLIP_LEN, SIGMA)
from model import build_joint_model, score_stream_joint
from ctc_decode import (BLANK, greedy_decode, prefix_beam_decode,
                        move_error_rate)

MODEL_PATH = "move_ctc.pt"


def resolve_sessions(patterns: list[str]) -> list[Path]:
    dirs = [Path(p) for pattern in patterns
            for p in (Path(".").glob(pattern) if "*" in pattern
                      else [Path(pattern)])]
    return sorted(d for d in dirs if d.is_dir())


def evaluate_streams(model, streams, device, beam: int = 0) -> tuple:
    """
    Move error rate over held-out sessions, aggregated by total edit
    distance over total true moves (not a mean of per-session rates, so a
    long session is not diluted by a short one).

    `beam=0` uses greedy best-path decoding — much faster, and what the
    training loop wants per epoch. A reported number should use a beam.
    """
    per_session = []
    tot_d = tot_n = 0
    tot = {"sub": 0, "ins": 0, "del": 0}

    for s in streams:
        _, class_prob, _ = score_stream_joint(model, s, device)
        log_probs = np.log(np.maximum(class_prob, 1e-12))
        if beam:
            labels, _ = prefix_beam_decode(log_probs, beam=beam)
        else:
            labels, _ = greedy_decode(log_probs)
        true = list(s.onset_class[np.argsort(s.onset_idx)])
        mer, parts = move_error_rate(labels, true)
        per_session.append((s.name, mer, parts))
        tot_d += parts["sub"] + parts["ins"] + parts["del"]
        tot_n += parts["n_true"]
        for k in tot:
            tot[k] += parts[k]

    agg = {"mer": tot_d / max(tot_n, 1), "n_true": tot_n, **tot}
    return agg, per_session


def train(args):
    session_dirs = resolve_sessions(args.sessions)
    if not session_dirs:
        sys.exit("No session directories found. Check --sessions.")

    torch.manual_seed(args.seed)

    streams = load_joint_streams(session_dirs, sigma=args.sigma,
                                 mmap=args.workers > 0)
    if not streams:
        sys.exit("No prepared colour sessions. Run "
                 "`prepare_data.py --color` first.")

    crop_regime = check_crop_regime(streams, allow_mixed=args.allow_uncropped)
    label_regime = check_label_regime(streams)
    train_s, val_s = split_streams(streams, args.val_sessions,
                                   val_names=args.val_session_names,
                                   seed=args.seed)

    print(f"\n{'='*70}")
    print(f"  CTC Move-Sequence Model — Training")
    print(f"{'='*70}")
    print(f"  Train: {len(train_s)} session(s), "
          f"{sum(len(s) for s in train_s)} frames, "
          f"{sum(len(s.onset_idx) for s in train_s)} onsets")
    print(f"  Val:   {len(val_s)} session(s) fully held out, "
          f"{sum(len(s.onset_idx) for s in val_s)} onsets")
    for s in val_s:
        print(f"           {s.name}  ({len(s)} frames, "
              f"{len(s.onset_idx)} onsets)")

    train_ds = CTCClipDataset(train_s, clip_len=args.clip_len,
                              stride=args.stride, augment=True,
                              seed=args.seed,
                              aug_strength=args.aug_strength,
                              speed_aug=args.speed_aug)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=True,
                              collate_fn=ctc_collate,
                              worker_init_fn=seed_worker if args.workers else None,
                              persistent_workers=bool(args.workers),
                              pin_memory=True,
                              generator=torch.Generator().manual_seed(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_joint_model(device, dropout=args.dropout)
    n_par = sum(p.numel() for p in model.parameters())

    lens = np.array([train_ds[i][2] for i in
                     range(0, len(train_ds), max(1, len(train_ds) // 200))])
    print(f"\n  Clips: {len(train_ds)} of {args.clip_len} frames "
          f"(stride {args.stride}), seed {args.seed}")
    print(f"  Labels/clip: mean {lens.mean():.1f}, max {lens.max()} "
          f"(vs {args.clip_len} input frames — CTC needs T >= L)")
    print(f"  Model: {n_par/1e6:.2f}M params, receptive field "
          f"{model.receptive_field} frames")
    print(f"  Loss:  CTC (blank={BLANK})"
          + (f" + {args.aux_onset_weight:g} * BCE(onset) aux"
             if args.aux_onset_weight else "  [pure CTC]"))
    print(f"  Device: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Same mixed-precision setup as train_joint.py, deliberately — the arms
    # have to differ only in the thing under test. The CTC loss itself
    # stays in fp32 (see the training step): its forward-backward sums over
    # every alignment of a 96-frame clip, and in fp16 those sums underflow.
    amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_mer, best_epoch = float("inf"), 0
    out_path = Path(args.output)

    print(f"\n  {'Epoch':<6} {'Loss':<9} {'Val MER':<10} {'sub':<7} "
          f"{'ins':<7} {'del':<7} {'time'}")
    print(f"  {'-'*68}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, count = 0.0, 0
        for x, labels, lens_b in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                _, class_logits = model(x)
            # CTCLoss wants (T, N, C) log-probs and CPU-side int lengths.
            log_probs = torch.log_softmax(class_logits.float(),
                                          dim=-1).transpose(0, 1)
            in_lens = torch.full((x.size(0),), x.size(1), dtype=torch.long)
            loss = ctc(log_probs, labels, in_lens, lens_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * x.size(0)
            count += x.size(0)
        scheduler.step()

        agg, _ = evaluate_streams(model, val_s, device)
        marker = " <- best" if agg["mer"] < best_mer else ""
        print(f"  {epoch:<6} {total/max(count,1):<9.4f} "
              f"{agg['mer']*100:<9.1f}% {agg['sub']:<7} {agg['ins']:<7} "
              f"{agg['del']:<7} {time.time()-t0:.0f}s{marker}")

        if agg["mer"] < best_mer:
            best_mer, best_epoch = agg["mer"], epoch
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_mer": agg["mer"],
                "val_parts": {k: agg[k] for k in ("sub", "ins", "del")},
                "holdout": "session",
                "val_session_names": [s.name for s in val_s],
                "train_session_names": [s.name for s in train_s],
                "crop_regime": crop_regime,
                "label_regime": label_regime,
                "sigma": args.sigma,
                "clip_len": args.clip_len,
                "seed": args.seed,
                "model_type": "ctc",
                "in_channels": model.encoder.net[0].in_channels,
                "n_classes": model.n_classes,
                "blank": BLANK,
                "aux_onset_weight": args.aux_onset_weight,
                "aug_strength": args.aug_strength,
                "speed_aug": args.speed_aug,
            }, out_path)

        if epoch - best_epoch >= args.patience:
            print(f"  Early stop: no MER improvement in {args.patience} epochs")
            break

    if best_epoch == 0:
        sys.exit("\n  Never improved on an infinite MER — nothing saved.")

    print(f"\n  Best greedy MER: {best_mer*100:.1f}% (epoch {best_epoch})")

    ckpt = torch.load(out_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    agg, per_session = evaluate_streams(model, val_s, device, beam=args.beam)
    ckpt["val_mer_beam"] = agg["mer"]
    ckpt["beam"] = args.beam
    torch.save(ckpt, out_path)

    print(f"\n  Held-out, prefix beam search (beam={args.beam}):")
    print(f"  {'-'*68}")
    for name, mer, parts in per_session:
        print(f"    {name[-24:]:<26} MER {mer*100:5.1f}%   "
              f"sub {parts['sub']:<4} ins {parts['ins']:<4} "
              f"del {parts['del']:<4} (of {parts['n_true']} moves)")
    print(f"  {'-'*68}")
    print(f"    {'AGGREGATE':<26} MER {agg['mer']*100:5.1f}%   "
          f"sub {agg['sub']:<4} ins {agg['ins']:<4} del {agg['del']:<4} "
          f"(of {agg['n_true']} moves)")
    print(f"\n  ins = phantoms, del = misses — the two quantities this "
          f"experiment exists to move.")
    print(f"  Compare against train_joint.py's model through the same "
          f"metric with:")
    print(f"    python verify_ctc.py --ctc-model {out_path} "
          f"--joint-model checkpoints/move_joint_seed0.pt")
    print(f"\n  Model saved to: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip-len", type=int, default=CLIP_LEN)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--sigma", type=float, default=SIGMA,
                   help="only affects the (unused) onset target; kept so "
                        "streams load identically to train_joint.py")
    p.add_argument("--aux-onset-weight", type=float, default=0.0,
                   help="0 = pure CTC (the clean test); >0 = hybrid")
    p.add_argument("--aug-strength", type=float, default=1.0,
                   help="Scales every photometric augmentation toward "
                        "identity (dataset.AUG_*). 1.0 = the widened "
                        "ranges added 2026-07-31 for the evening-lighting "
                        "gap; 0 = geometry only. The pre-07-31 checkpoints "
                        "were trained on a much narrower range than any "
                        "value here reproduces, so compare against them by "
                        "checkpoint, not by this flag.")
    p.add_argument("--speed-aug", type=float, default=0.0,
                   help="P(a clip is time-warped to look like a faster "
                        "solve); dataset.AUG_SPEED sets the multiplier "
                        "range. 0 = off, reproducing every checkpoint "
                        "before 2026-08-05. The corpus runs ~2.4 TPS and "
                        "speed_sim.py measured count retention falling to "
                        "0.81 at 6 TPS, so the onset arm has simply never "
                        "seen a fast turn. VAL MER WILL LOOK WORSE with "
                        "this on and that is expected — the val split is "
                        "slow footage too, so the extra capacity spent on "
                        "speeds it does not contain scores as a regression "
                        "there. Judge it on speed_sim.py's retention curve, "
                        "which is the thing it is meant to move.")
    p.add_argument("--grad-clip", type=float, default=5.0,
                   help="CTC gradients spike early; clipping is not optional")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--no-amp", action="store_true",
                   help="disable mixed precision (1.7x slower)")
    p.add_argument("--val-sessions", type=int, default=None)
    p.add_argument("--val-session-names", nargs="+", default=None)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--output", type=str, default=MODEL_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=3,
                   help="dataloader workers; each costs ~1.8GB (Windows "
                        "spawn copies the streams). 0 starves the GPU.")
    p.add_argument("--allow-uncropped", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
