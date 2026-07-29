#!/bin/sh
# 2026-07-27 optical-flow sweep. Same 39 sessions, same named holdout and
# same recipe as the encoding sweep, so the diffstack pair already measured
# (deployed 86.0 / fresh control 83.7 live) is the envelope these are read
# against. Seeded, per the seed-variance finding.
HOLD="solve_20260720_142006 solve_20260721_102711 solve_20260722_101225 solve_20260723_105530_solve solve_20260724_100120_solve"
for E in flowres flow flowwheel; do
  echo "=== $E ==="
  PYTHONIOENCODING=utf-8 python -u train_move_classifier.py \
    --sessions "training_data/solve_*/" \
    --encoding "$E" --anchor-jitter --epochs 40 --workers 16 --seed 1 \
    --val-session-names $HOLD \
    --output "move_classifier_enc_${E}.pt" \
    > "training_run_20260727_enc_${E}.log" 2>&1
  echo "$E done: $(grep 'Best val accuracy' training_run_20260727_enc_${E}.log)"
done
