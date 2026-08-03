#!/bin/sh
# 2026-07-27 encoding sweep. Every run: same 39 sessions, same named
# holdout, same recipe as the deployed all39_jitter model. The control is
# RETRAINED rather than cited so the comparison is same-sitting and
# same-code-path on both sides.
HOLD="solve_20260720_142006 solve_20260721_102711 solve_20260722_101225 solve_20260723_105530_solve solve_20260724_100120_solve"
for E in diffstack chroma chroma8 rgbtime rgbtime0; do
  echo "=== $E ==="
  PYTHONIOENCODING=utf-8 python -u train_move_classifier.py \
    --sessions "training_data/solve_*/" \
    --encoding "$E" --anchor-jitter --epochs 40 --workers 16 \
    --val-session-names $HOLD \
    --output "move_classifier_enc_${E}.pt" \
    > "training_run_20260727_enc_${E}.log" 2>&1
  echo "$E done: $(grep 'Best val accuracy' training_run_20260727_enc_${E}.log)"
done
