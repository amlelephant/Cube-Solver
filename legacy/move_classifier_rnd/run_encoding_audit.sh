#!/bin/sh
# End-to-end scoring for the 2026-07-27 encoding sweep.
#
# The held-out val number the trainer prints is measured on RECORDED
# sessions and on the trainer's own canonical windows. What the pipeline
# actually meets is a live take: windows anchored on detected onsets, crops
# from the live detector, and sessions that no classifier has ever trained
# on (these 10 have no moves_labeled.jsonl at all, so they cannot leak).
# Same 10 takes, same detector, same threshold for every checkpoint —
# only the classifier changes.
TAKES="../training_data/solve_20260724_134516_scramble ../training_data/solve_20260724_134516_solve
../training_data/solve_20260725_134032_scramble ../training_data/solve_20260725_134032_solve
../training_data/solve_20260725_134744_scramble ../training_data/solve_20260725_134744_solve
../training_data/solve_20260725_180216_scramble ../training_data/solve_20260725_180216_solve
../training_data/solve_20260726_100142_scramble ../training_data/solve_20260726_100142_solve"

cd move_detector || exit 1
for E in flowres flow flowwheel; do
  CKPT="../move_classifier_enc_${E}.pt"
  [ -f "$CKPT" ] || { echo "skip $E (no checkpoint)"; continue; }
  echo "=== $E ==="
  PYTHONIOENCODING=utf-8 python -u metric_audit.py --sessions $TAKES \
    --classifier "$CKPT" \
    --json "../metric_audit_enc_${E}.json" \
    > "../metric_audit_enc_${E}.log" 2>&1
  grep -A3 "metric  " "../metric_audit_enc_${E}.log" | head -4
done
