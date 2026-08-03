#!/bin/sh
# Three arms of the onset-representation experiment, on ONE identical
# 38-train/4-val split. The baseline is RETRAINED here rather than reusing
# checkpoints/move_joint_seed0.pt: that checkpoint saw 36 training sessions, two colour
# streams have been prepared since, and comparing a 38-session model to a
# 36-session number would attribute a data difference to an architecture
# difference.
set -e
PY="../../.venv/Scripts/python.exe"
VAL="solve_20260721_102711 solve_20260722_101225 solve_20260723_105530_solve solve_20260724_100120_solve"
SEED=${1:-0}
EPOCHS=${2:-40}
PATIENCE=10

echo "=== arm 0: baseline joint (no count head), seed $SEED ==="
"$PY" -u train_joint.py --sessions "../training_data/*" \
    --val-session-names $VAL --seed "$SEED" \
    --epochs "$EPOCHS" --patience "$PATIENCE" \
    --output "move_joint_base_s$SEED.pt" > "train_joint_base_s$SEED.log" 2>&1

echo "=== arm A: joint + count head, seed $SEED ==="
"$PY" -u train_joint.py --sessions "../training_data/*" \
    --val-session-names $VAL --seed "$SEED" --count-head \
    --epochs "$EPOCHS" --patience "$PATIENCE" \
    --output "move_joint_count_s$SEED.pt" > "train_joint_count_s$SEED.log" 2>&1

echo "=== arm B: CTC, seed $SEED ==="
"$PY" -u train_ctc.py --sessions "../training_data/*" \
    --val-session-names $VAL --seed "$SEED" \
    --epochs "$EPOCHS" --patience "$PATIENCE" \
    --output "move_ctc_s$SEED.pt" > "train_ctc_s$SEED.log" 2>&1

echo "=== all three arms done (seed $SEED) ==="
