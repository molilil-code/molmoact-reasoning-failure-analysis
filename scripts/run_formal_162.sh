#!/usr/bin/env bash

set -euo pipefail

CKPT=/root/autodl-tmp/molmoact_project/models/MolmoAct-7B-D-Pretrain-0812
ROOT=/root/autodl-tmp/molmoact_project/results/formal
SEED=20260905

cd /root/autodl-tmp/molmoact_project/code/molmoact-reasoning-failure-analysis

mkdir -p "$ROOT"

echo "========================================"
echo "FORMAL MOLMOACT ROLLOUT COLLECTION"
echo "backend = HF"
echo "seed    = $SEED"
echo "========================================"

echo
echo "===== [1/3] PICK COKE CAN: 75 ====="

python experiments/run_rollouts.py \
  --checkpoint "$CKPT" \
  --task pick_coke_can \
  --backend hf \
  --start-index 0 \
  --limit 75 \
  --base-seed "$SEED" \
  --output-dir "$ROOT" \
  --save-video \
  --resume


echo
echo "===== [2/3] MOVE NEAR: 60 ====="

python experiments/run_rollouts.py \
  --checkpoint "$CKPT" \
  --task move_near \
  --backend hf \
  --start-index 0 \
  --limit 60 \
  --base-seed "$SEED" \
  --output-dir "$ROOT" \
  --save-video \
  --resume


echo
echo "===== [3/3] OPEN DRAWER: 27 ====="

python experiments/run_rollouts.py \
  --checkpoint "$CKPT" \
  --task open_drawer \
  --backend hf \
  --start-index 0 \
  --limit 27 \
  --base-seed "$SEED" \
  --output-dir "$ROOT" \
  --save-video \
  --resume


echo
echo "========================================"
echo "FORMAL COLLECTION COMPLETE"
echo "========================================"

find "$ROOT" -name summary.json | wc -l

df -h /root/autodl-tmp
