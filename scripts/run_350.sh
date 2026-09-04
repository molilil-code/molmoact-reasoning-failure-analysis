#!/bin/bash

set -e

CKPT=/root/autodl-tmp/molmoact_project/models/MolmoAct-7B-D-Pretrain-0812
OUT=/root/autodl-tmp/molmoact_project/results/formal_350

echo "===== Pick Coke Can: 150 ====="

python experiments/run_rollouts.py \
    --checkpoint "$CKPT" \
    --env-name google_robot_pick_coke_can \
    --episodes 150 \
    --max-steps 80 \
    --backend vllm \
    --start-seed 10000 \
    --output-dir "$OUT" \
    --save-video


echo "===== Open Drawer: 100 ====="

python experiments/run_rollouts.py \
    --checkpoint "$CKPT" \
    --env-name google_robot_open_drawer \
    --episodes 100 \
    --max-steps 80 \
    --backend vllm \
    --start-seed 20000 \
    --output-dir "$OUT" \
    --save-video


echo "===== Move Near: 100 ====="

python experiments/run_rollouts.py \
    --checkpoint "$CKPT" \
    --env-name google_robot_move_near \
    --episodes 100 \
    --max-steps 80 \
    --backend vllm \
    --start-seed 30000 \
    --output-dir "$OUT" \
    --save-video


echo "===== ALL 350 EPISODES FINISHED ====="