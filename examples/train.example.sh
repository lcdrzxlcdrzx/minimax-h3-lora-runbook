#!/usr/bin/env bash
set -euo pipefail

# Replace WORKDIR and model paths before use. Run from musubi-tuner's project root.
WORKDIR="${WORKDIR:?Set WORKDIR to the absolute runbook workspace path}"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/.venv/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$WORKDIR/.venv/bin/accelerate}"

"$ACCELERATE_BIN" launch --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  minimax_h3_train_network.py \
  --dit "$WORKDIR/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
  --int8_convrot_base \
  --dataset_config "$WORKDIR/configs/dataset-train.toml" \
  --h3_training_mode ref2va_omni \
  --network_module networks.lora_minimax_h3 \
  --network_dim 16 --network_alpha 16 \
  --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --blocks_to_swap 24 --block_swap_h2d_only --block_swap_ring_size 2 \
  --optimizer_type AdamW8bit --learning_rate 1e-4 \
  --max_train_steps 1 \
  --save_every_n_steps 20 --save_state --autoresume \
  --output_dir "$WORKDIR/outputs" --output_name h3_ref2va_lora_smoke

# Smoke-test gate: use max_train_steps=1 first. Then change to 10, verify resume,
# measure VRAM and stable seconds/step, and only then configure a long run.
