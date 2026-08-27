#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATA_ROOT="${DATA_ROOT:-data/train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eda-llama31-gsmplus-seed123}"

python train.py \
  --model-name-or-path "${MODEL_PATH}" \
  --source-train "${DATA_ROOT}/source8k_random10p_seed123.jsonl" \
  --target-train "${DATA_ROOT}/target8k_random10p_seed123.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --log-dir "${OUTPUT_DIR}/tb" \
  --use-qlora \
  --train-batch-size 10 \
  --gradient-accumulation-steps 64 \
  --learning-rate 1e-6 \
  --num-train-epochs 2 \
  --cutoff-length 4096 \
  --domain-loss-weight 0.2 \
  --activation-pooling mean \
  --enable-multi-domain \
  --num-target-domains 8 \
  --target-domain-key variant_id \
  --esm-use-fusion \
  --esm-fusion-method film \
  --esm-use-adapter \
  --esm-adapter-depth 15 \
  --enable-als \
  --als-temperature-init 5.0 \
  --als-temperature-min 0.5 \
  --als-use-ste \
  --als-ste-threshold 0.1 \
  --als-distance-weight 1 \
  --als-distance-metric l2 \
  --discriminator-pretrain-steps 19 \
  --discriminator-pretrain-layer 4 \
  --seed 123
