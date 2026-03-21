#!/usr/bin/env bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
N_GPUS=8
BATCH_SIZE=64
REPO_ROOT="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/results/logs_tri_seq"
HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/results"

# ── Define ONLY TRILINGUAL_SEQUENTIAL_MODELS ──────────────────────────────
MODELS=(
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h25"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h50"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h100"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h25"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h50"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h75"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h100"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h25"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h50"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h75"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h100"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h25"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h50"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h75"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h100"

# EWC only
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150"

# MAML only
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h25"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h50"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h75"
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h100"

# Baseline
"BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_baseline"
)

# ── Export as comma-separated string ──────────────────────────────────────
MODEL_LIST=$(IFS=,; echo "${MODELS[*]}")
export MODEL_LIST

echo "Running ${#MODELS[@]} trilingual SEQUENTIAL models"

# ── Launch ────────────────────────────────────────────────────────────────
PIDS=()

for (( rank=0; rank<N_GPUS; rank++ )); do
  log="${LOG_DIR}/rank${rank}.log"

  CUDA_VISIBLE_DEVICES="${rank}" \
  MODEL_LIST="${MODEL_LIST}" \
  python3 "${SCRIPT_DIR}/eval_model.py" \
    --rank "${rank}" \
    --world_size "${N_GPUS}" \
    --output_dir "${REPO_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --resume \
    > "${log}" 2>&1 &

  PIDS+=($!)
  sleep 3
done

echo "Launched ${#PIDS[@]} jobs"

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "Done."
