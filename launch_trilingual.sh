#!/usr/bin/env bash
# =============================================================================
# launch_trilingual.sh — Launch BeetleLM trilingual evaluation sweep.
#
# Evaluates ONLY the 37 trilingual eng–nld–zho model repos across the
# benchmarks that cover those three languages:
#
#   blimp_eng  (English)
#   blimp_nl   (Dutch)
#   zhoblimp   (Chinese)
#   xcomps/zho (Chinese)
#   xnli/en    (English)
#   xnli/zh    (Chinese)
#
# By default uses all 8 GPUs, but this can be overridden with --world_size.
# The 37 trilingual models are distributed across GPUs via rank::world_size
# slicing of ALL_TRILINGUAL_MODELS.
#
# Usage:
#   cd /path/to/beetle-analyze
#   bash eval/launch_trilingual.sh                   # 8 GPUs, full checkpoint history
#   bash eval/launch_trilingual.sh --latest_only     # highest step-N ckpt only
#   bash eval/launch_trilingual.sh --no_push         # skip git push (debug / dry run)
#   bash eval/launch_trilingual.sh --world_size 4    # use 4 GPUs
#   bash eval/launch_trilingual.sh --mode slurm      # submit via SLURM
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="${MODE:-local}"
N_GPUS=8
BATCH_SIZE=64
NO_PUSH_FLAG=""
LATEST_ONLY_FLAG=""

REPO_ROOT="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/results/logs_trilingual"
HF_TOKEN="${HF_TOKEN:-}"

# ── Parse flags ───────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --mode=*)       MODE="${arg#*=}"        ;;
    --mode)         shift; MODE="$1"        ;;
    --no_push)      NO_PUSH_FLAG="--no_push"    ;;
    --latest_only)  LATEST_ONLY_FLAG="--latest_only" ;;
    --world_size=*) N_GPUS="${arg#*=}"      ;;
    --world_size)   shift; N_GPUS="$1"      ;;
  esac
done

mkdir -p "${REPO_ROOT}/results" "${LOG_DIR}"

# ── Git sanity check ──────────────────────────────────────────────────────────
echo "Checking git remote …"
if ! git -C "${REPO_ROOT}" remote get-url origin &>/dev/null; then
  echo "ERROR: no git remote 'origin' found in ${REPO_ROOT}."
  echo "       Run: git remote add origin git@github.com:suchirsalhan/beetle-analyze.git"
  exit 1
fi
REMOTE_URL="$(git -C "${REPO_ROOT}" remote get-url origin)"
echo "  Remote : ${REMOTE_URL}"

if [[ -z "${NO_PUSH_FLAG}" ]]; then
  echo "  Pulling latest from origin/main …"
  git -C "${REPO_ROOT}" pull --rebase origin main 2>/dev/null || true
fi
echo ""

# ── Shared HF dataset cache ───────────────────────────────────────────────────
HF_DATASETS_CACHE="${REPO_ROOT}/results/.hf_cache"
export HF_DATASETS_CACHE
mkdir -p "${HF_DATASETS_CACHE}"

# ── Pre-fetch all_configs datasets → .pkl ─────────────────────────────────────
# The trilingual models are evaluated on blimp_eng, blimp_nl, and zhoblimp —
# all three are all_configs datasets that need pre-fetching.
PKL_DIR="${REPO_ROOT}/results/.pkl_cache"
mkdir -p "${PKL_DIR}"

echo "Pre-fetching all_configs datasets → ${PKL_DIR} …"
PKL_DIR="${PKL_DIR}" python3 "${SCRIPT_DIR}/prefetch_datasets.py"
echo ""

echo "================================================================"
echo "  BeetleLM TRILINGUAL evaluation"
echo "  Mode        : ${MODE}"
echo "  GPUs        : ${N_GPUS}"
echo "  Repo root   : ${REPO_ROOT}"
echo "  Results     : ${REPO_ROOT}/results/"
echo "  Logs        : ${LOG_DIR}/"
echo "  Benchmarks  : blimp_eng · blimp_nl · zhoblimp · xcomps/zho · xnli/en · xnli/zh"
[[ -n "${LATEST_ONLY_FLAG}" ]] && \
echo "  Checkpoints : latest step-N only"
echo "================================================================"
echo ""

# =============================================================================
# LOCAL MODE
# =============================================================================
if [[ "${MODE}" == "local" ]]; then

  PIDS=()

  for (( rank=0; rank<N_GPUS; rank++ )); do
    log="${LOG_DIR}/rank${rank}.log"

    CMD=(
      python3 "${SCRIPT_DIR}/eval_model.py"
        --rank           "${rank}"
        --world_size     "${N_GPUS}"
        --output_dir     "${REPO_ROOT}"
        --batch_size     "${BATCH_SIZE}"
        --resume
        --trilingual_only          # ← key flag: restricts model pool to trilingual repos
    )
    [[ -n "${NO_PUSH_FLAG}"     ]] && CMD+=(--no_push)
    [[ -n "${LATEST_ONLY_FLAG}" ]] && CMD+=(--latest_only)
    [[ -n "${HF_TOKEN}"         ]] && CMD+=(--hf_token "${HF_TOKEN}")

    echo "  Launching rank=${rank} → ${log}"

    CUDA_VISIBLE_DEVICES="${rank}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
      "${CMD[@]}" > "${log}" 2>&1 &

    PIDS+=($!)
    (( rank < N_GPUS - 1 )) && sleep 4
  done

  echo ""
  echo "All ${#PIDS[@]} processes running."
  echo ""
  echo "  Monitor live:   tail -f ${LOG_DIR}/rank*.log"
  echo "  Count results:  watch -n 10 'wc -l ${REPO_ROOT}/results/*.csv 2>/dev/null'"
  echo ""

  FAIL=0
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      echo "WARNING: PID ${pid} exited non-zero"
      FAIL=1
    fi
  done

  if [[ "${FAIL}" -eq 0 ]]; then
    echo "All trilingual evaluations completed successfully."
  else
    echo "Some processes failed. Check logs in ${LOG_DIR}/"
    exit 1
  fi

# =============================================================================
# SLURM MODE
# =============================================================================
elif [[ "${MODE}" == "slurm" ]]; then

  jobscript="${LOG_DIR}/job_beetlelm_trilingual.sh"

  cat > "${jobscript}" << SLURM
#!/usr/bin/env bash
#SBATCH --job-name=beetlelm_tri
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_GPUS}
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=${LOG_DIR}/slurm_%j_%t.log
#SBATCH --error=${LOG_DIR}/slurm_%j_%t.err

module load cuda/12.1
source activate beetlelm    # ← update to match your environment

export HF_TOKEN="${HF_TOKEN}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"

srun --ntasks=${N_GPUS} --ntasks-per-node=${N_GPUS} bash -c "
  rank=\${SLURM_LOCALID}
  sleep \$((rank * 4))
  CUDA_VISIBLE_DEVICES=\${rank} \\
  HF_DATASETS_CACHE='${HF_DATASETS_CACHE}' \\
  python3 '${SCRIPT_DIR}/eval_model.py' \\
    --rank \${rank} \\
    --world_size ${N_GPUS} \\
    --output_dir '${REPO_ROOT}' \\
    --batch_size ${BATCH_SIZE} \\
    --resume \\
    --trilingual_only \\
    ${NO_PUSH_FLAG} \\
    ${LATEST_ONLY_FLAG} \\
    \$([ -n '${HF_TOKEN}' ] && echo '--hf_token ${HF_TOKEN}')
"
SLURM

  job_id=$(sbatch --parsable "${jobscript}")
  echo "Submitted → job ${job_id}"
  echo "Monitor:  squeue -u \$USER"
  echo "Logs:     ${LOG_DIR}/slurm_${job_id}_*.log"

else
  echo "Unknown MODE='${MODE}'. Use: local | slurm"
  exit 1
fi
