#!/usr/bin/env bash
# =============================================================================
# launch_all.sh — Fan out all BeetleLM evaluations across 8 A100s
#
# Two modes (uncomment the one that matches your cluster setup):
#   MODE=slurm    — submit via sbatch (each benchmark is one job, 8 tasks/GPUs)
#   MODE=local    — bare-metal: background processes, one per GPU
#
# Usage:
#   bash launch_all.sh
#   bash launch_all.sh --mode local
#   bash launch_all.sh --mode slurm
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="${MODE:-local}"               # override with --mode flag or env var
N_GPUS=8
BATCH_SIZE=64                       # safe for A100 80 GB; bump to 128 if needed
OUTPUT_DIR="$(pwd)/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"
HF_TOKEN="${HF_TOKEN:-}"            # export HF_TOKEN=hf_xxx before running

# Benchmarks and their eval scripts
declare -A BENCHMARKS
BENCHMARKS["multiblimp"]="eval_minimal_pairs.py --benchmark multiblimp"
BENCHMARKS["zhoblimp"]="eval_minimal_pairs.py --benchmark zhoblimp"
BENCHMARKS["blimp_nl"]="eval_minimal_pairs.py --benchmark blimp_nl"
BENCHMARKS["xcomps"]="eval_minimal_pairs.py --benchmark xcomps"
BENCHMARKS["xnli"]="eval_xnli.py"

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --mode=*) MODE="${arg#*=}" ;;
    --mode)   shift; MODE="$1" ;;
  esac
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "=================================================="
echo "  BeetleLM Evaluation Launch"
echo "  Mode       : ${MODE}"
echo "  GPUs       : ${N_GPUS}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "=================================================="

# =============================================================================
# LOCAL MODE — spawn one background process per GPU × benchmark
# =============================================================================
if [[ "${MODE}" == "local" ]]; then

  PIDS=()
  for benchmark in "${!BENCHMARKS[@]}"; do
    SCRIPT_ARGS="${BENCHMARKS[$benchmark]}"

    for (( rank=0; rank<N_GPUS; rank++ )); do
      gpu=$rank
      log="${LOG_DIR}/${benchmark}_gpu${gpu}.log"

      # CUDA_VISIBLE_DEVICES=${gpu} already restricts this process to one GPU,
      # which CUDA renumbers as cuda:0. Always pass --gpu 0 here.
      cmd="python ${SCRIPT_DIR}/${SCRIPT_ARGS} \
          --gpu 0 \
          --rank ${rank} \
          --world_size ${N_GPUS} \
          --output_dir ${OUTPUT_DIR} \
          --batch_size ${BATCH_SIZE} \
          --resume"

      if [[ -n "${HF_TOKEN}" ]]; then
        cmd="${cmd} --hf_token ${HF_TOKEN}"
      fi

      echo "  Launching [${benchmark}] rank=${rank} gpu=${gpu} → ${log}"
      CUDA_VISIBLE_DEVICES=${gpu} ${cmd} > "${log}" 2>&1 &
      PIDS+=($!)
    done
  done

  echo ""
  echo "All ${#PIDS[@]} processes launched. Waiting …"
  echo "(tail -f ${LOG_DIR}/*.log  to monitor)"
  echo ""

  # Wait for all and report failures
  FAIL=0
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      echo "WARNING: process ${pid} exited with non-zero status"
      FAIL=1
    fi
  done

  if [[ $FAIL -eq 0 ]]; then
    echo "All evaluations completed successfully."
  else
    echo "Some evaluations failed — check logs in ${LOG_DIR}/"
    exit 1
  fi

# =============================================================================
# SLURM MODE — one sbatch job per benchmark, 8 GPU tasks per job
# =============================================================================
elif [[ "${MODE}" == "slurm" ]]; then

  for benchmark in "${!BENCHMARKS[@]}"; do
    SCRIPT_ARGS="${BENCHMARKS[$benchmark]}"
    jobscript="${LOG_DIR}/job_${benchmark}.sh"

    cat > "${jobscript}" <<SLURM
#!/usr/bin/env bash
#SBATCH --job-name=beetle_${benchmark}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_GPUS}
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=${LOG_DIR}/${benchmark}_%j_%t.log
#SBATCH --error=${LOG_DIR}/${benchmark}_%j_%t.err

module load cuda/12.1
source activate beetlelm   # replace with your conda/venv activation

export HF_TOKEN="${HF_TOKEN}"

srun --ntasks=${N_GPUS} --ntasks-per-node=${N_GPUS} \\
  bash -c "
    rank=\${SLURM_LOCALID}
    CUDA_VISIBLE_DEVICES=\${rank} python ${SCRIPT_DIR}/${SCRIPT_ARGS} \\
        --gpu 0 \\
        --rank \${rank} \\
        --world_size ${N_GPUS} \\
        --output_dir ${OUTPUT_DIR} \\
        --batch_size ${BATCH_SIZE} \\
        --resume \\
        \$([ -n '${HF_TOKEN}' ] && echo '--hf_token ${HF_TOKEN}')
  "
SLURM

    job_id=$(sbatch --parsable "${jobscript}")
    echo "  Submitted [${benchmark}] → job ${job_id}"
  done

  echo ""
  echo "All jobs submitted. Monitor with: squeue -u \$USER"

else
  echo "Unknown MODE='${MODE}'. Use --mode local  or  --mode slurm"
  exit 1
fi
