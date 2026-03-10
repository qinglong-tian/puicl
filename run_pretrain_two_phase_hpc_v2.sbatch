#!/bin/bash
# Add your own site-specific account and mail directives if needed.
#SBATCH --job-name=slim-pretrain-v2-2phase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=TERM@120

set -euo pipefail

INPUT_REPO_DIR="${REPO_DIR:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYOUT="${LAYOUT:-}"
LAUNCH_MODULE="${LAUNCH_MODULE:-}"
IMPORT_CHECK="${IMPORT_CHECK:-}"
PACKAGE_NAME="${PACKAGE_NAME:-}"

configure_standalone_layout() {
  local package_dir="$1"
  local package_name
  package_name="$(basename "$package_dir")"
  INPUT_REPO_DIR="$package_dir"
  LAYOUT="standalone"
  PACKAGE_NAME="$package_name"
  LAUNCH_MODULE="${package_name}.train.run_pretrain_hpc"
  IMPORT_CHECK="import ${package_name}; print('${package_name}', ${package_name}.__file__)"
}

configure_full_repo_layout() {
  local repo_root="$1"
  local package_dir="$repo_root/slim_pretrain"
  local package_name
  package_name="$(basename "$repo_root")"
  INPUT_REPO_DIR="$package_dir"
  LAYOUT="full_repo"
  PACKAGE_NAME="$package_name"
  LAUNCH_MODULE="slim_pretrain.${package_name}.train.run_pretrain_hpc"
  IMPORT_CHECK="import slim_pretrain.${package_name} as pkg; print('${package_name}', pkg.__file__)"
}

if [[ -z "$INPUT_REPO_DIR" ]]; then
  for CANDIDATE in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(dirname "$SCRIPT_DIR")"; do
    [[ -z "${CANDIDATE:-}" ]] && continue
    if [[ -f "$CANDIDATE/slim_pretrain/__init__.py" && -f "$CANDIDATE/__init__.py" ]]; then
      configure_full_repo_layout "$CANDIDATE"
      break
    fi
    if [[ -f "$CANDIDATE/__init__.py" && -f "$CANDIDATE/slim_pretrain/__init__.py" ]]; then
      configure_full_repo_layout "$CANDIDATE"
      break
    fi
    if [[ -f "$CANDIDATE/__init__.py" && -d "$CANDIDATE/train" && -d "$CANDIDATE/simplified_prior" && -d "$CANDIDATE/data" ]]; then
      configure_standalone_layout "$CANDIDATE"
      break
    fi
  done
fi

if [[ -n "$INPUT_REPO_DIR" && -z "$LAUNCH_MODULE" ]]; then
  if [[ -f "$INPUT_REPO_DIR/slim_pretrain/__init__.py" && -f "$INPUT_REPO_DIR/__init__.py" ]]; then
    configure_full_repo_layout "$INPUT_REPO_DIR"
  elif [[ -f "$INPUT_REPO_DIR/__init__.py" && -d "$INPUT_REPO_DIR/train" && -d "$INPUT_REPO_DIR/simplified_prior" && -d "$INPUT_REPO_DIR/data" ]]; then
    configure_standalone_layout "$INPUT_REPO_DIR"
  fi
fi

if [[ -z "$INPUT_REPO_DIR" ]]; then
  echo "Could not locate repository root for training run." >&2
  echo "Set REPO_DIR to either:" >&2
  echo "  1) a repo root containing slim_pretrain/<package>, or" >&2
  echo "  2) a standalone package folder containing train/, simplified_prior/, and data/." >&2
  exit 2
fi

REPO_DIR="$INPUT_REPO_DIR"
cd "$REPO_DIR"

mkdir -p logs artifacts

CURRIC_TOTAL_STAGES="${CURRIC_TOTAL_STAGES:-100}"
CURRIC_STEPS_PER_STAGE="${CURRIC_STEPS_PER_STAGE:-1000}"
CURRIC_TOTAL_STEPS="${CURRIC_TOTAL_STEPS:-$((CURRIC_TOTAL_STAGES * CURRIC_STEPS_PER_STAGE))}"

TAIL_STEPS="${TAIL_STEPS:-25000}"
TAIL_TOTAL_STEPS="${TAIL_TOTAL_STEPS:-$((CURRIC_TOTAL_STEPS + TAIL_STEPS))}"
TAIL_PHASE_START_STEP="${TAIL_PHASE_START_STEP:-$CURRIC_TOTAL_STEPS}"
TAIL_TOTAL_STAGES="${TAIL_TOTAL_STAGES:-1}"
TAIL_STEPS_PER_STAGE="${TAIL_STEPS_PER_STAGE:-$TAIL_STEPS}"

BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_FEATURES_MIN="${NUM_FEATURES_MIN:-5}"
NUM_FEATURES_MAX="${NUM_FEATURES_MAX:-24}"
POSITIVE_SIZE_MIN="${POSITIVE_SIZE_MIN:-100}"
POSITIVE_SIZE_MAX="${POSITIVE_SIZE_MAX:-300}"

NUM_LAYERS_MIN="${NUM_LAYERS_MIN:-4}"
NUM_LAYERS_MAX="${NUM_LAYERS_MAX:-12}"
HIDDEN_DIM_MIN="${HIDDEN_DIM_MIN:-12}"
HIDDEN_DIM_MAX="${HIDDEN_DIM_MAX:-36}"
NONLINEARITIES="${NONLINEARITIES:-tanh,relu,gelu,sine,identity,abs,square,sign,heaviside,rbf}"

BASE_LR="${BASE_LR:-1.6e-4}"
MIN_LR="${MIN_LR:-1.6e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-4000}"
LR_DECAY_POWER="${LR_DECAY_POWER:-1.5}"

TAIL_LR_SCALE="${TAIL_LR_SCALE:-0.25}"
if [[ -z "${TAIL_BASE_LR:-}" ]]; then
  TAIL_BASE_LR="$(awk "BEGIN {printf \"%.12g\", ${BASE_LR} * ${TAIL_LR_SCALE}}")"
fi
if [[ -z "${TAIL_MIN_LR:-}" ]]; then
  TAIL_MIN_LR="$(awk "BEGIN {printf \"%.12g\", ${MIN_LR} * ${TAIL_LR_SCALE}}")"
fi
TAIL_WARMUP_STEPS="${TAIL_WARMUP_STEPS:-2000}"
TAIL_LR_DECAY_POWER="${TAIL_LR_DECAY_POWER:-$LR_DECAY_POWER}"

UNLABELED_RATIO_START_LO="${UNLABELED_RATIO_START_LO:-1.0}"
UNLABELED_RATIO_START_HI="${UNLABELED_RATIO_START_HI:-1.0}"
UNLABELED_RATIO_END_LO="${UNLABELED_RATIO_END_LO:-0.5}"
UNLABELED_RATIO_END_HI="${UNLABELED_RATIO_END_HI:-2.5}"

TEST_OUTLIER_START_LO="${TEST_OUTLIER_START_LO:-0.5}"
TEST_OUTLIER_START_HI="${TEST_OUTLIER_START_HI:-0.5}"
TEST_OUTLIER_END_LO="${TEST_OUTLIER_END_LO:-0.1}"
TEST_OUTLIER_END_HI="${TEST_OUTLIER_END_HI:-0.9}"

EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_BATCHES="${EVAL_BATCHES:-16}"
LOG_EVERY="${LOG_EVERY:-200}"
SEED="${SEED:-0}"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  CHECKPOINT_DIR="artifacts/checkpoints"
fi
SAVE_EVERY="${SAVE_EVERY:-500}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
mkdir -p "$CHECKPOINT_DIR"

LATEST_CHECKPOINT="${CHECKPOINT_DIR}/latest.pt"
PHASE1_RESUME_FROM="${PHASE1_RESUME_FROM:-}"
PHASE1_INIT_FROM="${PHASE1_INIT_FROM:-}"
PHASE2_RESUME_FROM="${PHASE2_RESUME_FROM:-$LATEST_CHECKPOINT}"

if [[ -z "$PHASE1_RESUME_FROM" && -f "$LATEST_CHECKPOINT" ]]; then
  PHASE1_RESUME_FROM="$LATEST_CHECKPOINT"
fi

if [[ -z "${PHASE1_HISTORY_JSON:-}" ]]; then
  PHASE1_HISTORY_JSON="artifacts/history_phase1_${SLURM_JOB_ID:-local}.json"
fi
if [[ -z "${PHASE2_HISTORY_JSON:-}" ]]; then
  PHASE2_HISTORY_JSON="artifacts/history_phase2_${SLURM_JOB_ID:-local}.json"
fi

if [[ "$CURRIC_TOTAL_STEPS" -ne $((CURRIC_TOTAL_STAGES * CURRIC_STEPS_PER_STAGE)) ]]; then
  echo "CURRIC_TOTAL_STEPS must equal CURRIC_TOTAL_STAGES * CURRIC_STEPS_PER_STAGE." >&2
  exit 2
fi
if [[ "$TAIL_TOTAL_STEPS" -le "$TAIL_PHASE_START_STEP" ]]; then
  echo "TAIL_TOTAL_STEPS must be greater than TAIL_PHASE_START_STEP." >&2
  exit 2
fi
if [[ "$TAIL_PHASE_START_STEP" -ne "$CURRIC_TOTAL_STEPS" ]]; then
  echo "TAIL_PHASE_START_STEP must match CURRIC_TOTAL_STEPS for the default two-phase plan." >&2
  exit 2
fi
if [[ -n "$PHASE1_RESUME_FROM" && -n "$PHASE1_INIT_FROM" ]]; then
  echo "Both PHASE1_RESUME_FROM and PHASE1_INIT_FROM are set; resume takes precedence."
fi
if [[ -n "$PHASE1_RESUME_FROM" && ! -f "$PHASE1_RESUME_FROM" ]]; then
  echo "Requested PHASE1_RESUME_FROM checkpoint does not exist: $PHASE1_RESUME_FROM" >&2
  exit 2
fi
if [[ -z "$PHASE1_RESUME_FROM" && -n "$PHASE1_INIT_FROM" && ! -f "$PHASE1_INIT_FROM" ]]; then
  echo "Requested PHASE1_INIT_FROM checkpoint does not exist: $PHASE1_INIT_FROM" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

module --force purge
module load StdEnv/2023
source ~/venvs/icl/bin/activate

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$(dirname "$REPO_DIR"):${PYTHONPATH:-}"

NNODES="${SLURM_NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
OMP_THREADS_DEFAULT=$(( SLURM_CPUS_PER_TASK / NPROC_PER_NODE ))
if [[ "$OMP_THREADS_DEFAULT" -lt 1 ]]; then
  OMP_THREADS_DEFAULT=1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OMP_THREADS_DEFAULT}"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"
  else
    MASTER_PORT=29500
  fi
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "Could not resolve MASTER_ADDR from SLURM_JOB_NODELIST." >&2
  exit 2
fi

run_phase() {
  local phase_name="$1"
  shift
  local -a run_args=("$@")
  local run_args_str
  run_args_str="$(printf ' %q' "${run_args[@]}")"

  echo "Starting ${phase_name}"
  if [[ "$NNODES" -gt 1 ]]; then
    srun --ntasks="$NNODES" --ntasks-per-node=1 \
      bash -lc "torchrun \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_PROCID \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m $LAUNCH_MODULE${run_args_str}"
  else
    torchrun \
      --nproc_per_node="$NPROC_PER_NODE" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      -m "$LAUNCH_MODULE" \
      "${run_args[@]}"
  fi
}

echo "Using MASTER_PORT=${MASTER_PORT}"
echo "Using MASTER_ADDR=${MASTER_ADDR}"
echo "Using NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "Using OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "Layout=${LAYOUT}"
echo "Repo dir=${REPO_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "Checkpoint dir=${CHECKPOINT_DIR}"
echo "Phase 1: stages=${CURRIC_TOTAL_STAGES}, steps/stage=${CURRIC_STEPS_PER_STAGE}, total_steps=${CURRIC_TOTAL_STEPS}"
echo "Phase 2: fixed final stage for ${TAIL_STEPS} extra steps, total_steps=${TAIL_TOTAL_STEPS}, lr_scale=${TAIL_LR_SCALE}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "$IMPORT_CHECK"

PHASE1_ARGS=(
  --device cuda
  --seed "$SEED"
  --total-steps "$CURRIC_TOTAL_STEPS"
  --total-stages "$CURRIC_TOTAL_STAGES"
  --steps-per-stage "$CURRIC_STEPS_PER_STAGE"
  --batch-size "$BATCH_SIZE"
  --num-features-min "$NUM_FEATURES_MIN"
  --num-features-max "$NUM_FEATURES_MAX"
  --positive-size-min "$POSITIVE_SIZE_MIN"
  --positive-size-max "$POSITIVE_SIZE_MAX"
  --num-layers-min "$NUM_LAYERS_MIN"
  --num-layers-max "$NUM_LAYERS_MAX"
  --hidden-dim-min "$HIDDEN_DIM_MIN"
  --hidden-dim-max "$HIDDEN_DIM_MAX"
  --nonlinearities "$NONLINEARITIES"
  --base-lr "$BASE_LR"
  --min-lr "$MIN_LR"
  --warmup-steps "$WARMUP_STEPS"
  --lr-decay-power "$LR_DECAY_POWER"
  --unlabeled-ratio-start-lo "$UNLABELED_RATIO_START_LO"
  --unlabeled-ratio-start-hi "$UNLABELED_RATIO_START_HI"
  --unlabeled-ratio-end-lo "$UNLABELED_RATIO_END_LO"
  --unlabeled-ratio-end-hi "$UNLABELED_RATIO_END_HI"
  --test-outlier-start-lo "$TEST_OUTLIER_START_LO"
  --test-outlier-start-hi "$TEST_OUTLIER_START_HI"
  --test-outlier-end-lo "$TEST_OUTLIER_END_LO"
  --test-outlier-end-hi "$TEST_OUTLIER_END_HI"
  --log-every "$LOG_EVERY"
  --eval-every "$EVAL_EVERY"
  --eval-batches "$EVAL_BATCHES"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --save-every "$SAVE_EVERY"
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
  --history-json "$PHASE1_HISTORY_JSON"
  --phase-local-schedule
  --no-auto-resume
)

if [[ -n "$PHASE1_INIT_FROM" && -z "$PHASE1_RESUME_FROM" ]]; then
  PHASE1_ARGS+=(--init-from "$PHASE1_INIT_FROM")
fi
if [[ -n "$PHASE1_RESUME_FROM" ]]; then
  PHASE1_ARGS+=(--resume-from "$PHASE1_RESUME_FROM")
fi

run_phase "phase 1 curriculum" "${PHASE1_ARGS[@]}"

if [[ ! -f "$LATEST_CHECKPOINT" && ! -f "$PHASE2_RESUME_FROM" ]]; then
  echo "Phase 2 requires a checkpoint from phase 1, but none was found." >&2
  exit 2
fi
if [[ ! -f "$PHASE2_RESUME_FROM" ]]; then
  PHASE2_RESUME_FROM="$LATEST_CHECKPOINT"
fi

PHASE2_ARGS=(
  --device cuda
  --seed "$SEED"
  --total-steps "$TAIL_TOTAL_STEPS"
  --total-stages "$TAIL_TOTAL_STAGES"
  --steps-per-stage "$TAIL_STEPS_PER_STAGE"
  --batch-size "$BATCH_SIZE"
  --num-features-min "$NUM_FEATURES_MIN"
  --num-features-max "$NUM_FEATURES_MAX"
  --positive-size-min "$POSITIVE_SIZE_MIN"
  --positive-size-max "$POSITIVE_SIZE_MAX"
  --num-layers-min "$NUM_LAYERS_MIN"
  --num-layers-max "$NUM_LAYERS_MAX"
  --hidden-dim-min "$HIDDEN_DIM_MIN"
  --hidden-dim-max "$HIDDEN_DIM_MAX"
  --nonlinearities "$NONLINEARITIES"
  --base-lr "$TAIL_BASE_LR"
  --min-lr "$TAIL_MIN_LR"
  --warmup-steps "$TAIL_WARMUP_STEPS"
  --lr-decay-power "$TAIL_LR_DECAY_POWER"
  --unlabeled-ratio-start-lo "$UNLABELED_RATIO_END_LO"
  --unlabeled-ratio-start-hi "$UNLABELED_RATIO_END_HI"
  --unlabeled-ratio-end-lo "$UNLABELED_RATIO_END_LO"
  --unlabeled-ratio-end-hi "$UNLABELED_RATIO_END_HI"
  --test-outlier-start-lo "$TEST_OUTLIER_END_LO"
  --test-outlier-start-hi "$TEST_OUTLIER_END_HI"
  --test-outlier-end-lo "$TEST_OUTLIER_END_LO"
  --test-outlier-end-hi "$TEST_OUTLIER_END_HI"
  --log-every "$LOG_EVERY"
  --eval-every "$EVAL_EVERY"
  --eval-batches "$EVAL_BATCHES"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --save-every "$SAVE_EVERY"
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
  --history-json "$PHASE2_HISTORY_JSON"
  --phase-local-schedule
  --phase-start-step "$TAIL_PHASE_START_STEP"
  --resume-from "$PHASE2_RESUME_FROM"
  --no-auto-resume
)

run_phase "phase 2 final-stage tail" "${PHASE2_ARGS[@]}"
