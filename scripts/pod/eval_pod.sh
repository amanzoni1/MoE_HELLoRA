#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TASK="gsm8k"
MODEL="allenai/OLMoE-1B-7B-0924"
MODEL_TAG="olmoe"
MODEL_SEEDS_CSV="42"
KS_CSV="4,8,12,14,16,18,20,22,24,32,48"
EVAL_SEED="123"
BACKEND="vllm"
MERGE_DIR="/workspace/merged_models"
OUTPUT_DIR="/workspace/eval_results"
MAX_NEW_TOKENS="512"
EVAL_N=""
RUN_HOT=1
RUN_FULL=1
CLEANUP_MERGE=1
USE_WANDB=1
PIP_INSTALL=1
ADAPTER_TEMPLATE=""
FULL_ADAPTER_TEMPLATE=""
EXTRA_EVAL_ARGS=()
SPIDER_OFFICIAL=0
TS_EVAL_REPO="/workspace/test-suite-sql-eval"
TS_DB_DIR=""
TS_TABLE=""
TS_ETYPE="exec"
TS_TIMEOUT_SEC="0"
TS_PLUG_VALUE=0
TS_KEEP_DISTINCT=0
TS_PROGRESS_BAR=0

usage() {
  cat <<EOF
Usage: bash scripts/pod/eval_pod.sh [options] [-- extra args for scripts.eval]

Options:
  --task <name>                 Task key (default: ${TASK})
  --model <hf_model>            Base model id (default: ${MODEL})
  --model-tag <tag>             Adapter naming tag (default: ${MODEL_TAG})
  --model-seeds <csv>           Adapter seeds as comma-list (default: ${MODEL_SEEDS_CSV})
  --ks <csv>                    Hot-k values as comma-list (default: ${KS_CSV})
  --adapter-template <tmpl>     Hot template with {seed} {k} (default: AManzoni/<tag>_<task>_s{seed}_hotk{k})
  --full-adapter-template <t>   Full template with {seed} (default: AManzoni/<tag>_<task>_s{seed}_full_lora)
  --eval-seed <int>             Eval reproducibility seed (default: ${EVAL_SEED})
  --backend <hf|vllm>           Inference backend (default: ${BACKEND})
  --merge-dir <path>            Merge cache dir (default: ${MERGE_DIR})
  --output-dir <path>           Eval output dir (default: ${OUTPUT_DIR})
  --max-new-tokens <int>        Generation length (default: ${MAX_NEW_TOKENS})
  --n <int>                     Optional eval subset size
  --hot-only                    Skip full-LoRA eval
  --full-only                   Skip hot sweep eval
  --no-cleanup-merge            Keep merged model dirs
  --no-wandb                    Disable W&B logging
  --no-pip-install              Skip dependency install
  --spider-official             Enable Spider official test-suite eval integration
  --ts-eval-repo <path>         test-suite-sql-eval repo path (default: ${TS_EVAL_REPO})
  --ts-db-dir <path>            Optional database dir override
  --ts-table <path>             Optional tables.json path override
  --ts-etype <exec|match|all>   Official eval mode (default: ${TS_ETYPE})
  --ts-timeout-sec <int>        Timeout passed to scripts.eval Spider wrapper (default: ${TS_TIMEOUT_SEC})
  --ts-plug-value               Pass --plug_value to official evaluator
  --ts-keep-distinct            Pass --keep_distinct to official evaluator
  --ts-progress-bar             Pass --progress_bar_for_each_datapoint to official evaluator
  -h, --help                    Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --model-seeds) MODEL_SEEDS_CSV="$2"; shift 2 ;;
    --ks) KS_CSV="$2"; shift 2 ;;
    --adapter-template) ADAPTER_TEMPLATE="$2"; shift 2 ;;
    --full-adapter-template) FULL_ADAPTER_TEMPLATE="$2"; shift 2 ;;
    --eval-seed) EVAL_SEED="$2"; shift 2 ;;
    --seed) EVAL_SEED="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --merge-dir) MERGE_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --n) EVAL_N="$2"; shift 2 ;;
    --hot-only) RUN_FULL=0; shift ;;
    --full-only) RUN_HOT=0; shift ;;
    --no-cleanup-merge) CLEANUP_MERGE=0; shift ;;
    --no-wandb) USE_WANDB=0; shift ;;
    --no-pip-install) PIP_INSTALL=0; shift ;;
    --spider-official) SPIDER_OFFICIAL=1; shift ;;
    --ts-eval-repo) TS_EVAL_REPO="$2"; shift 2 ;;
    --ts-db-dir) TS_DB_DIR="$2"; shift 2 ;;
    --ts-table) TS_TABLE="$2"; shift 2 ;;
    --ts-etype) TS_ETYPE="$2"; shift 2 ;;
    --ts-timeout-sec) TS_TIMEOUT_SEC="$2"; shift 2 ;;
    --ts-plug-value) TS_PLUG_VALUE=1; shift ;;
    --ts-keep-distinct) TS_KEEP_DISTINCT=1; shift ;;
    --ts-progress-bar) TS_PROGRESS_BAR=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_EVAL_ARGS=("$@"); break ;;
    *) die "Unknown argument: $1" ;;
  esac
done

cd "$REPO_ROOT"

if [[ "$PIP_INSTALL" -eq 1 ]]; then
  log "Installing eval dependencies..."
  install_eval_deps
fi

if [[ "$SPIDER_OFFICIAL" -eq 1 ]]; then
  if [[ "$TASK" != "spider" ]]; then
    die "--spider-official can only be used with --task spider"
  fi
  if [[ "$PIP_INSTALL" -eq 1 ]]; then
    python3 -m pip install --no-cache-dir -U sqlparse nltk
    python3 - <<'PY'
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
print("NLTK punkt assets ready.")
PY
  else
    log "Skipping Spider official deps install (--no-pip-install). Ensure sqlparse/nltk + punkt are already installed."
  fi
fi

setup_cache_dirs
if ! hf_login_if_token; then
  log "HF_TOKEN not set; using unauthenticated HF access."
fi

if [[ "$USE_WANDB" -eq 1 ]]; then
  if ! wandb_login_if_key; then
    log "WANDB_API_KEY missing; disabling W&B for this run."
    USE_WANDB=0
  fi
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$MERGE_DIR"

if [[ -z "$ADAPTER_TEMPLATE" ]]; then
  ADAPTER_TEMPLATE="AManzoni/${MODEL_TAG}_${TASK}_s{seed}_hotk{k}"
fi
if [[ -z "$FULL_ADAPTER_TEMPLATE" ]]; then
  FULL_ADAPTER_TEMPLATE="AManzoni/${MODEL_TAG}_${TASK}_s{seed}_full_lora"
fi

if [[ "$SPIDER_OFFICIAL" -eq 1 ]]; then
  EXTRA_EVAL_ARGS+=(--ts_run_official --ts_eval_repo "$TS_EVAL_REPO" --ts_etype "$TS_ETYPE" --ts_timeout_sec "$TS_TIMEOUT_SEC")
  if [[ -n "$TS_DB_DIR" ]]; then
    EXTRA_EVAL_ARGS+=(--ts_db_dir "$TS_DB_DIR")
  fi
  if [[ -n "$TS_TABLE" ]]; then
    EXTRA_EVAL_ARGS+=(--ts_table "$TS_TABLE")
  fi
  if [[ "$TS_PLUG_VALUE" -eq 1 ]]; then
    EXTRA_EVAL_ARGS+=(--ts_plug_value)
  fi
  if [[ "$TS_KEEP_DISTINCT" -eq 1 ]]; then
    EXTRA_EVAL_ARGS+=(--ts_keep_distinct)
  fi
  if [[ "$TS_PROGRESS_BAR" -eq 1 ]]; then
    EXTRA_EVAL_ARGS+=(--ts_progress_bar)
  fi
fi

split_csv "$MODEL_SEEDS_CSV" MODEL_SEEDS
split_csv "$KS_CSV" KS

common_args=(
  python3 -m scripts.eval
  --task "$TASK"
  --model "$MODEL"
  --seed "$EVAL_SEED"
  --backend "$BACKEND"
  --merge_dir "$MERGE_DIR"
  --output_dir "$OUTPUT_DIR"
  --max_new_tokens "$MAX_NEW_TOKENS"
)
if [[ "$CLEANUP_MERGE" -eq 1 ]]; then
  common_args+=(--cleanup_merge)
fi
if [[ "$USE_WANDB" -eq 1 ]]; then
  common_args+=(--wandb)
fi
if [[ -n "$EVAL_N" ]]; then
  common_args+=(--n "$EVAL_N")
fi

if [[ "$RUN_HOT" -eq 1 ]]; then
  cmd=("${common_args[@]}"
    --adapter_template "$ADAPTER_TEMPLATE"
    --model_seeds "$MODEL_SEEDS_CSV"
    --ks "$KS_CSV"
  )
  cmd+=("${EXTRA_EVAL_ARGS[@]}")
  log "Launching HOT eval sweep..."
  "${cmd[@]}"
fi

if [[ "$RUN_FULL" -eq 1 ]]; then
  for seed in "${MODEL_SEEDS[@]}"; do
    adapter="${FULL_ADAPTER_TEMPLATE//\{seed\}/$seed}"
    run_name="${MODEL_TAG}_${TASK}_s${seed}_full_lora"
    cmd=("${common_args[@]}"
      --adapter "$adapter"
      --run_name "$run_name"
    )
    cmd+=("${EXTRA_EVAL_ARGS[@]}")
    log "Launching FULL eval: seed=${seed}"
    "${cmd[@]}"
  done
fi

log "Eval launcher completed."
