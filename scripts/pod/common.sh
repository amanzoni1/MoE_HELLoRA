#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log() {
  printf '[pod] %s\n' "$*"
}

die() {
  printf '[pod][error] %s\n' "$*" >&2
  exit 1
}

split_csv() {
  local value="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<<"$value"
  local i
  for i in "${!out_ref[@]}"; do
    out_ref[$i]="$(printf '%s' "${out_ref[$i]}" | xargs)"
  done
}

setup_cache_dirs() {
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
  export WANDB_DIR="${WANDB_DIR:-/workspace/.cache/wandb}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/workspace/.cache/wandb}"
  mkdir -p "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR"
}

install_train_deps() {
  python3 -m pip install -U pip
  python3 -m pip install --no-cache-dir -U \
    "transformers==4.57.6" \
    "accelerate==1.12.0" \
    "peft==0.18.1" \
    datasets \
    wandb \
    huggingface_hub
}

install_eval_deps() {
  python3 -m pip install -U pip
  python3 -m pip install --no-cache-dir -U \
    transformers \
    datasets \
    peft \
    accelerate \
    bitsandbytes \
    wandb \
    vllm \
    huggingface_hub
}

hf_login_if_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    return 1
  fi
  python3 - <<'PY'
import os
from huggingface_hub import login, whoami
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit(1)
login(token=token)
print("HF login:", whoami()["name"])
PY
}

wandb_login_if_key() {
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    return 1
  fi
  python3 - <<'PY'
import os
import wandb
key = os.environ.get("WANDB_API_KEY")
if not key:
    raise SystemExit(1)
wandb.login(key=key, relogin=True)
print("W&B login: ok")
PY
}
