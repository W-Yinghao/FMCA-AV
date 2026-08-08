#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning"
BASE_PYTHON="/home/infres/yinwang/anaconda3/envs/icml/bin/python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-dir) ENV_DIR="$2"; shift 2 ;;
    --python) BASE_PYTHON="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

"$BASE_PYTHON" -c 'import torch; print("Reusing PyTorch", torch.__version__, "from", torch.__file__)'
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install -r environment/requirements.txt
"$ENV_DIR/bin/python" -c 'import lightning, torch; print("Lightning", lightning.__version__); print("PyTorch", torch.__version__, torch.__file__)'

