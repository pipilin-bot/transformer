#!/usr/bin/env bash
# Reproduce the baseline Transformer training run with fixed hyperparameters.
# Usage: ./scripts/run.sh [seed]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

SEED="${1:-42}"
export TRANSFORMER_ROOT="${ROOT_DIR}"
export TRANSFORMER_SEED="${SEED}"

python - <<'PY'
import os
import random
import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch 未安装，请先运行 pip install -r requirements.txt") from exc

seed = int(os.environ.get("TRANSFORMER_SEED", "42"))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

root = os.environ["TRANSFORMER_ROOT"]

import sys
sys.argv = [
    "train.py",
    "--d_model", "128",
    "--num_layers", "2",
    "--num_heads", "4",
    "--d_ff", "512",
    "--max_len", "128",
    "--batch_size", "64",
    "--lr", "3e-4",
    "--num_epochs", "20",
    "--dropout", "0.1",
    "--lr_scheduler", "cosine",
    "--warmup_steps", "2000",
    "--tokenizer", os.path.join(root, "opus_mt_en_de_tokenizer"),
    "--data_dir", os.path.join(root, "IWSLT2017"),
    "--save_dir", os.path.join(root, "results", "fast_run"),
]

from src import train
train.main()
PY

