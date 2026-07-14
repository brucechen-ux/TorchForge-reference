#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH=.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
mkdir -p logs/rerun_adamw logs/rerun_v4_muon_hybrid logs/rerun_muon_standard_ns
python scripts/check_muon_config_diff.py
torchrun --standalone --nproc_per_node=8 -m src.train --config configs/rerun_adamw.yaml --log_dir logs/rerun_adamw
torchrun --standalone --nproc_per_node=8 -m src.train --config configs/rerun_v4_muon_hybrid.yaml --log_dir logs/rerun_v4_muon_hybrid
torchrun --standalone --nproc_per_node=8 -m src.train --config configs/rerun_muon_standard_ns.yaml --log_dir logs/rerun_muon_standard_ns
