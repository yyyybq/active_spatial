#!/bin/bash
# =============================================================
# Evaluate a frozen (untrained) VLM as zero-shot baseline
# Requires vLLM + GPU
# =============================================================
set -e

cd /scratch/by2593/project/Active_Spatial/VAGEN

echo "============================================"
echo "  Frozen VLM Zero-Shot Evaluation"
echo "============================================"

python evaluation/run_eval.py \
    --config evaluation/configs/eval_frozen_vlm.yaml

echo "Done! Results in evaluation/outputs/eval_frozen/"
