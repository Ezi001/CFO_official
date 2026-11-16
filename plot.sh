#!/bin/bash
#SBATCH --account=nn8104k
#SBATCH --job-name=plot
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=logs/plot-%j.out
#SBATCH --error=logs/plot-%j.err

#  10 * 100

set -euo pipefail

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# Modules first, then PATH, so the purge can't interfere
module --force purge
module load NRIS/GPU

export PATH="/cluster/work/projects/nn8104k/esther_thesis/envs/jax_env/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH



cd /cluster/work/projects/nn8104k/esther_thesis/CFO_official

echo "===== NVIDIA GPU ====="
nvidia-smi

echo "===== PYTHON CHECK ====="
python -c "
import platform, jax
print('JAX:', jax.__version__, jax.devices())
print('Architecture:', platform.machine())
"

echo "===== START CFO TRAINING ====="
date

python scripts/evaluate_cfo.py \
    --dataset nordic \
    --model UNet2D \
    --ckpt-dir checkpoints/best \
    --ckpt-prefix debug_run \
    --split test \
    --solver RK4 \
    --steps-per-segment 2 \
    --save-pred pred.npy

echo "===== TRAINING FINISHED ====="
date