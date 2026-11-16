#!/bin/bash
#SBATCH --account=nn8104k
#SBATCH --job-name=debug
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=logs/debug-%j.out
#SBATCH --error=logs/debug-%j.err

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

python scripts/train_cfo.py \
    --dataset nordic \
    --model UNet2D \
    --dataset-path data/nordic/nordic_seas.h5 \
    --epochs 5000\
    --batch-size 16 \
    --lr 1e-5 \
    --eval-interval 500 \
    --ckpt-prefix debug_run \
    --no-eval


echo "===== TRAINING FINISHED ====="
date