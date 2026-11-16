#!/bin/bash
#SBATCH --job-name=h5py
#SBATCH --account=nn8104k
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/.out
#SBATCH --error=logs/splines-%j.err

set -e

module --force purge
module load NRIS/CPU

# OVERLAY=/cluster/work/projects/nn8104k/esther_thesis/containers/cfo_overlay.img
SIF=/cluster/work/projects/nn8104k/esther_thesis/containers/cfo.sif
PROJECT=/cluster/work/projects/nn8104k/esther_thesis/FNO-Nordic-Sea

cd "$PROJECT"

# Packages
apptainer exec  \
    "$SIF" \
    env PYTHONNOUSERSITE=1 \
    python -c "
import numpy
import xarray
import zarr


print('NumPy:', numpy.__version__)
print('xarray:', xarray.__version__)
print('Zarr:', zarr.__version__)
"

apptainer exec \
    $SIF \
    env PYTHONNOUSERSITE=1 \
    python read_splines.py