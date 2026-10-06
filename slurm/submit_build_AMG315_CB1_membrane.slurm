#!/bin/bash
#SBATCH --job-name=AMG315_build
#SBATCH --account=def-vasilisa
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/AMG315_build_%j.out
#SBATCH --error=slurm_logs/AMG315_build_%j.err

set -euo pipefail

cd ~/cannabinoid_receptor_modeling

ENVROOT="$PWD/envs/openff_mamba"
export PATH="$ENVROOT/bin:$PATH"
export AMBERHOME="$ENVROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"$ENVROOT/bin/python" \
    scripts/build_AMG315_CB1_membrane.py
