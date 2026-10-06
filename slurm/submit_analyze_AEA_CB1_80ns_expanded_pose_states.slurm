#!/bin/bash
#SBATCH --job-name=AEA_80n_STATE
#SBATCH --account=def-vasilisa
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/AEA_80n_STATE_%j.out
#SBATCH --error=slurm_logs/AEA_80n_STATE_%j.err

set -euo pipefail

cd ~/cannabinoid_receptor_modeling

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"$PWD/envs/mdtraj_analysis/bin/python" \
    scripts/analyze_AEA_CB1_80ns_expanded_pose_states.py
