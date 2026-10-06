#!/bin/bash
#SBATCH --job-name=AEA_80n_STR
#SBATCH --account=def-vasilisa_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/AEA_80n_STR_%j.out
#SBATCH --error=slurm_logs/AEA_80n_STR_%j.err

set -euo pipefail
cd ~/cannabinoid_receptor_modeling

envs/mdtraj_analysis/bin/python \
    scripts/compare_AEA_CB1_80ns_focused_structures.py
