#!/bin/bash
#SBATCH --job-name=AMG315_80n_FOC
#SBATCH --account=def-vasilisa
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/AMG315_80n_FOC_%j.out
#SBATCH --error=slurm_logs/AMG315_80n_FOC_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

envs/mdtraj_analysis/bin/python \
    scripts/compare_AMG315_CB1_80ns_focused_candidate_ensembles.py
