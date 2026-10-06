#!/bin/bash
#SBATCH --job-name=AMG315_80n_EXP
#SBATCH --account=def-vasilisa_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/AMG315_80n_EXP_%j.out
#SBATCH --error=slurm_logs/AMG315_80n_EXP_%j.err

set -euo pipefail

cd ~/cannabinoid_receptor_modeling

envs/mdtraj_analysis/bin/python \
    scripts/analyze_AMG315_CB1_80ns_expanded_pose_states.py
