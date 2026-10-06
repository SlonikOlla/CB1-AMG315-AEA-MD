#!/bin/bash
#SBATCH --job-name=AEA_80n_FOCUS
#SBATCH --account=def-vasilisa
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/AEA_80n_FOCUS_%j.out
#SBATCH --error=slurm_logs/AEA_80n_FOCUS_%j.err

set -euo pipefail
cd ~/cannabinoid_receptor_modeling
"$PWD/envs/mdtraj_analysis/bin/python" scripts/compare_AEA_CB1_80ns_focused_pose_states.py
