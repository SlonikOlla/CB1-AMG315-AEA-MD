#!/bin/bash
#SBATCH --job-name=AMG315_80n_CAL
#SBATCH --account=def-vasilisa_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/AMG315_80n_CAL_%j.out
#SBATCH --error=slurm_logs/AMG315_80n_CAL_%j.err

set -euo pipefail

cd ~/cannabinoid_receptor_modeling

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.11

"$PWD/envs/mdtraj_analysis/bin/python" \
  scripts/calibrate_AMG315_CB1_80ns_frozen_pose_distances.py
