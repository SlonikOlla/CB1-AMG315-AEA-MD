#!/bin/bash
#SBATCH --job-name=AMG_AEA_80n
#SBATCH --account=def-vasilisa
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/AMG_AEA_80n_%j.out
#SBATCH --error=slurm_logs/AMG_AEA_80n_%j.err

set -euo pipefail

cd ~/cannabinoid_receptor_modeling

"$PWD/envs/mdtraj_analysis/bin/python" \
    scripts/compare_AMG315_AEA_CB1_80ns.py
