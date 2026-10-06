#!/bin/bash
#SBATCH --job-name=AMG_AEA_MICRO
#SBATCH --account=def-vasilisa
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/AMG_AEA_MICRO_%j.out
#SBATCH --error=slurm_logs/AMG_AEA_MICRO_%j.err

set -euo pipefail

cd ~/cannabinoid_receptor_modeling

"$PWD/envs/mdtraj_analysis/bin/python" \
    scripts/analyze_AMG315_AEA_CB1_80ns_microswitches.py
