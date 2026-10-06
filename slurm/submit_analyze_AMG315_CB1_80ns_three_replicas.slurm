#!/bin/bash
#SBATCH --job-name=AMG315_80n_QC
#SBATCH --account=def-vasilisa_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/AMG315_80n_QC_%j.out
#SBATCH --error=slurm_logs/AMG315_80n_QC_%j.err

set -euo pipefail

cd ~/cannabinoid_receptor_modeling

"$PWD/envs/mdtraj_analysis/bin/python" \
  scripts/analyze_AMG315_CB1_80ns_three_replicas.py
