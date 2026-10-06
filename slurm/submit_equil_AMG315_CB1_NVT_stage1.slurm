#!/bin/bash
#SBATCH --job-name=AMG315_NVT1
#SBATCH --account=def-vasilisa
#SBATCH --gres=gpu:h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/AMG315_NVT1_%j.out
#SBATCH --error=slurm_logs/AMG315_NVT1_%j.err

set -euo pipefail
cd ~/cannabinoid_receptor_modeling

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load python/3.11
module load openmm/8.5.2

python scripts/equil_AMG315_CB1_NVT_stage1.py
