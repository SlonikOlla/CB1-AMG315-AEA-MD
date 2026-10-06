#!/bin/bash
#SBATCH --job-name=AMG315_BCC
#SBATCH --account=def-vasilisa
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/AMG315_BCC_%j.out
#SBATCH --error=slurm_logs/AMG315_BCC_%j.err

set -euo pipefail

cd ~/cannabinoid_receptor_modeling

ENVROOT="$PWD/envs/openff_mamba"
export PATH="$ENVROOT/bin:$PATH"
export AMBERHOME="$ENVROOT"

"$ENVROOT/bin/python" \
    scripts/parameterize_AMG315_AM1BCC.py |
    tee parameters/AMG315/AMG315_AM1BCC_report.txt
