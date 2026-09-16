#!/bin/bash
# Week-3 pilot and baselines on the BISITE HPC (validation week only). Submit from the project folder:
#   sbatch slurm/pilot.sbatch                    # default student
#   sbatch slurm/pilot.sbatch --student-width 16 # if the gate fails
#SBATCH --job-name=kd-pilot
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source ~/venvs/kd-traffic/bin/activate
export KD_DATA_ROOT=${KD_DATA_ROOT:-$HOME/datasets}
nvidia-smi
python scripts/05_pilot.py --size S --workers 8 "$@"
python scripts/04_baselines.py --size S --workers 8
echo "Gate:"
cat "$(ls -td results/pilot/*/ | grep -v _smoke | head -1)gate.json"
