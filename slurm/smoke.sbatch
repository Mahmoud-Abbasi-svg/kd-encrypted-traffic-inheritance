#!/bin/bash
# Pipeline check on the BISITE HPC (tiny sizes; numbers are meaningless). Submit from the project folder:
#   sbatch slurm/smoke.sbatch
#SBATCH --job-name=kd-smoke
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source ~/venvs/kd-traffic/bin/activate
export KD_DATA_ROOT=${KD_DATA_ROOT:-$HOME/datasets}
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests -q
python scripts/05_pilot.py --smoke --size S
python scripts/04_baselines.py --smoke --size S
