#!/bin/bash
#SBATCH --job-name=appendix_ladder
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/scratch/users/ghoshavr/eoss/results/APPENDIX_DIRECTIONAL_LADDERS_20260906/logs/%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DATASETS=/scratch/users/ghoshavr/datasets
export RESULTS=/scratch/users/ghoshavr/eoss/results
export PYTHONPATH=/scratch/users/ghoshavr/eoss/marc_files/stability_regimes:/scratch/users/ghoshavr/eoss
srun /usr/local/linux/miniforge-3.13/bin/python3 -u /scratch/users/ghoshavr/eoss/marc_files/stability_regimes/moment_gpu_guard.py --bind-allocated-uuid --minimum-free-gb 12 --report /scratch/users/ghoshavr/eoss/results/APPENDIX_DIRECTIONAL_LADDERS_20260906/logs/gpu_${SLURM_ARRAY_TASK_ID}.json -- /usr/local/linux/miniforge-3.13/bin/python3 -u /scratch/users/ghoshavr/eoss/results/APPENDIX_DIRECTIONAL_LADDERS_20260906/source/runner.py "$SLURM_ARRAY_TASK_ID"
