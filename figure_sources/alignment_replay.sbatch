#!/bin/bash
#SBATCH --job-name=overleaf_alignment
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/ghoshavr/eoss/results/OVERLEAF_FIGURES_20260906/logs/%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DATASETS=/scratch/users/ghoshavr/datasets
export RESULTS=/scratch/users/ghoshavr/eoss/results
export PYTHONPATH=/scratch/users/ghoshavr/eoss/marc_files/stability_regimes:/scratch/users/ghoshavr/eoss
srun /usr/local/linux/miniforge-3.13/bin/python3 -u /scratch/users/ghoshavr/eoss/marc_files/stability_regimes/moment_gpu_guard.py --bind-allocated-uuid --minimum-free-gb 8 --report /scratch/users/ghoshavr/eoss/results/OVERLEAF_FIGURES_20260906/logs/gpu.json -- /usr/local/linux/miniforge-3.13/bin/python3 -u /scratch/users/ghoshavr/eoss/results/OVERLEAF_FIGURES_20260906/source/replay_alignment.py --index 0
