#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --mail-user=<zhengbang.zhou@mail.utoronto.ca>
#SBATCH --mail-type=ALL
#SBATCH --mem=48000M
#SBATCH --account=def-ybkim
#
# Generic per-Jpm compute array job for bfg_compute.py
#
# Usage:
#   cd /scratch/zhouzb79/bfg_pipeline
#   sbatch --export=CLUSTER=3x3_to --array=0-39 --job-name=comp_3x3to submit_compute.sh
#   sbatch --export=CLUSTER=3x3    --array=0-39 --job-name=comp_3x3   submit_compute.sh
#   sbatch --export=CLUSTER=2x3    --array=0-29 --job-name=comp_2x3   submit_compute.sh

CLUSTER="${CLUSTER:?ERROR: set CLUSTER via --export=CLUSTER=<2x3|3x3|3x3_to>}"

module --force purge
module load StdEnv/2023 python scipy-stack hdf5

source ~/pystandard/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

echo "=== bfg_compute.py --cluster ${CLUSTER} --worker --index ${SLURM_ARRAY_TASK_ID} ==="
python bfg_compute.py \
  --cluster "${CLUSTER}" \
  --worker \
  --index "${SLURM_ARRAY_TASK_ID}"

echo "=== Compute done for task ${SLURM_ARRAY_TASK_ID} ==="
