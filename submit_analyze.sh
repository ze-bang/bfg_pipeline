#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mail-user=<zhengbang.zhou@mail.utoronto.ca>
#SBATCH --mail-type=ALL
#SBATCH --mem=128000M
#SBATCH --account=def-ybkim
#
# Generic analysis job for bfg_analyze.py
# (spectrum + structure factors + diagnostics — NO RDM, that's in bfg_compute.py)
#
# Usage:
#   cd /scratch/zhouzb79/bfg_pipeline
#
#   # With dependency on compute array job:
#   COMP_JOB=$(sbatch --export=CLUSTER=3x3_to --array=0-39 --job-name=comp_3x3to submit_compute.sh | awk '{print $4}')
#   sbatch --dependency=afterok:${COMP_JOB} --export=CLUSTER=3x3_to --job-name=ana_3x3to submit_analyze.sh
#
#   # Standalone (if compute is already done):
#   sbatch --export=CLUSTER=3x3_to --job-name=ana_3x3to submit_analyze.sh

CLUSTER="${CLUSTER:?ERROR: set CLUSTER via --export=CLUSTER=<2x3|3x3|3x3_to>}"

module --force purge
module load StdEnv/2023 python scipy-stack

source ~/pystandard/bin/activate

export OMP_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

echo "=== bfg_analyze.py --cluster ${CLUSTER} (spectrum + SF + diagnostics) ==="
python bfg_analyze.py --cluster "${CLUSTER}"

echo "=== Analysis done ==="
