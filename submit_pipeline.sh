#!/bin/bash
#
# submit_pipeline.sh — One-command launcher for the BFG ED pipeline
#
# Usage:
#   ./submit_pipeline.sh <cluster>               # full pipeline (compute + analyze)
#   ./submit_pipeline.sh <cluster> --compute-only # compute array only
#   ./submit_pipeline.sh <cluster> --analyze-only # analysis only (compute done)
#
# Clusters: 2x3, 3x3, 3x3_to
#
# Examples:
#   ./submit_pipeline.sh 3x3_to
#   ./submit_pipeline.sh 3x3 --compute-only
#   ./submit_pipeline.sh 2x3 --analyze-only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Parse arguments ---
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <cluster> [--compute-only|--analyze-only]"
    echo "  Clusters: 2x3, 3x3, 3x3_to"
    exit 1
fi

CLUSTER="$1"
MODE="${2:-full}"

case "$CLUSTER" in
    2x3|3x3|3x3_to) ;;
    *)
        echo "ERROR: unknown cluster '$CLUSTER'. Use: 2x3, 3x3, 3x3_to"
        exit 1
        ;;
esac

# --- Auto-detect number of Jpm values ---
echo "Detecting Jpm values for cluster=${CLUSTER}..."

# Use Python to discover Jpm count (same logic as bfg_compute.py)
module --force purge 2>/dev/null || true
module load StdEnv/2023 python scipy-stack hdf5 2>/dev/null || true
source ~/pystandard/bin/activate 2>/dev/null || true

N_JPM=$(python3 -c "
import sys
sys.path.insert(0, '.')
from bfg_compute import get_config, discover_jpm_values
cfg = get_config('${CLUSTER}')
jpms = discover_jpm_values(cfg)
print(len(jpms))
")

if [[ "$N_JPM" -le 0 ]]; then
    echo "ERROR: no Jpm values found for cluster=${CLUSTER}"
    exit 1
fi

ARRAY_MAX=$((N_JPM - 1))
echo "  Found ${N_JPM} Jpm values → array=0-${ARRAY_MAX}"

# --- Short name for job labels ---
case "$CLUSTER" in
    3x3_to) TAG="3x3to" ;;
    *)      TAG="$CLUSTER" ;;
esac

# --- Submit ---
case "$MODE" in
    --compute-only)
        echo ""
        echo "=== Submitting compute array (${N_JPM} tasks) ==="
        sbatch --export=CLUSTER="${CLUSTER}" \
               --array="0-${ARRAY_MAX}" \
               --job-name="comp_${TAG}" \
               submit_compute.sh
        ;;

    --analyze-only)
        echo ""
        echo "=== Submitting analysis (standalone) ==="
        sbatch --export=CLUSTER="${CLUSTER}" \
               --job-name="ana_${TAG}" \
               submit_analyze.sh
        ;;

    full|"")
        echo ""
        echo "=== Submitting compute array (${N_JPM} tasks) + chained analysis ==="

        COMP_JOB=$(sbatch --export=CLUSTER="${CLUSTER}" \
                          --array="0-${ARRAY_MAX}" \
                          --job-name="comp_${TAG}" \
                          submit_compute.sh | awk '{print $4}')

        echo "  Compute job: ${COMP_JOB}"

        ANA_JOB=$(sbatch --dependency=afterok:"${COMP_JOB}" \
                         --export=CLUSTER="${CLUSTER}" \
                         --job-name="ana_${TAG}" \
                         submit_analyze.sh | awk '{print $4}')

        echo "  Analysis job: ${ANA_JOB} (depends on ${COMP_JOB})"
        echo ""
        echo "Monitor: squeue -u \$USER"
        echo "Cancel:  scancel ${COMP_JOB} ${ANA_JOB}"
        ;;

    *)
        echo "ERROR: unknown mode '$MODE'. Use --compute-only or --analyze-only"
        exit 1
        ;;
esac

echo "Done."
