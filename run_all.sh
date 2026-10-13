#!/usr/bin/env bash
# Reproduce every number, table and figure of the revised manuscript.
# Usage: PY=python3.11 TABPFN_PY=python3.12 TABPFN_CHECKPOINT=/path/to/tabpfn-v3-classifier-v3_default.ckpt ./run_all.sh
# TABPFN_PY may be the same interpreter as PY if tabpfn is installed there; if tabpfn is
# not available, stage 8 still runs the six conventional models.
set -euo pipefail
PY=${PY:-python}
TABPFN_PY=${TABPFN_PY:-$PY}
mkdir -p outputs
run() { echo; echo "=== $1"; "${2:-$PY}" "$1"; }
run src/data/01_build_cohort.py
run src/data/02_extract_signals.py
run src/features/03_rr_features.py
run src/features/04_pwave_features.py
run src/features/05_ptfv1_classic.py
run src/matching/07_seed_averaged_matching.py
run src/validation/06_pipeline_controls.py
run src/models/08_model_comparison.py "$TABPFN_PY"
run src/validation/09_ptbxl_replication.py
run src/validation/10_s1_rmssd_validity.py
run src/validation/10b_rmssd_random_windows.py
run src/models/11_age_interaction.py
run src/revision/12_temporal_and_followup.py
run src/revision/13_ectopy.py
run src/revision/14_pr_verification.py
run src/revision/15_tables_and_reduction.py
run src/revision/16_figures.py
echo; echo "All stages finished; results in outputs/."
