# How much AF risk is measurable on routine sinus-rhythm ECG?

Analysis code for the manuscript:

> **How much atrial fibrillation risk is measurable on routine sinus-rhythm ECG? A case-control analysis with age-ordered index recordings and patient-level cross-validation** (revised version, JBHI major revision)

Every number, table and figure in the revised manuscript is produced by the scripts in `src/`, run in the order given in `run_all.sh`.

## Data

The datasets are not redistributed. Download them from their original sources and place them in `data/`:

| Path | Source | Used for |
|---|---|---|
| `data/exams.csv` | [CODE-15%](https://doi.org/10.5281/zenodo.4916206) | Cohort construction |
| `data/exams_part*.hdf5` | CODE-15% (18 parts) | Waveforms |
| `data/cohort_tracings/exams_part*.npz` | optional, pre-extracted cohort waveforms (`exam_id`, `tracings`) | used instead of the HDF5 parts if those are absent |
| `data/ptbxl/` | [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/) (`records500/`, `ptbxl_database.csv`) | External replication |
| `data/ptbxl_cohort.csv` | PTB-XL case-control cohort (135 cases, 540 controls; columns `ecg_id, patient_id, age, sex, file, af_oncesi, gun, split`) | External replication |
| `data/afpdb/` | [PAF Prediction Challenge Database](https://physionet.org/content/afpdb/) (n01-n50, p01-p50) | Short-window RMSSD substudy |

Do not upload or commit raw ECG data or patient-level derived files. `data/` and `outputs/` are git-ignored.

## Requirements

Analysis (stages 1-7, 9-16): Python 3.11 with the packages pinned in `requirements.txt`.

TabPFN (stage 8): `tabpfn` with the TabPFN-3 classifier checkpoint `tabpfn-v3-classifier-v3_default.ckpt`, passed explicitly through the environment variable `TABPFN_CHECKPOINT`. The original submission used tabpfn 8.1.0; the revision used tabpfn 9.0.0 (Python 3.12, scikit-learn 1.9.1). The checkpoint is licensed for research and evaluation only. If `tabpfn` is not installed, stage 8 runs the six conventional models and skips TabPFN.

## Reproducing the analysis

```bash
PY=python3.11 TABPFN_PY=python3.12 \
TABPFN_CHECKPOINT=/path/to/tabpfn-v3-classifier-v3_default.ckpt \
./run_all.sh
```

All outputs (tables as CSV/JSON, figures as PNG) are written to `outputs/`.

| Stage | Script | Main output | Manuscript |
|---|---|---|---|
| 0 | `src/data/00_verify_af_label.py` | feasibility counts | Methods II-B |
| 1 | `src/data/01_build_cohort.py` | `cohort_raw.csv` | II-B, Fig. 1 |
| 2 | `src/data/02_extract_signals.py` | `cohort_signals.npz`, `signal_bounds.csv`, `cohort_clean.csv` | II-D, Fig. 1 |
| 3 | `src/features/03_rr_features.py` | `rr_features.csv` | II-D, II-E |
| 4 | `src/features/04_pwave_features.py` | `pwave_features.csv` | II-E, II-F |
| 5 | `src/features/05_ptfv1_classic.py` | `ptfv1_true.csv` | II-E, Table III |
| 7 | `src/matching/07_seed_averaged_matching.py` | `matched_seed42.csv`, `07_primary.json`, `07_seed_results.csv` | III-D, III-I, III-L, Table V |
| 6 | `src/validation/06_pipeline_controls.py` | `06_pipeline_controls.json` | II-H, III-O |
| 8 | `src/models/08_model_comparison.py` | `08_models.json` | III-D, III-J, III-P, Tables IV, VII |
| 9 | `src/validation/09_ptbxl_replication.py` | `09_ptbxl.json`, `ptbxl_features.csv` | II-M, III-R, Fig. 6 |
| 10 | `src/validation/10_s1_rmssd_validity.py` | printed ICCs (10-s sub-window median) | II-L, III-Q, Table VIII |
| 10b | `src/validation/10b_rmssd_random_windows.py` | `10b_random_windows.json` | II-L, III-Q, Table VIII |
| 11 | `src/models/11_age_interaction.py` | `11_age.json` | III-N |
| 12 | `src/revision/12_temporal_and_followup.py` | `12_temporal_followup.json` | II-C, III-B, III-F, III-G |
| 13 | `src/revision/13_ectopy.py` | `13_ectopy.json` | II-I, III-O |
| 14 | `src/revision/14_pr_verification.py` | `14_pr_verification.json` | II-G, III-S, Fig. 7 |
| 15 | `src/revision/15_tables_and_reduction.py` | `15_tables_reduction.json` | III-C, III-E, III-H, III-K, III-M, Tables I, III, VI |
| 16 | `src/revision/16_figures.py` | `outputs/figures/` | Figs. 1-7 |

Stage 6 runs after stage 7 because the negative control uses the seed-42 matched cohort.

## Design decisions that are easy to get wrong

- **Cohort.** Cases: latest AF = 0 exam at a recorded age below the first AF-labelled exam; patients with an AF = 0 exam at the same age as the first AF-labelled exam are excluded from the primary analysis. No age filter is applied. Controls: random sample of four per case (`numpy.random.default_rng(42)` on the sorted eligible patient ids), index = earliest AF = 0 exam.
- **Signal quality.** Records with a non-standard length are excluded at stage 2; the analysis requires at least six detected R peaks on lead II (`n_R >= 6`, i.e. five RR intervals), applied in `common.load_pool`.
- **Missing archive part.** In the manuscript analysis, `exams_part0.hdf5` was not extracted, so its records (about 6% of the cohort) are absent. `SKIP_PARTS` in stage 2 reproduces this; set it to an empty set to include all parts (results will then differ slightly from the manuscript).
- **Positive-control records.** For each case one AF-labelled exam at the first-AF age is taken (first in file order). The original analysis broke ties at that age differently for about 7% of patients; only the positive control (stage 6) is affected.
- **Two V1 terminal measures.** `PTFV1_proxy` (40 ms pre-offset integral, a proxy for P-wave amplitude) is not the classic P-terminal force `PTFV1_true` (Morris index).
- **Matching is random.** All primary estimates are averaged over the 11 seeds in `config/seeds.txt`; seed 42 is used for single-cohort tables and figures.
- **Outcome-informed steps.** See Table I of the manuscript; stage 15 quantifies them (nested Stage 2, model without a V1 measure).

## Generative-AI assistance and author responsibility

Claude Code (Anthropic) assisted with drafting and refactoring the analysis scripts, including the scripts added for the revision. The author defined the research question, cohort definitions and analytic requirements, executed the code locally, reviewed the implementation and outputs, and retains full responsibility for the work.

## Citation

```text
[Add full manuscript citation and DOI after publication]
```
