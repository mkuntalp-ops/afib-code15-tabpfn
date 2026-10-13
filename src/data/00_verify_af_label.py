"""
Stage 0 - G0 feasibility check (Methods 2.2).

Confirms, before building the full cohort, that patient_id linkage in
CODE-15% recovers enough patients with both an AF-labelled record and an
earlier sinus-rhythm record to make the study feasible.

Input : exams.csv (from CODE-15%, https://doi.org/10.5281/zenodo.4916206)
Output: printed feasibility counts only (no file written)
"""
import pandas as pd

EXAMS_CSV = "data/exams.csv"


def main():
    e = pd.read_csv(EXAMS_CSV)

    print(f"Total exams: {len(e):,} | Total patients: {e.patient_id.nunique():,}")

    af_patients = set(e.loc[e["AF"] == 1, "patient_id"])
    sinus_patients = set(e.loc[e["normal_ecg"] == True, "patient_id"])  # noqa: E712
    candidates = af_patients & sinus_patients

    print(f"Patients with >=1 AF-labelled exam        : {len(af_patients):,}")
    print(f"Patients with >=1 normal_ecg exam          : {len(sinus_patients):,}")
    print(f"Candidate patients (both)                  : {len(candidates):,}")
    print(f"Feasibility threshold (n>=200)              : "
          f"{'PASSED' if len(candidates) >= 200 else 'below threshold'}")


if __name__ == "__main__":
    main()
