"""
Stage 1 - Cohort construction (Methods II-B, Fig. 1).

Reproduces the cohort used in the manuscript exactly (no age filter):
  cases   : patients with >=1 AF-labelled exam and, at a lower recorded age, >=1 exam with
            AF == 0. The latest such exam (closest to the first AF-labelled exam) is the
            index. Patients who also have an AF == 0 exam at the same recorded age as the first
            AF-labelled exam are flagged 'vaka_belirsiz' (ordering ambiguous; excluded from
            the primary analysis).
  controls: random sample, 4 per clear-ordering case, of patients with no AF-labelled exam
            and >=1 AF == 0 exam (numpy default_rng(42), sampled from the sorted patient ids).
            Index = earliest AF == 0 exam (ties: first in file order after sorting by age).
  sentinel_af: for every clear-ordering case, one AF-labelled exam at the first-AF age (used
            only for the positive control in stage 6).

Input : data/exams.csv
Output: outputs/cohort_raw.csv
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from common import EXAMS, OUT

CONTROLS_PER_CASE = 4
SEED = 42


def main():
    df = pd.read_csv(EXAMS)
    rng = np.random.default_rng(SEED)
    af_pat = set(df.loc[df.AF == 1, "patient_id"])
    sinus_pat = set(df.loc[df.AF == 0, "patient_id"])
    by_pat = df.groupby("patient_id")

    cases, sentinels = [], []
    for pid in af_pat & sinus_pat:
        sub = by_pat.get_group(pid)
        af_rows = sub[sub.AF == 1].sort_values("age")
        af_age = af_rows.iloc[0]["age"]
        before = sub[(sub.AF == 0) & (sub.age < af_age)].sort_values("age", ascending=False)
        if before.empty:
            continue
        same_age = ((sub.AF == 0) & (sub.age == af_age)).any()
        idx = before.iloc[0]
        split = "vaka_belirsiz" if same_age else "vaka"
        cases.append(dict(patient_id=pid, exam_id=idx.exam_id, age_at_index=idx.age,
                          age_at_first_af=af_age, durum="ayni_yas_riskli" if same_age else "net",
                          split=split))
        if split == "vaka":
            s = af_rows.iloc[0]
            sentinels.append(dict(patient_id=pid, exam_id=s.exam_id, age_at_index=s.age,
                                  age_at_first_af=af_age, durum="af_ritim", split="sentinel_af"))
    cases = pd.DataFrame(cases)
    n_clear = int((cases.split == "vaka").sum())

    pool = sinus_pat - af_pat
    chosen = rng.choice(sorted(pool), size=min(len(pool), n_clear * CONTROLS_PER_CASE), replace=False)
    controls = []
    for pid in chosen:
        sub = by_pat.get_group(pid)
        idx = sub[sub.AF == 0].sort_values("age").iloc[0]
        controls.append(dict(patient_id=pid, exam_id=idx.exam_id, age_at_index=idx.age,
                             age_at_first_af=np.nan, durum="kontrol", split="kontrol"))

    cohort = pd.concat([cases, pd.DataFrame(controls), pd.DataFrame(sentinels)], ignore_index=True)
    cohort.to_csv(OUT / "cohort_raw.csv", index=False)
    print(f"AF-labelled patients              : {len(af_pat):,}")
    print(f"  with a pre-AF sinus exam         : {len(cases):,}")
    print(f"  clear ordering (cases)           : {n_clear:,}")
    print(f"  same recorded age (excluded)     : {(cases.split == 'vaka_belirsiz').sum():,}")
    print(f"Eligible control patients          : {len(pool):,}")
    print(f"Sampled controls (1:{CONTROLS_PER_CASE}, seed {SEED}) : {len(controls):,}")
    print(f"-> {OUT / 'cohort_raw.csv'}")


if __name__ == "__main__":
    main()
