"""
Revision 12 - Temporal resolution of the age-based ordering, discrimination by recorded
interval, follow-up and ECG-contact asymmetry, and the controls-with->=2-ECGs sensitivity
analysis (Methods II-C, II-J; Results III-B, III-F, III-G).

Recorded ages a_i < a_af are whole years. With true age t = a + u, u ~ U(0,1) (truncation) or
U(-0.5,0.5) (rounding), the true interval is T = (a_af - a_i) + (u_af - u_i), which is always
positive; its distribution is obtained by simulation.

Input : outputs/cohort_clean.csv and feature tables, data/exams.csv
Output: outputs/12_temporal_followup.json
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import OUT, EXAMS, SEEDS, FINAL3, RRB, load_pool, match_once, oof_pred, save_json


def main():
    res = {}
    pool = load_pool()
    pool["dAge"] = pool.age_at_first_af - pool.age_at_index
    res["recorded_interval_counts_pool"] = {int(k): int(v) for k, v in pool[pool.split == "vaka"].dAge.value_counts().sort_index().items()}
    rng = np.random.default_rng(2026)
    N = 2_000_000
    for conv, lo in [("floor", 0.0), ("round", -0.5)]:
        u1 = rng.uniform(0, 1, N) + lo; u2 = rng.uniform(0, 1, N) + lo
        for delta in [1, 2]:
            T = delta + (u1 - u2)
            res[f"{conv}_delta{delta}"] = {"P_lt_1y": float((T < 1).mean()), "P_lt_6m": float((T < .5).mean()),
                                           "P_lt_1m": float((T < 1 / 12).mean()), "P_le_0": float((T <= 0).mean())}
    m42 = pd.read_csv(OUT / "matched_seed42.csv")
    n1 = int(((m42.split == "vaka") & (m42.age_at_first_af - m42.age_at_index == 1)).sum())
    res["matched_delta1_cases"] = n1
    res["expected_true_lt"] = {k: n1 * res["floor_delta1"][v] for k, v in [("1y", "P_lt_1y"), ("6m", "P_lt_6m"), ("1m", "P_lt_1m")]}

    strata = {"delta1": [], "delta_ge2": []}
    for s in SEEDS:
        m, pairs = match_once(pool, s)
        dd = dict(zip(m[m.split == "vaka"].exam_id, m[m.split == "vaka"].dAge))
        for name, cond in [("delta1", lambda x: x == 1), ("delta_ge2", lambda x: x >= 2)]:
            keep = [(c, k) for c, k in pairs if cond(dd[c])]
            sub = m[m.exam_id.isin([x for pr in keep for x in pr])]
            y, p = oof_pred(sub, FINAL3); _, q = oof_pred(sub, RRB)
            strata[name].append((len(keep), roc_auc_score(y, p), roc_auc_score(y, q)))
    res["interval_strata"] = {k: {"pairs": v[0][0], "Final3": [float(np.mean([x[1] for x in v])), float(np.std([x[1] for x in v]))],
                                  "RR": [float(np.mean([x[2] for x in v])), float(np.std([x[2] for x in v]))]} for k, v in strata.items()}

    ex = pd.read_csv(EXAMS, usecols=["exam_id", "patient_id", "age", "timey", "death"])
    n_ecg = ex.groupby("patient_id").size()
    first = ex[ex.timey.notna()].set_index("patient_id")
    f = m42.drop(columns=["timey", "death"], errors="ignore").join(first[["age", "timey", "death"]].rename(columns={"age": "age_first", "death": "died"}), on="patient_id")
    f["n_ecg"] = f.patient_id.map(n_ecg)
    f["remaining"] = f.age_first + f.timey - f.age_at_index        # timey in years
    res["followup"] = {g: {"n_ecg_median": float(s.n_ecg.median()), "n_ecg_ge2": float((s.n_ecg >= 2).mean()),
                           "remaining_median": float(s.remaining.median()),
                           "remaining_IQR": [float(s.remaining.quantile(.25)), float(s.remaining.quantile(.75))],
                           "death_frac": float(s.died.astype(float).mean())}
                       for g, s in f.groupby("split")}
    c2 = f[(f.split == "kontrol") & (f.n_ecg >= 2)]
    res["followup"]["kontrol_ge2"] = {"n": int(len(c2)), "remaining_median": float(c2.remaining.median())}

    pool["n_ecg"] = pool.patient_id.map(n_ecg)
    p2 = pool[(pool.split == "vaka") | (pool.n_ecg >= 2)]
    vals = [(len(match_once(p2, s)[1]), roc_auc_score(*oof_pred(match_once(p2, s)[0], FINAL3))) for s in SEEDS]
    res["controls_ge2_ecgs"] = {"pairs": vals[0][0], "Final3": [float(np.mean([v[1] for v in vals])), float(np.std([v[1] for v in vals]))]}
    save_json(res, "12_temporal_followup.json"); print(res)


if __name__ == "__main__":
    main()
