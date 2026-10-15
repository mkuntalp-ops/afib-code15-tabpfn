"""
Stage 11 - Discrimination by age and age-by-feature interaction (Results III-N).

Input : outputs/matched_seed42.csv
Output: outputs/11_age.json
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from common import OUT, FINAL3, oof_pred, save_json


def main():
    m = pd.read_csv(OUT / "matched_seed42.csv")
    y, p = oof_pred(m, FINAL3)
    res = {"strata": {}}
    for name, msk in [("<65", m.age_at_index < 65), (">=65", m.age_at_index >= 65),
                      ("<75", m.age_at_index < 75), (">=75", m.age_at_index >= 75)]:
        res["strata"][name] = {"n": int(msk.sum()), "AUROC": float(roc_auc_score(y[msk.values], p[msk.values]))}
    Z = m.copy()
    for f in FINAL3:
        Z[f] = Z[f].fillna(Z[f].median()); Z[f] = (Z[f] - Z[f].mean()) / Z[f].std()
    Z["age_c"] = (Z.age_at_index - Z.age_at_index.mean()) / 10
    X = Z[FINAL3 + ["age_c"]].copy()
    for f in FINAL3:
        X[f + ":age"] = Z[f] * Z.age_c
    fit = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    res["interaction_p"] = {f: float(fit.pvalues[f + ":age"]) for f in FINAL3}
    save_json(res, "11_age.json"); print(res)


if __name__ == "__main__":
    main()
