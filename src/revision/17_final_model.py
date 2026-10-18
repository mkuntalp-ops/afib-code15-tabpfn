"""
Revision 17 - Full specification of the final model (TRIPOD+AI item 22).

Fits the final three-feature pipeline (median imputation -> standardisation -> logistic
regression) once on the complete seed-42 matched cohort and writes every parameter needed
to compute a prediction for a new record:

    z_k   = (x_k - mean_k) / sd_k            (missing x_k replaced by median_k first)
    logit = intercept + sum_k coef_k * z_k
    p     = 1 / (1 + exp(-logit))

Because the cohort is 1:1 matched, p is the probability of belonging to the case group at 50%
prevalence; it is not an absolute risk.

Input : outputs/matched_seed42.csv
Output: outputs/final_model.json
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import OUT, FINAL3, save_json


def main():
    m = pd.read_csv(OUT / "matched_seed42.csv")
    y = (m.split == "vaka").astype(int).values
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(m[FINAL3].values, y)
    imp, sc, lr = pipe
    spec = {"features": FINAL3, "units": {"RMSSD": "ms", "PR_med": "ms", "Pamp_V1": "mV"},
            "impute_median": dict(zip(FINAL3, map(float, imp.statistics_))),
            "mean": dict(zip(FINAL3, map(float, sc.mean_))), "sd": dict(zip(FINAL3, map(float, sc.scale_))),
            "coef_standardised": dict(zip(FINAL3, map(float, lr.coef_[0]))), "intercept": float(lr.intercept_[0]),
            "n_records": int(len(m)), "prevalence": 0.5,
            "note": "Fitted on the seed-42 matched cohort; probability at 50% prevalence, not absolute risk."}
    spec["check_prediction_first_record"] = float(pipe.predict_proba(m[FINAL3].values[:1])[0, 1])
    save_json(spec, "final_model.json"); print(spec)


if __name__ == "__main__":
    main()
