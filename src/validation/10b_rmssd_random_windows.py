"""Stage 10b - Short-window RMSSD validity with randomly placed 7-10 s windows (Reviewer 3, point 6).

For each 5-minute reference window of the PAF Prediction Challenge database (afpdb,
128 Hz, lead 0) we draw K random 7-10 s segments (uniform start, uniform duration) and
compute RMSSD from the RR intervals inside each segment, using the CODE acceptance rule
(>=6 detected R peaks, RR in 300-2000 ms). From the draws we estimate
  (a) agreement with the 5-min reference for a single random segment (ICC(2,1), ICC(3,1),
      Bland-Altman bias and limits, share within +-20% of reference), and
  (b) test-retest reliability of a single segment (ICC(2,1) between two independent
      random segments taken from the same 5-minute window).
Recording-level clustering is respected by bootstrapping whole recordings.
"""
import sys, glob, json, os, zlib
sys.path.insert(0, "src")
from common import DATA, OUT
import numpy as np, pandas as pd, wfdb, neurokit2 as nk
from scipy.signal import butter, filtfilt
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

FS = 128
K = 200
AFPDB = str(DATA / "afpdb")
BB, BA = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")


def icc21(a, b):
    X = np.column_stack([a, b]); n, k = X.shape; gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def icc31(a, b):
    X = np.column_stack([a, b]); n, k = X.shape; gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse)


def rmssd_from_peaks(t_peaks):
    """RMSSD over RR intervals (s->ms) from peak times in seconds, CODE acceptance rule."""
    if len(t_peaks) < 6:
        return np.nan
    rr = np.diff(t_peaks) * 1000
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 5:
        return np.nan
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)))


def process(rec_path):
    rng = np.random.default_rng(zlib.crc32(os.path.basename(rec_path).encode()))  # deterministic per record
    rec = rec_path[:-4]
    group = "PAF" if os.path.basename(rec)[0] == "p" else "control"
    sig, _ = wfdb.rdsamp(rec)
    x = filtfilt(BB, BA, sig[:, 0])
    _, info = nk.ecg_peaks(x, sampling_rate=FS)
    rp = info["ECG_R_Peaks"] / FS  # seconds
    rows = []
    dur = len(x) / FS
    for w0 in np.arange(0, dur - 300 + 1e-9, 300):
        sel = rp[(rp >= w0) & (rp < w0 + 300)]
        rr = np.diff(sel) * 1000
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) < 200:
            continue
        gold = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        singles = np.full((K, 2), np.nan)
        for kk in range(K):
            for j in range(2):   # two independent segments per window (test-retest)
                D = rng.uniform(7, 10)
                st = rng.uniform(w0, w0 + 300 - D)
                seg = sel[(sel >= st) & (sel < st + D)]
                singles[kk, j] = rmssd_from_peaks(seg)
        for kk in range(K):
            rows.append({"rec": os.path.basename(rec), "group": group, "w0": w0, "draw": kk,
                         "gold": gold, "seg1": singles[kk, 0], "seg2": singles[kk, 1]})
    return rows


if __name__ == "__main__":
    recs = sorted(glob.glob(f"{AFPDB}/*.dat"))
    recs = [r for r in recs if os.path.basename(r)[0] in "np"]  # 100 study recordings (50 PAF, 50 non-AF)
    with Pool(8) as p:
        out_rows = p.map(process, recs)
    df = pd.DataFrame([r for rows in out_rows for r in rows]).dropna(subset=["gold"])
    df.to_csv(OUT / "10b_rmssd_random_windows.csv", index=False)

    def summarise(sub):
        g = sub.gold.values
        res = {"windows": int(sub[sub.draw == 0].shape[0]), "recordings": int(sub.rec.nunique())}
        # agreement of one random segment (seg1) with reference, per draw index, then summarised
        icc2_d, icc3_d, bias_d, lo_d, hi_d, within20 = [], [], [], [], [], []
        for kk in range(K):
            s = sub[sub.draw == kk].dropna(subset=["seg1"])
            if len(s) < 10:
                continue
            a, b = s.gold.values, s.seg1.values
            icc2_d.append(icc21(a, b)); icc3_d.append(icc31(a, b))
            diff = b - a
            bias_d.append(diff.mean()); lo_d.append(diff.mean() - 1.96 * diff.std(ddof=1))
            hi_d.append(diff.mean() + 1.96 * diff.std(ddof=1))
            within20.append(np.mean(np.abs(b - a) <= 0.2 * a))
        # test-retest reliability of a single segment: seg1 vs seg2 from same window
        tr = []
        for kk in range(K):
            s = sub[sub.draw == kk].dropna(subset=["seg1", "seg2"])
            if len(s) >= 10:
                tr.append(icc21(s.seg1.values, s.seg2.values))
        res.update({
            "ICC21_single_vs_ref_median": float(np.median(icc2_d)),
            "ICC21_single_vs_ref_IQR": [float(np.percentile(icc2_d, 25)), float(np.percentile(icc2_d, 75))],
            "ICC31_single_vs_ref_median": float(np.median(icc3_d)),
            "bias_ms_median": float(np.median(bias_d)),
            "LoA_low_ms_median": float(np.median(lo_d)), "LoA_high_ms_median": float(np.median(hi_d)),
            "within_20pct_frac_median": float(np.median(within20)),
            "test_retest_ICC21_median": float(np.median(tr)),
            "test_retest_ICC21_IQR": [float(np.percentile(tr, 25)), float(np.percentile(tr, 75))],
            "gold_median_ms": float(np.median(g)),
            "median_ratio_single_over_ref": float(np.median((sub.seg1 / sub.gold).dropna())),
            "seg1_median_ms": float(np.median(sub.seg1.dropna())),
            "seg1_CV_within_window_median": float(np.median(
                sub.groupby(["rec", "w0"]).seg1.apply(lambda v: v.std() / v.mean() if v.mean() > 0 else np.nan).dropna())),
        })
        return res

    out = {"All": summarise(df), "PAF": summarise(df[df.group == "PAF"]),
           "control": summarise(df[df.group == "control"])}
    json.dump(out, open(OUT / "10b_random_windows.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
