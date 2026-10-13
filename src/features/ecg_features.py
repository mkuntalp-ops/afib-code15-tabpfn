"""Per-record feature extraction shared by stages 3-5 (Methods II-D, II-E).

Filtering: 0.5-40 Hz third-order Butterworth band-pass, zero-phase (filtfilt), applied to
the zero-padding-free segment. R peaks: NeuroKit2 ecg_peaks. Delineation: NeuroKit2
ecg_delineate(method="dwt") in every lead.
"""
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt

FS = 400
LEAD_II, V1 = 1, 6
BB, BA = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
PRE_OFFSET_SAMPLES = 16    # 40 ms at 400 Hz


def rr_features(x_ii):
    """RR features from band-passed lead II (all records kept; quality filter applied later)."""
    try:
        rp = nk.ecg_peaks(x_ii, sampling_rate=FS)[1]["ECG_R_Peaks"]
    except Exception:
        rp = np.array([])
    rr = np.diff(rp) / FS * 1000 if len(rp) > 1 else np.array([])
    rr = rr[(rr > 300) & (rr < 2000)]
    return {"n_R": int(len(rp)), "n_RR": int(len(rr)),
            "meanRR": float(rr.mean()) if len(rr) else np.nan,
            "SDNN": float(rr.std(ddof=1)) if len(rr) > 1 else np.nan,
            "RMSSD": float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else np.nan}


def pwave_features(sig12):
    """P-wave and PR features over 12 leads (sig12 already stripped of zero padding).

    PR: per lead, median over beats of (R onset - P onset), kept if 60-400 ms; record value =
    median over leads (PR_med) and maximum (PR_max). P amplitude: median over beats of the
    band-passed signal at the P peak. PTFV1_proxy: median over beats of the signed integral of
    V1 over the 40 ms before P offset (superseded proxy, used for the artefact analysis).
    """
    rec, p_dur, prs = {}, [], []
    for lead in range(12):
        try:
            x = filtfilt(BB, BA, sig12[:, lead])
            rp = nk.ecg_peaks(x, sampling_rate=FS)[1]["ECG_R_Peaks"]
            if len(rp) < 4:
                continue
            w = nk.ecg_delineate(x, rp, sampling_rate=FS, method="dwt")[1]
            on = np.array(w["ECG_P_Onsets"], float); off = np.array(w["ECG_P_Offsets"], float)
            r_on = np.array(w["ECG_R_Onsets"], float); pk = np.array(w["ECG_P_Peaks"], float)
            valid = ~(np.isnan(on) | np.isnan(off))
            if valid.sum() >= 2:
                dur = float(np.median(off[valid] - on[valid])) / FS * 1000
                if 40 < dur < 250:
                    p_dur.append(dur); rec[f"Pdur_{lead}"] = dur
            pr_valid = ~(np.isnan(on) | np.isnan(r_on))
            if pr_valid.sum() >= 2:
                pr = float(np.median(r_on[pr_valid] - on[pr_valid])) / FS * 1000
                if 60 < pr < 400:
                    prs.append(pr); rec[f"PR_{lead}"] = pr
            amp_valid = ~np.isnan(pk)
            if amp_valid.sum() >= 2:
                rec[f"Pamp_{lead}"] = float(np.median(x[pk[amp_valid].astype(int)]))
            if lead == V1 and valid.sum() >= 2:
                seg = [x[max(0, int(o) - PRE_OFFSET_SAMPLES): int(o)] for o in off[valid] if int(o) > PRE_OFFSET_SAMPLES]
                if seg:
                    rec["PTFV1_proxy"] = float(np.median([t.sum() / FS * 1000 for t in seg]))
        except Exception:
            continue
    rec["n_lead_ok"] = len(p_dur)
    if p_dur:
        rec.update(Pdur_max=max(p_dur), Pdur_min=min(p_dur), Pdisp=max(p_dur) - min(p_dur),
                   Pdur_med=float(np.median(p_dur)))
    if prs:
        rec.update(PR_med=float(np.median(prs)), PR_max=max(prs))
    rec["Pamp_V1"] = rec.get(f"Pamp_{V1}", np.nan)
    return rec


def terminal_negative_area(segment, fs=FS):
    """Signed area (mV ms) of the terminal negative deflection: the last contiguous run of
    negative samples, found by scanning backwards from the end of the P wave."""
    if len(segment) == 0 or not (segment < 0).any():
        return 0.0
    k = len(segment) - 1
    while k >= 0 and segment[k] >= 0:
        k -= 1
    j = k
    while j >= 0 and segment[j] < 0:
        j -= 1
    term = segment[j + 1: k + 1]
    return float(term.sum() / fs * 1000) if len(term) else 0.0


def ptfv1_classic(sig12):
    """Classic P-terminal force in V1 (Morris index), median over beats."""
    x = filtfilt(BB, BA, sig12[:, V1])
    try:
        rp = nk.ecg_peaks(x, sampling_rate=FS)[1]["ECG_R_Peaks"]
        if len(rp) < 4:
            return np.nan
        w = nk.ecg_delineate(x, rp, sampling_rate=FS, method="dwt")[1]
        on = np.array(w["ECG_P_Onsets"], float); off = np.array(w["ECG_P_Offsets"], float)
        valid = ~(np.isnan(on) | np.isnan(off))
        vals = [terminal_negative_area(x[o1:o2]) for o1, o2 in zip(on[valid].astype(int), off[valid].astype(int))
                if o1 < o2 <= len(x)]
        return float(np.median(vals)) if vals else np.nan
    except Exception:
        return np.nan


def load_segments(out_dir):
    """Yield (exam_id, zero-padding-free 12-lead segment) for every QC-passed record."""
    import pandas as pd
    z = np.load(out_dir / "cohort_signals.npz")
    ids, tr = z["exam_id"], z["tracings"]
    clean = pd.read_csv(out_dir / "cohort_clean.csv")
    pos = {int(e): i for i, e in enumerate(ids)}
    return [(int(r.exam_id), tr[pos[int(r.exam_id)]][int(r.start): int(r.end) + 1, :]) for r in clean.itertuples()]
