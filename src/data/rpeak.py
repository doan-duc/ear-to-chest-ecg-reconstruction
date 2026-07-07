"""R-peak detection (Pan-Tompkins) and PQRST mask generation.

Masks are derived from the filtered chest reference ECG so the reconstruction
metric focuses on the clinically relevant beat morphology.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def _bandpass_1d(x, fs=250, lowcut=5.0, highcut=15.0, order=2):
    x = np.asarray(x, dtype=np.float32)
    nyq = fs / 2.0
    highcut = min(highcut, nyq * 0.95)
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def _local_max_abs_index(x, center, radius):
    L = len(x)
    start, end = max(0, center - radius), min(L, center + radius + 1)
    if end <= start:
        return center, 0.0
    idx = start + int(np.argmax(np.abs(x[start:end])))
    return idx, float(np.abs(x[idx]))


def _local_slope(x, center, radius):
    L = len(x)
    start, end = max(0, center - radius), min(L, center + radius + 1)
    if end - start < 3:
        return 0.0
    return float(np.max(np.abs(np.diff(x[start:end]))))


def _deduplicate_peaks(peaks, signal, min_distance):
    if len(peaks) == 0:
        return np.array([], dtype=np.int64)
    peaks = np.sort(np.unique(np.asarray(peaks, dtype=np.int64)))
    final = []
    for p in peaks:
        if not final or p - final[-1] >= min_distance:
            final.append(p)
        elif abs(signal[p]) > abs(signal[final[-1]]):
            final[-1] = p
    return np.asarray(final, dtype=np.int64)


def pan_tompkins_rpeaks_1d(ecg_1d, fs=250, lowcut=5.0, highcut=15.0, filter_order=2,
                           integration_window_sec=0.150, refractory_sec=0.200,
                           t_wave_check_sec=0.360, refine_window_sec=0.080,
                           init_sec=2.0, eps=1e-8):
    """Detect R-peaks on a 1D reference ECG. Returns sample indices."""
    ecg = np.nan_to_num(np.asarray(ecg_1d, dtype=np.float32))
    L = len(ecg)
    if L < int(1.0 * fs) or np.std(ecg) < eps:
        return np.array([], dtype=np.int64)
    try:
        ecg_f = _bandpass_1d(ecg, fs, lowcut, highcut, filter_order)
    except Exception:
        return np.array([], dtype=np.int64)

    ecg_d = np.convolve(ecg_f, np.array([-1, -2, 0, 2, 1], dtype=np.float32) / 8.0, mode="same")
    ecg_sq = ecg_d ** 2
    win = max(1, int(integration_window_sec * fs))
    ecg_i = np.convolve(ecg_sq, np.ones(win, dtype=np.float32) / win, mode="same")
    if np.std(ecg_i) < eps:
        return np.array([], dtype=np.int64)

    candidate_peaks, _ = find_peaks(ecg_i, distance=max(1, int(0.150 * fs)))
    if len(candidate_peaks) == 0:
        return np.array([], dtype=np.int64)

    search_radius = int(0.150 * fs)
    abs_ecg_f = np.abs(ecg_f)
    cand_i_pos, cand_i_val, cand_f_pos, cand_f_val, cand_slope = [], [], [], [], []
    for p in candidate_peaks:
        cand_i_pos.append(p)
        cand_i_val.append(ecg_i[p])
        f_idx, f_val = _local_max_abs_index(abs_ecg_f, p, search_radius)
        cand_f_pos.append(f_idx)
        cand_f_val.append(f_val)
        cand_slope.append(_local_slope(ecg_f, f_idx, int(0.075 * fs)))
    cand_i_pos = np.asarray(cand_i_pos, dtype=np.int64)
    cand_i_val = np.asarray(cand_i_val, dtype=np.float32)
    cand_f_pos = np.asarray(cand_f_pos, dtype=np.int64)
    cand_f_val = np.asarray(cand_f_val, dtype=np.float32)
    cand_slope = np.asarray(cand_slope, dtype=np.float32)

    init_mask = cand_i_pos < min(L, int(init_sec * fs))
    init_i = cand_i_val[init_mask] if np.any(init_mask) else cand_i_val
    init_f = cand_f_val[init_mask] if np.any(init_mask) else cand_f_val
    SPKI, NPKI = 0.25 * np.max(init_i), 0.50 * np.mean(init_i)
    SPKF, NPKF = 0.25 * np.max(init_f), 0.50 * np.mean(init_f)

    refractory = int(refractory_sec * fs)
    t_wave = int(t_wave_check_sec * fs)
    detected_qrs_i, detected_qrs_f, detected_scores_i, detected_slopes, detected_idx = [], [], [], [], []
    rr_intervals = []
    last_qrs_i, last_qrs_cand, last_qrs_slope = None, -1, None

    def thr():
        ti1 = NPKI + 0.25 * (SPKI - NPKI)
        tf1 = NPKF + 0.25 * (SPKF - NPKF)
        return ti1, 0.5 * ti1, tf1, 0.5 * tf1

    def rr_missed_limit():
        if len(rr_intervals) >= 2:
            return int(1.66 * np.mean(np.asarray(rr_intervals[-8:], dtype=np.float32)))
        return int(1.66 * fs)

    def accept(j):
        nonlocal SPKI, SPKF, last_qrs_i, last_qrs_cand, last_qrs_slope
        qi, qf = int(cand_i_pos[j]), int(cand_f_pos[j])
        if last_qrs_i is not None:
            rr_intervals.append(qi - last_qrs_i)
        detected_qrs_i.append(qi); detected_qrs_f.append(qf)
        detected_scores_i.append(float(cand_i_val[j])); detected_slopes.append(float(cand_slope[j]))
        detected_idx.append(j)
        last_qrs_i, last_qrs_cand, last_qrs_slope = qi, j, float(cand_slope[j])
        SPKI = 0.125 * cand_i_val[j] + 0.875 * SPKI
        SPKF = 0.125 * cand_f_val[j] + 0.875 * SPKF

    def reject(j):
        nonlocal NPKI, NPKF
        NPKI = 0.125 * cand_i_val[j] + 0.875 * NPKI
        NPKF = 0.125 * cand_f_val[j] + 0.875 * NPKF

    for j in range(len(cand_i_pos)):
        p_i = int(cand_i_pos[j])
        THRESHOLD_I1, THRESHOLD_I2, THRESHOLD_F1, THRESHOLD_F2 = thr()
        if last_qrs_i is not None and p_i - last_qrs_i > rr_missed_limit():
            missed = [k for k in range(last_qrs_cand + 1, j)
                      if cand_i_pos[k] - last_qrs_i >= refractory
                      and cand_i_val[k] > THRESHOLD_I2 and cand_f_val[k] > THRESHOLD_F2]
            if missed:
                accept(max(missed, key=lambda k: cand_i_val[k]))
                THRESHOLD_I1, THRESHOLD_I2, THRESHOLD_F1, THRESHOLD_F2 = thr()
        if not (cand_i_val[j] >= THRESHOLD_I1 and cand_f_val[j] >= THRESHOLD_F1):
            reject(j)
            continue
        if last_qrs_i is not None and p_i - last_qrs_i < refractory:
            if detected_qrs_i and cand_i_val[j] > detected_scores_i[-1]:
                detected_qrs_i[-1] = int(cand_i_pos[j]); detected_qrs_f[-1] = int(cand_f_pos[j])
                detected_scores_i[-1] = float(cand_i_val[j]); detected_slopes[-1] = float(cand_slope[j])
                detected_idx[-1] = j
                last_qrs_i, last_qrs_cand, last_qrs_slope = int(cand_i_pos[j]), j, float(cand_slope[j])
                SPKI = 0.125 * cand_i_val[j] + 0.875 * SPKI
                SPKF = 0.125 * cand_f_val[j] + 0.875 * SPKF
            else:
                reject(j)
            continue
        if last_qrs_i is not None and p_i - last_qrs_i < t_wave:
            prev = float(last_qrs_slope) if last_qrs_slope is not None else float(cand_slope[j])
            if float(cand_slope[j]) < 0.5 * prev:
                reject(j)
                continue
        accept(j)

    if not detected_qrs_f:
        return np.array([], dtype=np.int64)
    refine_radius = int(refine_window_sec * fs)
    refined = np.asarray([_local_max_abs_index(ecg, q, refine_radius)[0] for q in detected_qrs_f], dtype=np.int64)
    return _deduplicate_peaks(refined, ecg, int(0.250 * fs))


def build_pqrst_masks(y_2d, fs=250, require_full_pqrst=False):
    """Build (N, 4, T) P/QRS/T/Other masks from reference ECG (N, T)."""
    y = np.asarray(y_2d, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    N, T = y.shape
    masks = np.zeros((N, 4, T), dtype=np.float32)
    p_left, p_right = int(0.250 * fs), int(0.080 * fs)
    qrs_left = qrs_right = int(0.080 * fs)
    t_left, t_right = int(0.120 * fs), int(0.400 * fs)
    r_peaks_all = []
    for i in range(N):
        r_peaks = pan_tompkins_rpeaks_1d(y[i], fs=fs)
        r_peaks_all.append(r_peaks)
        mp, mq, mt = (np.zeros(T, dtype=np.float32) for _ in range(3))
        for r in r_peaks:
            if require_full_pqrst and (r - p_left < 0 or r + t_right >= T):
                continue
            ps, pe = max(0, r - p_left), max(0, r - p_right)
            qs, qe = max(0, r - qrs_left), min(T, r + qrs_right + 1)
            ts, te = min(T, r + t_left), min(T, r + t_right + 1)
            if pe > ps:
                mp[ps:pe] = 1.0
            if qe > qs:
                mq[qs:qe] = 1.0
            if te > ts:
                mt[ts:te] = 1.0
        mq = np.clip(mq, 0, 1)
        mp = np.clip(mp, 0, 1) * (1 - mq)
        mt = np.clip(mt, 0, 1) * (1 - mq) * (1 - mp)
        masks[i, 0], masks[i, 1], masks[i, 2] = mp, mq, mt
        masks[i, 3] = 1.0 - np.clip(mp + mq + mt, 0, 1)
    return masks, r_peaks_all
