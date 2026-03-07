#!/usr/bin/env python
"""
クロスセッション分析 v2
- キャリブレーション = 60秒
- アンチエイリアスフィルタ付きダウンサンプル
- Z-score正規化も試行
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

OUTPUT_DIR = Path(__file__).parent

SESSIONS = {
    'session1': {
        'csv': 'session1/muse_app_2026-03-04--08-05-52.csv',
        'target': {
            'birds': 5, 'recoveries': 3,
            'active_sec': 19, 'neutral_sec': 398, 'calm_sec': 369,
            'calm_pct': 46, 'score': 46, 'duration_min': 15,
        },
    },
    'session2': {
        'csv': 'session2/muse_app_2026-03-04--17-46-58.csv',
        'target': {
            'birds': 29, 'recoveries': 3,
            'active_sec': 5, 'neutral_sec': 139, 'calm_sec': 403,
            'calm_pct': 73, 'score': 73, 'duration_min': 10,
        },
    },
}

CALIBRATION_SEC = 60
TARGET_SFREQ = 51.5
frontal_channels = ['RAW_AF7', 'RAW_AF8']


def downsample_with_aa(data, original_sfreq, target_sfreq):
    """アンチエイリアスフィルタ付きダウンサンプル"""
    ratio = int(round(original_sfreq / target_sfreq))
    if ratio <= 1:
        return data, original_sfreq
    # ローパスフィルタ (カットオフ = target_sfreq/2 の 0.9倍)
    nyq = original_sfreq / 2
    cutoff = (target_sfreq / 2) * 0.9
    sos = signal.butter(8, cutoff / nyq, btype='low', output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    downsampled = filtered[::ratio]
    return downsampled, original_sfreq / ratio


def compute_band_power(data, sfreq, band, window_sec=2.0, step_sec=1.0):
    """バンドパワー計算"""
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    powers = []
    for start in range(0, len(data) - window_samples, step_samples):
        seg = data[start:start + window_samples]
        nperseg = min(window_samples, int(sfreq))
        freqs, psd = signal.welch(seg, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        powers.append(np.trapz(psd[mask], freqs[mask]))
    return np.array(powers)


def count_sustained_events(values, threshold, time_vec, min_duration_sec):
    above = values >= threshold
    events = []
    in_event = False
    event_start = None
    for i in range(len(above)):
        if above[i] and not in_event:
            in_event = True
            event_start = i
        elif not above[i] and in_event:
            in_event = False
            dur = time_vec[i - 1] - time_vec[event_start]
            if dur >= min_duration_sec:
                events.append({
                    'start_idx': event_start, 'end_idx': i - 1,
                    'start_sec': time_vec[event_start], 'end_sec': time_vec[i - 1],
                    'duration': dur, 'peak': np.max(values[event_start:i]),
                })
    if in_event:
        dur = time_vec[-1] - time_vec[event_start]
        if dur >= min_duration_sec:
            events.append({
                'start_idx': event_start, 'end_idx': len(above) - 1,
                'start_sec': time_vec[event_start], 'end_sec': time_vec[-1],
                'duration': dur, 'peak': np.max(values[event_start:]),
            })
    return events


def classify_mind_state(ratio, calm_th, active_th):
    states = np.full(len(ratio), 'Neutral', dtype=object)
    states[ratio >= calm_th] = 'Calm'
    states[ratio <= active_th] = 'Active'
    return states


# ============================================================
# 各セッション処理
# ============================================================
results = {}

for sn, si in SESSIONS.items():
    print(f"\n{'='*70}")
    print(f"  {sn.upper()}")
    print(f"{'='*70}")

    df = pd.read_csv(OUTPUT_DIR / si['csv'])
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    t = si['target']

    duration_s = (df['TimeStamp'].iloc[-1] - df['TimeStamp'].iloc[0]).total_seconds()
    orig_sfreq = len(df) / duration_s
    print(f"元: {len(df)} samples, {duration_s:.1f}s, {orig_sfreq:.1f} Hz")

    # ダウンサンプル (アンチエイリアス付き)
    ch_data = {}
    actual_sfreq = None
    for ch in frontal_channels:
        raw = df[ch].values.astype(float)
        ds, sf = downsample_with_aa(raw, orig_sfreq, TARGET_SFREQ)
        ch_data[ch] = ds
        actual_sfreq = sf
    print(f"ダウンサンプル後: {len(ch_data[frontal_channels[0]])} samples, {actual_sfreq:.1f} Hz")

    # バンドパワー
    print("バンドパワー計算中...")
    ch_alpha = [compute_band_power(ch_data[ch], actual_sfreq, (8, 13)) for ch in frontal_channels]
    ch_beta = [compute_band_power(ch_data[ch], actual_sfreq, (13, 25)) for ch in frontal_channels]

    min_len = min(len(ch_alpha[0]), len(ch_alpha[1]), len(ch_beta[0]), len(ch_beta[1]))
    alpha = np.mean([a[:min_len] for a in ch_alpha], axis=0)
    beta = np.mean([b[:min_len] for b in ch_beta], axis=0)

    time_sec = np.arange(min_len) * 1.0 + 1.0

    # Alpha/Beta比
    ratio = alpha / (beta + 1e-12)
    ratio_smooth = pd.Series(ratio).rolling(4, center=True, min_periods=1).mean().values

    # キャリブレーション (60秒)
    calib_mask = time_sec <= CALIBRATION_SEC
    calib_ratio = ratio_smooth[calib_mask]
    calib_median = np.median(calib_ratio)
    calib_mad = np.median(np.abs(calib_ratio - calib_median))
    calib_mean = np.mean(calib_ratio)
    calib_std = np.std(calib_ratio)

    print(f"\nキャリブレーション (0-{CALIBRATION_SEC}s):")
    print(f"  median={calib_median:.3f}, MAD={calib_mad:.3f}")
    print(f"  mean={calib_mean:.3f}, std={calib_std:.3f}")

    # セッション期間
    med_mask = time_sec > CALIBRATION_SEC
    med_time = time_sec[med_mask]
    med_ratio = ratio_smooth[med_mask]

    print(f"  瞑想期間: {med_time[0]:.0f}s - {med_time[-1]:.0f}s ({len(med_time)}ポイント)")
    print(f"  瞑想期間 ratio: median={np.median(med_ratio):.3f}, "
          f"mean={np.mean(med_ratio):.3f}, std={np.std(med_ratio):.3f}")

    # --------------------------------------------------
    # Z-score正規化版 (キャリブレーションで正規化)
    # --------------------------------------------------
    ratio_zscore = (ratio_smooth - calib_mean) / (calib_std + 1e-12)
    med_zscore = ratio_zscore[med_mask]

    print(f"\n  Z-score版 瞑想期間: median={np.median(med_zscore):.3f}, "
          f"mean={np.mean(med_zscore):.3f}, std={np.std(med_zscore):.3f}")

    # --------------------------------------------------
    # Mind State: MADベース探索 (60秒キャリブレーション)
    # --------------------------------------------------
    print(f"\n--- Mind State (MADベース, calib=60s) ---")
    best_ms_mad = {'error': float('inf')}
    for kc in np.arange(-2.0, 3.0, 0.05):
        for ka in np.arange(0.5, 6.0, 0.1):
            c_th = calib_median + kc * calib_mad
            a_th = calib_median - ka * calib_mad
            if c_th <= a_th:
                continue
            states = classify_mind_state(med_ratio, c_th, a_th)
            a_s = np.sum(states == 'Active')
            n_s = np.sum(states == 'Neutral')
            c_s = np.sum(states == 'Calm')
            err = abs(a_s - t['active_sec']) + abs(n_s - t['neutral_sec']) + abs(c_s - t['calm_sec'])
            if err < best_ms_mad['error']:
                best_ms_mad = {
                    'error': err, 'kc': kc, 'ka': ka,
                    'calm_th': c_th, 'active_th': a_th,
                    'active': a_s, 'neutral': n_s, 'calm': c_s,
                }

    print(f"  最適: kc={best_ms_mad['kc']:.2f}, ka={best_ms_mad['ka']:.2f}, err={best_ms_mad['error']}s")
    print(f"  A={best_ms_mad['active']}s(t={t['active_sec']}), "
          f"N={best_ms_mad['neutral']}s(t={t['neutral_sec']}), "
          f"C={best_ms_mad['calm']}s(t={t['calm_sec']})")

    # --------------------------------------------------
    # Mind State: Z-scoreベース探索
    # --------------------------------------------------
    print(f"\n--- Mind State (Z-scoreベース, calib=60s) ---")
    best_ms_z = {'error': float('inf')}
    for zc in np.arange(-3.0, 3.0, 0.05):
        for za in np.arange(0.1, 5.0, 0.1):
            c_th = zc
            a_th = -za
            if c_th <= a_th:
                continue
            states = classify_mind_state(med_zscore, c_th, a_th)
            a_s = np.sum(states == 'Active')
            n_s = np.sum(states == 'Neutral')
            c_s = np.sum(states == 'Calm')
            err = abs(a_s - t['active_sec']) + abs(n_s - t['neutral_sec']) + abs(c_s - t['calm_sec'])
            if err < best_ms_z['error']:
                best_ms_z = {
                    'error': err, 'zc': zc, 'za': za,
                    'calm_th': zc, 'active_th': -za,
                    'active': a_s, 'neutral': n_s, 'calm': c_s,
                }

    print(f"  最適: calm_z={best_ms_z['zc']:.2f}, active_z=-{best_ms_z['za']:.2f}, err={best_ms_z['error']}s")
    print(f"  A={best_ms_z['active']}s(t={t['active_sec']}), "
          f"N={best_ms_z['neutral']}s(t={t['neutral_sec']}), "
          f"C={best_ms_z['calm']}s(t={t['calm_sec']})")

    # --------------------------------------------------
    # Birds検出: 各仮説
    # --------------------------------------------------
    print(f"\n--- Birds (target={t['birds']}) ---")

    # 仮説A: パーセンタイル + 持続時間
    best_A = {'error': float('inf')}
    for pct in np.arange(20, 95, 1):
        th = np.percentile(med_ratio, pct)
        for dur in np.arange(1, 30, 0.5):
            evts = count_sustained_events(med_ratio, th, med_time, dur)
            err = abs(len(evts) - t['birds'])
            if err < best_A['error'] or (err == best_A['error'] and dur > best_A.get('dur', 0)):
                best_A = {'error': err, 'pct': pct, 'th': th, 'dur': dur, 'n': len(evts)}
    print(f"  仮説A: pct={best_A['pct']}, dur={best_A['dur']}s → {best_A['n']} (err={best_A['error']})")

    # 仮説B: キャリブレーション基準 + 持続時間
    best_B = {'error': float('inf')}
    for k in np.arange(-3.0, 5.0, 0.1):
        th = calib_median + k * calib_mad
        for dur in np.arange(1, 30, 0.5):
            evts = count_sustained_events(med_ratio, th, med_time, dur)
            err = abs(len(evts) - t['birds'])
            if err < best_B['error'] or (err == best_B['error'] and dur > best_B.get('dur', 0)):
                best_B = {'error': err, 'k': k, 'th': th, 'dur': dur, 'n': len(evts)}
    print(f"  仮説B: k={best_B['k']:.1f}, dur={best_B['dur']}s → {best_B['n']} (err={best_B['error']})")

    # 仮説Bz: Z-score基準 + 持続時間
    best_Bz = {'error': float('inf')}
    for z_th in np.arange(-3.0, 5.0, 0.1):
        for dur in np.arange(1, 30, 0.5):
            evts = count_sustained_events(med_zscore, z_th, med_time, dur)
            err = abs(len(evts) - t['birds'])
            if err < best_Bz['error'] or (err == best_Bz['error'] and dur > best_Bz.get('dur', 0)):
                best_Bz = {'error': err, 'z': z_th, 'dur': dur, 'n': len(evts)}
    print(f"  仮説Bz: z={best_Bz['z']:.1f}, dur={best_Bz['dur']}s → {best_Bz['n']} (err={best_Bz['error']})")

    # 仮説D: Calm内 N秒/bird (best_ms_madのCalm使用)
    states_mad = classify_mind_state(med_ratio, best_ms_mad['calm_th'], best_ms_mad['active_th'])
    calm_binary = (states_mad == 'Calm').astype(int)
    segs = []
    in_s = False
    st = None
    for i in range(len(calm_binary)):
        if calm_binary[i] and not in_s:
            in_s = True; st = i
        elif not calm_binary[i] and in_s:
            in_s = False; segs.append((i - st) * 1.0)
    if in_s:
        segs.append((len(calm_binary) - st) * 1.0)

    best_D = {'error': float('inf')}
    for base in np.arange(3, 80, 0.5):
        nb = sum(int(s / base) for s in segs)
        err = abs(nb - t['birds'])
        if err < best_D['error']:
            best_D = {'error': err, 'base': base, 'n': nb}
    print(f"  仮説D: base={best_D['base']:.1f}s → {best_D['n']} (err={best_D['error']})")

    # 仮説Dz: Z-score CalmのN秒/bird
    states_z = classify_mind_state(med_zscore, best_ms_z['calm_th'], best_ms_z['active_th'])
    calm_z_binary = (states_z == 'Calm').astype(int)
    segs_z = []
    in_s = False
    st = None
    for i in range(len(calm_z_binary)):
        if calm_z_binary[i] and not in_s:
            in_s = True; st = i
        elif not calm_z_binary[i] and in_s:
            in_s = False; segs_z.append((i - st) * 1.0)
    if in_s:
        segs_z.append((len(calm_z_binary) - st) * 1.0)

    best_Dz = {'error': float('inf')}
    for base in np.arange(3, 80, 0.5):
        nb = sum(int(s / base) for s in segs_z)
        err = abs(nb - t['birds'])
        if err < best_Dz['error']:
            best_Dz = {'error': err, 'base': base, 'n': nb}
    print(f"  仮説Dz: base={best_Dz['base']:.1f}s → {best_Dz['n']} (err={best_Dz['error']})")

    # --------------------------------------------------
    # Recoveries
    # --------------------------------------------------
    print(f"\n--- Recoveries (target={t['recoveries']}) ---")

    # MADベース
    best_rec = {'error': float('inf')}
    for lb in [3, 5, 7, 10]:
        for lf in [3, 5, 7, 10]:
            for d_pct in np.arange(10, 45, 5):
                for r_pct in np.arange(50, 90, 5):
                    d_th = np.percentile(med_ratio, d_pct)
                    r_th = np.percentile(med_ratio, r_pct)
                    if r_th <= d_th:
                        continue
                    recs = []
                    gap = 30
                    for i in range(lb, len(med_ratio) - lf):
                        pre = np.mean(med_ratio[i-lb:i])
                        post = np.mean(med_ratio[i:i+lf])
                        if pre <= d_th and post >= r_th:
                            if not recs or (i - recs[-1]) >= gap:
                                recs.append(i)
                    err = abs(len(recs) - t['recoveries'])
                    if err < best_rec['error']:
                        best_rec = {
                            'error': err, 'lb': lb, 'lf': lf,
                            'd_pct': d_pct, 'r_pct': r_pct,
                            'n': len(recs), 'indices': recs,
                        }
    print(f"  Raw: lb={best_rec['lb']}s, lf={best_rec['lf']}s, "
          f"drop={best_rec['d_pct']}%ile, rise={best_rec['r_pct']}%ile → {best_rec['n']}")

    # Z-scoreベース
    best_rec_z = {'error': float('inf')}
    for lb in [3, 5, 7, 10]:
        for lf in [3, 5, 7, 10]:
            for d_z in np.arange(-3.0, 0.0, 0.2):
                for r_z in np.arange(0.0, 3.0, 0.2):
                    recs = []
                    gap = 30
                    for i in range(lb, len(med_zscore) - lf):
                        pre = np.mean(med_zscore[i-lb:i])
                        post = np.mean(med_zscore[i:i+lf])
                        if pre <= d_z and post >= r_z:
                            if not recs or (i - recs[-1]) >= gap:
                                recs.append(i)
                    err = abs(len(recs) - t['recoveries'])
                    if err < best_rec_z['error']:
                        best_rec_z = {
                            'error': err, 'lb': lb, 'lf': lf,
                            'd_z': d_z, 'r_z': r_z,
                            'n': len(recs), 'indices': recs,
                        }
    print(f"  Z-score: lb={best_rec_z['lb']}s, lf={best_rec_z['lf']}s, "
          f"drop={best_rec_z['d_z']:.1f}, rise={best_rec_z['r_z']:.1f} → {best_rec_z['n']}")

    results[sn] = {
        'calib_median': calib_median, 'calib_mad': calib_mad,
        'calib_mean': calib_mean, 'calib_std': calib_std,
        'med_ratio': med_ratio, 'med_zscore': med_zscore, 'med_time': med_time,
        'ratio_smooth': ratio_smooth, 'time_sec': time_sec,
        'alpha': alpha, 'beta': beta,
        'best_ms_mad': best_ms_mad, 'best_ms_z': best_ms_z,
        'best_birds_A': best_A, 'best_birds_B': best_B, 'best_birds_Bz': best_Bz,
        'best_birds_D': best_D, 'best_birds_Dz': best_Dz,
        'best_rec': best_rec, 'best_rec_z': best_rec_z,
        'target': t,
    }

# ============================================================
# 共通パラメータ探索
# ============================================================
print(f"\n{'='*70}")
print("  共通パラメータ探索")
print(f"{'='*70}")

r1, r2 = results['session1'], results['session2']

# ■ Mind State: Z-scoreベース共通
print("\n■ Mind State (Z-score共通パラメータ)")
best_common_z = {'total_error': float('inf')}
for zc in np.arange(-3.0, 3.0, 0.05):
    for za in np.arange(0.1, 5.0, 0.1):
        total_err = 0
        sr = {}
        for sn, r in results.items():
            states = classify_mind_state(r['med_zscore'], zc, -za)
            a_s = np.sum(states == 'Active')
            n_s = np.sum(states == 'Neutral')
            c_s = np.sum(states == 'Calm')
            err = abs(a_s - r['target']['active_sec']) + abs(n_s - r['target']['neutral_sec']) + abs(c_s - r['target']['calm_sec'])
            total_err += err
            sr[sn] = {'a': a_s, 'n': n_s, 'c': c_s, 'err': err}
        if total_err < best_common_z['total_error']:
            best_common_z = {'total_error': total_err, 'zc': zc, 'za': za, 'sessions': sr}

print(f"  calm_z={best_common_z['zc']:.2f}, active_z=-{best_common_z['za']:.2f}, total_err={best_common_z['total_error']}s")
for sn, sr in best_common_z['sessions'].items():
    t = results[sn]['target']
    print(f"  {sn}: A={sr['a']}s(t={t['active_sec']}), N={sr['n']}s(t={t['neutral_sec']}), C={sr['c']}s(t={t['calm_sec']}), err={sr['err']}s")

# ■ Mind State: MADベース共通
print("\n■ Mind State (MADベース共通パラメータ)")
best_common_mad = {'total_error': float('inf')}
for kc in np.arange(-2.0, 3.0, 0.05):
    for ka in np.arange(0.5, 6.0, 0.1):
        total_err = 0
        sr = {}
        for sn, r in results.items():
            c_th = r['calib_median'] + kc * r['calib_mad']
            a_th = r['calib_median'] - ka * r['calib_mad']
            if c_th <= a_th:
                total_err = float('inf'); break
            states = classify_mind_state(r['med_ratio'], c_th, a_th)
            a_s = np.sum(states == 'Active')
            n_s = np.sum(states == 'Neutral')
            c_s = np.sum(states == 'Calm')
            err = abs(a_s - r['target']['active_sec']) + abs(n_s - r['target']['neutral_sec']) + abs(c_s - r['target']['calm_sec'])
            total_err += err
            sr[sn] = {'a': a_s, 'n': n_s, 'c': c_s, 'err': err}
        if total_err < best_common_mad['total_error']:
            best_common_mad = {'total_error': total_err, 'kc': kc, 'ka': ka, 'sessions': sr}

print(f"  kc={best_common_mad['kc']:.2f}, ka={best_common_mad['ka']:.2f}, total_err={best_common_mad['total_error']}s")
for sn, sr in best_common_mad['sessions'].items():
    t = results[sn]['target']
    print(f"  {sn}: A={sr['a']}s(t={t['active_sec']}), N={sr['n']}s(t={t['neutral_sec']}), C={sr['c']}s(t={t['calm_sec']}), err={sr['err']}s")

# ■ Birds: Z-score + 持続時間 共通
print("\n■ Birds (Z-score + 持続時間 共通パラメータ)")
best_common_bz = {'total_error': float('inf')}
for z_th in np.arange(-3.0, 5.0, 0.1):
    for dur in np.arange(1, 30, 0.5):
        total_err = 0
        sr = {}
        for sn, r in results.items():
            evts = count_sustained_events(r['med_zscore'], z_th, r['med_time'], dur)
            err = abs(len(evts) - r['target']['birds'])
            total_err += err
            sr[sn] = len(evts)
        if total_err < best_common_bz['total_error'] or (
            total_err == best_common_bz['total_error'] and dur > best_common_bz.get('dur', 0)
        ):
            best_common_bz = {'total_error': total_err, 'z': z_th, 'dur': dur, 'sessions': sr}

print(f"  z_th={best_common_bz['z']:.1f}, dur={best_common_bz['dur']}s, total_err={best_common_bz['total_error']}")
for sn, n in best_common_bz['sessions'].items():
    print(f"  {sn}: {n} birds (target={results[sn]['target']['birds']})")

# ■ Birds: Z-score Calm内 N秒/bird 共通
print("\n■ Birds (Z-score Calm内 N秒/bird 共通)")
zc_common = best_common_z['zc']
za_common = best_common_z['za']
best_common_dz = {'total_error': float('inf')}
for base in np.arange(3, 80, 0.5):
    total_err = 0
    sr = {}
    for sn, r in results.items():
        states = classify_mind_state(r['med_zscore'], zc_common, -za_common)
        cb = (states == 'Calm').astype(int)
        segs = []
        in_s = False; st = None
        for i in range(len(cb)):
            if cb[i] and not in_s: in_s = True; st = i
            elif not cb[i] and in_s:
                in_s = False; segs.append((i - st) * 1.0)
        if in_s: segs.append((len(cb) - st) * 1.0)
        nb = sum(int(s / base) for s in segs)
        err = abs(nb - r['target']['birds'])
        total_err += err
        sr[sn] = nb
    if total_err < best_common_dz['total_error']:
        best_common_dz = {'total_error': total_err, 'base': base, 'sessions': sr}

print(f"  base={best_common_dz['base']:.1f}s, total_err={best_common_dz['total_error']}")
for sn, n in best_common_dz['sessions'].items():
    print(f"  {sn}: {n} birds (target={results[sn]['target']['birds']})")

# ■ Recoveries: Z-score共通
print("\n■ Recoveries (Z-score共通パラメータ)")
best_common_rz = {'total_error': float('inf')}
for lb in [3, 5, 7, 10]:
    for lf in [3, 5, 7, 10]:
        for d_z in np.arange(-3.0, 0.0, 0.2):
            for r_z in np.arange(0.0, 3.0, 0.2):
                total_err = 0
                sr = {}
                for sn, r in results.items():
                    recs = []
                    gap = 30
                    for i in range(lb, len(r['med_zscore']) - lf):
                        pre = np.mean(r['med_zscore'][i-lb:i])
                        post = np.mean(r['med_zscore'][i:i+lf])
                        if pre <= d_z and post >= r_z:
                            if not recs or (i - recs[-1]) >= gap:
                                recs.append(i)
                    err = abs(len(recs) - r['target']['recoveries'])
                    total_err += err
                    sr[sn] = len(recs)
                if total_err < best_common_rz['total_error']:
                    best_common_rz = {
                        'total_error': total_err, 'lb': lb, 'lf': lf,
                        'd_z': d_z, 'r_z': r_z, 'sessions': sr,
                    }

print(f"  lb={best_common_rz['lb']}s, lf={best_common_rz['lf']}s, "
      f"drop={best_common_rz['d_z']:.1f}, rise={best_common_rz['r_z']:.1f}, "
      f"total_err={best_common_rz['total_error']}")
for sn, n in best_common_rz['sessions'].items():
    print(f"  {sn}: {n} recoveries (target={results[sn]['target']['recoveries']})")

# ■ Recoveries: %ile共通
print("\n■ Recoveries (%ile共通パラメータ)")
best_common_rp = {'total_error': float('inf')}
for lb in [3, 5, 7, 10]:
    for lf in [3, 5, 7, 10]:
        for d_pct in np.arange(10, 45, 5):
            for r_pct in np.arange(50, 90, 5):
                total_err = 0
                sr = {}
                for sn, r in results.items():
                    d_th = np.percentile(r['med_ratio'], d_pct)
                    r_th = np.percentile(r['med_ratio'], r_pct)
                    if r_th <= d_th:
                        total_err = float('inf'); break
                    recs = []
                    gap = 30
                    for i in range(lb, len(r['med_ratio']) - lf):
                        pre = np.mean(r['med_ratio'][i-lb:i])
                        post = np.mean(r['med_ratio'][i:i+lf])
                        if pre <= d_th and post >= r_th:
                            if not recs or (i - recs[-1]) >= gap:
                                recs.append(i)
                    err = abs(len(recs) - r['target']['recoveries'])
                    total_err += err
                    sr[sn] = len(recs)
                if total_err < best_common_rp['total_error']:
                    best_common_rp = {
                        'total_error': total_err, 'lb': lb, 'lf': lf,
                        'd_pct': d_pct, 'r_pct': r_pct, 'sessions': sr,
                    }

print(f"  lb={best_common_rp['lb']}s, lf={best_common_rp['lf']}s, "
      f"drop={best_common_rp['d_pct']}%ile, rise={best_common_rp['r_pct']}%ile, "
      f"total_err={best_common_rp['total_error']}")
for sn, n in best_common_rp['sessions'].items():
    print(f"  {sn}: {n} recoveries (target={results[sn]['target']['recoveries']})")


# ============================================================
# 比較グラフ
# ============================================================
print("\n比較グラフ生成中...")
fig, axes = plt.subplots(4, 2, figsize=(20, 16))

for col, (sn, r) in enumerate(results.items()):
    t = r['target']
    med_time_min = r['med_time'] / 60

    # Z-score時系列
    ax = axes[0, col]
    smoothed = pd.Series(r['med_zscore']).rolling(5, center=True, min_periods=1).mean()
    ax.plot(med_time_min, smoothed, color='blue', linewidth=1)
    ax.axhline(best_common_z['zc'], color='green', ls='--', label=f'Calm z={best_common_z["zc"]:.2f}')
    ax.axhline(-best_common_z['za'], color='red', ls='--', label=f'Active z=-{best_common_z["za"]:.2f}')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_title(f'{sn}: Z-score Alpha/Beta (score={t["score"]})')
    ax.set_ylabel('Z-score')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mind State (Z-score)
    ax = axes[1, col]
    states = classify_mind_state(r['med_zscore'], best_common_z['zc'], -best_common_z['za'])
    for state, color in [('Active', 'red'), ('Neutral', 'gray'), ('Calm', 'green')]:
        ax.fill_between(med_time_min, 0, 1, where=(states == state), alpha=0.5, color=color, label=state)
    a_s = np.sum(states == 'Active'); n_s = np.sum(states == 'Neutral'); c_s = np.sum(states == 'Calm')
    ax.set_title(f'{sn}: Mind State Z (A={a_s}s/t={t["active_sec"]}, N={n_s}s/t={t["neutral_sec"]}, C={c_s}s/t={t["calm_sec"]})')
    ax.set_yticks([])
    ax.legend(fontsize=8)

    # Raw ratio
    ax = axes[2, col]
    raw_smoothed = pd.Series(r['med_ratio']).rolling(5, center=True, min_periods=1).mean()
    ax.plot(med_time_min, raw_smoothed, color='purple', linewidth=1)
    ax.set_title(f'{sn}: Raw Alpha/Beta Ratio (median={np.median(r["med_ratio"]):.2f})')
    ax.set_ylabel('Ratio')
    ax.grid(True, alpha=0.3)

    # Z-score distribution
    ax = axes[3, col]
    ax.hist(r['med_zscore'], bins=80, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(best_common_z['zc'], color='green', lw=2, label='Calm th')
    ax.axvline(-best_common_z['za'], color='red', lw=2, label='Active th')
    ax.axvline(0, color='gray', ls=':', lw=2, label='Baseline (z=0)')
    ax.set_title(f'{sn}: Z-score Distribution')
    ax.set_xlabel('Z-score')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = OUTPUT_DIR / 'cross_session_v2.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"グラフ保存: {fig_path}")
plt.close()


# ============================================================
# 最終結論
# ============================================================
print(f"\n{'='*70}")
print("  最終アルゴリズム推定 (v2, calib=60s)")
print(f"{'='*70}")

print(f"""
=== 正規化方式 ===
Z-scoreベース (calib mean/stdで正規化) vs MADベース比較:
  Z-score共通 total_err = {best_common_z['total_error']}s
  MAD共通     total_err = {best_common_mad['total_error']}s

=== Mind State ===
Z-score版:
  calm_z  = {best_common_z['zc']:.2f}  (ratio > mean + {best_common_z['zc']:.2f}*std → Calm)
  active_z = -{best_common_z['za']:.2f}  (ratio < mean - {best_common_z['za']:.2f}*std → Active)
MAD版:
  kc = {best_common_mad['kc']:.2f}  (ratio > median + kc*MAD → Calm)
  ka = {best_common_mad['ka']:.2f}  (ratio < median - ka*MAD → Active)

=== Birds ===
Z-score持続: z_th={best_common_bz['z']:.1f}, dur={best_common_bz['dur']}s, err={best_common_bz['total_error']}
Z-score Calm内/bird: base={best_common_dz['base']:.1f}s, err={best_common_dz['total_error']}

=== Recoveries ===
Z-score: lb={best_common_rz['lb']}s, lf={best_common_rz['lf']}s, drop={best_common_rz['d_z']:.1f}, rise={best_common_rz['r_z']:.1f}, err={best_common_rz['total_error']}
%ile:    lb={best_common_rp['lb']}s, lf={best_common_rp['lf']}s, drop={best_common_rp['d_pct']}%ile, rise={best_common_rp['r_pct']}%ile, err={best_common_rp['total_error']}
""")
