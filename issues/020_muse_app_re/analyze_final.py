#!/usr/bin/env python
"""
最終クロスセッション分析

Session1: 15min, calib=120s (graph starts at 2:00)
Session2: 10min, calib=60s  (graph starts at 1:00)

仮説: キャリブレーション = セッション時間に依存（短いセッションほど短い）
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
        'calib_sec': 120,
        'target': {
            'birds': 5, 'recoveries': 3,
            'active_sec': 19, 'neutral_sec': 398, 'calm_sec': 369,
            'calm_pct': 46, 'score': 46, 'duration_min': 15,
        },
    },
    'session2': {
        'csv': 'session2/muse_app_2026-03-04--17-46-58.csv',
        'calib_sec': 60,
        'target': {
            'birds': 29, 'recoveries': 3,
            'active_sec': 5, 'neutral_sec': 139, 'calm_sec': 403,
            'calm_pct': 73, 'score': 73, 'duration_min': 10,
        },
    },
}

TARGET_SFREQ = 51.5
frontal_channels = ['RAW_AF7', 'RAW_AF8']


def downsample_with_aa(data, original_sfreq, target_sfreq):
    ratio = int(round(original_sfreq / target_sfreq))
    if ratio <= 1:
        return data, original_sfreq
    nyq = original_sfreq / 2
    cutoff = (target_sfreq / 2) * 0.9
    sos = signal.butter(8, cutoff / nyq, btype='low', output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    return filtered[::ratio], original_sfreq / ratio


def compute_band_power(data, sfreq, band, window_sec=2.0, step_sec=1.0):
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
                    'start_sec': time_vec[event_start],
                    'end_sec': time_vec[i - 1],
                    'duration': dur,
                })
    if in_event:
        dur = time_vec[-1] - time_vec[event_start]
        if dur >= min_duration_sec:
            events.append({
                'start_sec': time_vec[event_start],
                'end_sec': time_vec[-1],
                'duration': dur,
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
    print(f"  {sn.upper()} (calib={si['calib_sec']}s)")
    print(f"{'='*70}")

    df = pd.read_csv(OUTPUT_DIR / si['csv'])
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    t = si['target']
    calib_sec = si['calib_sec']

    duration_s = (df['TimeStamp'].iloc[-1] - df['TimeStamp'].iloc[0]).total_seconds()
    orig_sfreq = len(df) / duration_s
    print(f"元: {len(df)} samples, {duration_s:.1f}s, {orig_sfreq:.1f} Hz")

    # ダウンサンプル
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

    # Raw EEG統計（デバッグ用）
    for ch in frontal_channels:
        vals = ch_data[ch]
        print(f"  {ch}: mean={np.mean(vals):.1f}, std={np.std(vals):.1f}, range=[{np.min(vals):.1f}, {np.max(vals):.1f}]")
    print(f"  Alpha power: mean={np.mean(alpha):.3f}, median={np.median(alpha):.3f}")
    print(f"  Beta power:  mean={np.mean(beta):.3f}, median={np.median(beta):.3f}")

    # Alpha/Beta比
    ratio = alpha / (beta + 1e-12)
    ratio_smooth = pd.Series(ratio).rolling(4, center=True, min_periods=1).mean().values

    # キャリブレーション
    calib_mask = time_sec <= calib_sec
    calib_ratio = ratio_smooth[calib_mask]
    calib_median = np.median(calib_ratio)
    calib_mad = np.median(np.abs(calib_ratio - calib_median))
    calib_mean = np.mean(calib_ratio)
    calib_std = np.std(calib_ratio)

    print(f"\nキャリブレーション (0-{calib_sec}s):")
    print(f"  median={calib_median:.3f}, MAD={calib_mad:.3f}")
    print(f"  mean={calib_mean:.3f}, std={calib_std:.3f}")

    # セッション期間
    med_mask = time_sec > calib_sec
    med_time = time_sec[med_mask]
    med_ratio = ratio_smooth[med_mask]

    # Z-score
    ratio_zscore = (ratio_smooth - calib_mean) / (calib_std + 1e-12)
    med_zscore = ratio_zscore[med_mask]

    session_secs = len(med_ratio)
    print(f"  瞑想期間: {session_secs}s = {session_secs/60:.1f}min")
    print(f"  Target total: {t['active_sec'] + t['neutral_sec'] + t['calm_sec']}s")

    # --------------------------------------------------
    # Mind State: MADベース
    # --------------------------------------------------
    print(f"\n--- Mind State (MADベース) ---")
    best_mad = {'error': float('inf')}
    for kc in np.arange(-3.0, 3.0, 0.05):
        for ka in np.arange(0.5, 8.0, 0.1):
            c_th = calib_median + kc * calib_mad
            a_th = calib_median - ka * calib_mad
            if c_th <= a_th:
                continue
            states = classify_mind_state(med_ratio, c_th, a_th)
            a_s = np.sum(states == 'Active')
            n_s = np.sum(states == 'Neutral')
            c_s = np.sum(states == 'Calm')
            err = abs(a_s - t['active_sec']) + abs(n_s - t['neutral_sec']) + abs(c_s - t['calm_sec'])
            if err < best_mad['error']:
                best_mad = {
                    'error': err, 'kc': kc, 'ka': ka,
                    'calm_th': c_th, 'active_th': a_th,
                    'active': a_s, 'neutral': n_s, 'calm': c_s,
                }

    print(f"  kc={best_mad['kc']:.2f}, ka={best_mad['ka']:.2f}, err={best_mad['error']}s")
    print(f"  A={best_mad['active']}s(t={t['active_sec']}), N={best_mad['neutral']}s(t={t['neutral_sec']}), C={best_mad['calm']}s(t={t['calm_sec']})")
    print(f"  Calm th={best_mad['calm_th']:.3f}, Active th={best_mad['active_th']:.3f}")

    # Mind State: Z-scoreベース
    print(f"\n--- Mind State (Z-scoreベース) ---")
    best_z = {'error': float('inf')}
    for zc in np.arange(-3.0, 3.0, 0.05):
        for za in np.arange(0.1, 6.0, 0.1):
            states = classify_mind_state(med_zscore, zc, -za)
            a_s = np.sum(states == 'Active')
            n_s = np.sum(states == 'Neutral')
            c_s = np.sum(states == 'Calm')
            err = abs(a_s - t['active_sec']) + abs(n_s - t['neutral_sec']) + abs(c_s - t['calm_sec'])
            if err < best_z['error']:
                best_z = {
                    'error': err, 'zc': zc, 'za': za,
                    'active': a_s, 'neutral': n_s, 'calm': c_s,
                }

    print(f"  calm_z={best_z['zc']:.2f}, active_z=-{best_z['za']:.2f}, err={best_z['error']}s")
    print(f"  A={best_z['active']}s(t={t['active_sec']}), N={best_z['neutral']}s(t={t['neutral_sec']}), C={best_z['calm']}s(t={t['calm_sec']})")

    # --------------------------------------------------
    # Birds
    # --------------------------------------------------
    print(f"\n--- Birds (target={t['birds']}) ---")

    # 仮説B: calib MAD基準
    best_B = {'error': float('inf')}
    for k in np.arange(-5.0, 5.0, 0.1):
        th = calib_median + k * calib_mad
        for dur in np.arange(1, 40, 0.5):
            evts = count_sustained_events(med_ratio, th, med_time, dur)
            err = abs(len(evts) - t['birds'])
            if err < best_B['error'] or (err == best_B['error'] and dur > best_B.get('dur', 0)):
                best_B = {'error': err, 'k': k, 'th': th, 'dur': dur, 'n': len(evts)}
    print(f"  仮説B (MAD): k={best_B['k']:.1f}, dur={best_B['dur']}s → {best_B['n']} (err={best_B['error']})")

    # 仮説Bz: Z-score基準
    best_Bz = {'error': float('inf')}
    for z_th in np.arange(-4.0, 5.0, 0.1):
        for dur in np.arange(1, 40, 0.5):
            evts = count_sustained_events(med_zscore, z_th, med_time, dur)
            err = abs(len(evts) - t['birds'])
            if err < best_Bz['error'] or (err == best_Bz['error'] and dur > best_Bz.get('dur', 0)):
                best_Bz = {'error': err, 'z': z_th, 'dur': dur, 'n': len(evts)}
    print(f"  仮説Bz (Z-score): z={best_Bz['z']:.1f}, dur={best_Bz['dur']}s → {best_Bz['n']} (err={best_Bz['error']})")

    # 仮説D: Calm N秒/bird (MADベース)
    states_for_birds = classify_mind_state(med_ratio, best_mad['calm_th'], best_mad['active_th'])
    segs = []
    in_s = False; st = None
    for i in range(len(states_for_birds)):
        if states_for_birds[i] == 'Calm' and not in_s: in_s = True; st = i
        elif states_for_birds[i] != 'Calm' and in_s:
            in_s = False; segs.append((i - st) * 1.0)
    if in_s: segs.append((len(states_for_birds) - st) * 1.0)

    best_D = {'error': float('inf')}
    for base in np.arange(3, 100, 0.5):
        nb = sum(int(s / base) for s in segs)
        err = abs(nb - t['birds'])
        if err < best_D['error']:
            best_D = {'error': err, 'base': base, 'n': nb, 'calm_total': sum(segs)}
    print(f"  仮説D (Calm/N秒): base={best_D['base']:.1f}s → {best_D['n']} "
          f"(err={best_D['error']}, calm_total={best_D['calm_total']:.0f}s)")

    # --------------------------------------------------
    # Recoveries
    # --------------------------------------------------
    print(f"\n--- Recoveries (target={t['recoveries']}) ---")
    best_rec_z = {'error': float('inf')}
    for lb in [3, 5, 7, 10]:
        for lf in [3, 5, 7, 10]:
            for d_z in np.arange(-3.0, 0.0, 0.1):
                for r_z in np.arange(-0.5, 3.0, 0.1):
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
                            'd_z': d_z, 'r_z': r_z, 'n': len(recs),
                            'indices': recs,
                        }
    print(f"  Z-score: lb={best_rec_z['lb']}s, lf={best_rec_z['lf']}s, "
          f"drop_z={best_rec_z['d_z']:.1f}, rise_z={best_rec_z['r_z']:.1f} → {best_rec_z['n']}")

    for i, idx in enumerate(best_rec_z['indices']):
        t_min = med_time[idx] / 60
        lb_i = best_rec_z['lb']
        lf_i = best_rec_z['lf']
        pre = np.mean(med_zscore[max(0,idx-lb_i):idx])
        post = np.mean(med_zscore[idx:min(len(med_zscore),idx+lf_i)])
        print(f"    R{i+1}: {t_min:.1f}min (z: {pre:.2f} → {post:.2f})")

    results[sn] = {
        'calib_sec': calib_sec,
        'calib_median': calib_median, 'calib_mad': calib_mad,
        'calib_mean': calib_mean, 'calib_std': calib_std,
        'med_ratio': med_ratio, 'med_zscore': med_zscore, 'med_time': med_time,
        'ratio_smooth': ratio_smooth, 'time_sec': time_sec,
        'alpha': alpha, 'beta': beta,
        'best_mad': best_mad, 'best_z': best_z,
        'best_B': best_B, 'best_Bz': best_Bz, 'best_D': best_D,
        'best_rec_z': best_rec_z,
        'target': t,
    }


# ============================================================
# 共通パラメータ探索
# ============================================================
print(f"\n{'='*70}")
print("  共通パラメータ探索 (各セッション独自のキャリブレーション)")
print(f"{'='*70}")

r1, r2 = results['session1'], results['session2']

# ■ Mind State MADベース共通
print("\n■ Mind State (MAD共通)")
best_cm = {'total_error': float('inf')}
for kc in np.arange(-3.0, 3.0, 0.05):
    for ka in np.arange(0.5, 8.0, 0.1):
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
        if total_err < best_cm['total_error']:
            best_cm = {'total_error': total_err, 'kc': kc, 'ka': ka, 'sessions': sr}

print(f"  kc={best_cm['kc']:.2f}, ka={best_cm['ka']:.2f}, total_err={best_cm['total_error']}s")
for sn, sr in best_cm['sessions'].items():
    t = results[sn]['target']
    print(f"  {sn}: A={sr['a']}s(t={t['active_sec']}), N={sr['n']}s(t={t['neutral_sec']}), C={sr['c']}s(t={t['calm_sec']}), err={sr['err']}s")

# ■ Mind State Z-score共通
print("\n■ Mind State (Z-score共通)")
best_cz = {'total_error': float('inf')}
for zc in np.arange(-3.0, 3.0, 0.05):
    for za in np.arange(0.1, 6.0, 0.1):
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
        if total_err < best_cz['total_error']:
            best_cz = {'total_error': total_err, 'zc': zc, 'za': za, 'sessions': sr}

print(f"  calm_z={best_cz['zc']:.2f}, active_z=-{best_cz['za']:.2f}, total_err={best_cz['total_error']}s")
for sn, sr in best_cz['sessions'].items():
    t = results[sn]['target']
    print(f"  {sn}: A={sr['a']}s(t={t['active_sec']}), N={sr['n']}s(t={t['neutral_sec']}), C={sr['c']}s(t={t['calm_sec']}), err={sr['err']}s")

# ■ Birds Z-score持続
print("\n■ Birds (Z-score持続 共通)")
best_cbz = {'total_error': float('inf')}
for z_th in np.arange(-4.0, 5.0, 0.1):
    for dur in np.arange(1, 40, 0.5):
        total_err = 0
        sr = {}
        for sn, r in results.items():
            evts = count_sustained_events(r['med_zscore'], z_th, r['med_time'], dur)
            err = abs(len(evts) - r['target']['birds'])
            total_err += err
            sr[sn] = len(evts)
        if total_err < best_cbz['total_error'] or (
            total_err == best_cbz['total_error'] and dur > best_cbz.get('dur', 0)
        ):
            best_cbz = {'total_error': total_err, 'z': z_th, 'dur': dur, 'sessions': sr}

print(f"  z={best_cbz['z']:.1f}, dur={best_cbz['dur']}s, total_err={best_cbz['total_error']}")
for sn, n in best_cbz['sessions'].items():
    print(f"  {sn}: {n} (target={results[sn]['target']['birds']})")

# ■ Birds MAD持続 共通
print("\n■ Birds (MAD持続 共通)")
best_cbm = {'total_error': float('inf')}
for k in np.arange(-5.0, 5.0, 0.1):
    for dur in np.arange(1, 40, 0.5):
        total_err = 0
        sr = {}
        for sn, r in results.items():
            th = r['calib_median'] + k * r['calib_mad']
            evts = count_sustained_events(r['med_ratio'], th, r['med_time'], dur)
            err = abs(len(evts) - r['target']['birds'])
            total_err += err
            sr[sn] = len(evts)
        if total_err < best_cbm['total_error'] or (
            total_err == best_cbm['total_error'] and dur > best_cbm.get('dur', 0)
        ):
            best_cbm = {'total_error': total_err, 'k': k, 'dur': dur, 'sessions': sr}

print(f"  k={best_cbm['k']:.1f}, dur={best_cbm['dur']}s, total_err={best_cbm['total_error']}")
for sn, n in best_cbm['sessions'].items():
    print(f"  {sn}: {n} (target={results[sn]['target']['birds']})")

# ■ Birds Calm/N秒 共通 (MAD best)
print("\n■ Birds (Calm/N秒 共通, 各セッション最適MS使用)")
best_cd = {'total_error': float('inf')}
for base in np.arange(3, 100, 0.5):
    total_err = 0
    sr = {}
    for sn, r in results.items():
        states = classify_mind_state(r['med_ratio'], r['best_mad']['calm_th'], r['best_mad']['active_th'])
        cb = (states == 'Calm').astype(int)
        segs = []
        in_s = False; st = None
        for i in range(len(cb)):
            if cb[i] and not in_s: in_s = True; st = i
            elif not cb[i] and in_s: in_s = False; segs.append((i - st) * 1.0)
        if in_s: segs.append((len(cb) - st) * 1.0)
        nb = sum(int(s / base) for s in segs)
        err = abs(nb - r['target']['birds'])
        total_err += err
        sr[sn] = nb
    if total_err < best_cd['total_error']:
        best_cd = {'total_error': total_err, 'base': base, 'sessions': sr}

print(f"  base={best_cd['base']:.1f}s, total_err={best_cd['total_error']}")
for sn, n in best_cd['sessions'].items():
    print(f"  {sn}: {n} (target={results[sn]['target']['birds']})")

# ■ Birds Calm/N秒 with 共通MS
print("\n■ Birds (Calm/N秒 共通, 共通MS使用)")
best_cd2 = {'total_error': float('inf')}
for base in np.arange(3, 100, 0.5):
    total_err = 0
    sr = {}
    for sn, r in results.items():
        c_th = r['calib_median'] + best_cm['kc'] * r['calib_mad']
        a_th = r['calib_median'] - best_cm['ka'] * r['calib_mad']
        states = classify_mind_state(r['med_ratio'], c_th, a_th)
        cb = (states == 'Calm').astype(int)
        segs = []
        in_s = False; st = None
        for i in range(len(cb)):
            if cb[i] and not in_s: in_s = True; st = i
            elif not cb[i] and in_s: in_s = False; segs.append((i - st) * 1.0)
        if in_s: segs.append((len(cb) - st) * 1.0)
        nb = sum(int(s / base) for s in segs)
        err = abs(nb - r['target']['birds'])
        total_err += err
        sr[sn] = nb
    if total_err < best_cd2['total_error']:
        best_cd2 = {'total_error': total_err, 'base': base, 'sessions': sr}

print(f"  base={best_cd2['base']:.1f}s, total_err={best_cd2['total_error']}")
for sn, n in best_cd2['sessions'].items():
    print(f"  {sn}: {n} (target={results[sn]['target']['birds']})")

# ■ Recoveries Z-score共通
print("\n■ Recoveries (Z-score共通)")
best_crz = {'total_error': float('inf')}
for lb in [3, 5, 7, 10]:
    for lf in [3, 5, 7, 10]:
        for d_z in np.arange(-3.0, 0.0, 0.1):
            for r_z in np.arange(-0.5, 3.0, 0.1):
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
                if total_err < best_crz['total_error']:
                    best_crz = {
                        'total_error': total_err, 'lb': lb, 'lf': lf,
                        'd_z': d_z, 'r_z': r_z, 'sessions': sr,
                    }

print(f"  lb={best_crz['lb']}s, lf={best_crz['lf']}s, drop_z={best_crz['d_z']:.1f}, rise_z={best_crz['r_z']:.1f}, total_err={best_crz['total_error']}")
for sn, n in best_crz['sessions'].items():
    print(f"  {sn}: {n} (target={results[sn]['target']['recoveries']})")


# ============================================================
# 比較グラフ
# ============================================================
print("\n比較グラフ生成中...")
fig, axes = plt.subplots(5, 2, figsize=(20, 22))

for col, (sn, r) in enumerate(results.items()):
    t = r['target']
    med_time_min = r['med_time'] / 60

    # Z-score時系列
    ax = axes[0, col]
    sm = pd.Series(r['med_zscore']).rolling(5, center=True, min_periods=1).mean()
    ax.plot(med_time_min, sm, color='blue', linewidth=1)
    ax.axhline(best_cz['zc'], color='green', ls='--', label=f'Calm z={best_cz["zc"]:.2f}')
    ax.axhline(-best_cz['za'], color='red', ls='--', label=f'Active z=-{best_cz["za"]:.2f}')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_title(f'{sn}: Z-score (calib={r["calib_sec"]}s, score={t["score"]})')
    ax.set_ylabel('Z-score')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mind State (MAD共通)
    ax = axes[1, col]
    c_th = r['calib_median'] + best_cm['kc'] * r['calib_mad']
    a_th = r['calib_median'] - best_cm['ka'] * r['calib_mad']
    states = classify_mind_state(r['med_ratio'], c_th, a_th)
    for state, color in [('Active', 'red'), ('Neutral', 'gray'), ('Calm', 'green')]:
        ax.fill_between(med_time_min, 0, 1, where=(states == state), alpha=0.5, color=color, label=state)
    a_s = np.sum(states == 'Active'); n_s = np.sum(states == 'Neutral'); c_s = np.sum(states == 'Calm')
    ax.set_title(f'{sn}: MAD MS (A={a_s}/t={t["active_sec"]}, N={n_s}/t={t["neutral_sec"]}, C={c_s}/t={t["calm_sec"]})')
    ax.set_yticks([]); ax.legend(fontsize=8)

    # Raw ratio
    ax = axes[2, col]
    raw_sm = pd.Series(r['med_ratio']).rolling(5, center=True, min_periods=1).mean()
    ax.plot(med_time_min, raw_sm, color='purple', linewidth=1)
    ax.axhline(r['calib_median'], color='blue', ls=':', label=f'calib median={r["calib_median"]:.2f}')
    ax.set_title(f'{sn}: Alpha/Beta Ratio')
    ax.set_ylabel('Ratio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # バンドパワー
    ax = axes[3, col]
    a_db = 10 * np.log10(r['alpha'] + 1e-12)
    b_db = 10 * np.log10(r['beta'] + 1e-12)
    tm = r['time_sec'] / 60
    ax.plot(tm, pd.Series(a_db).rolling(10, center=True, min_periods=1).mean(), color='green', label='Alpha')
    ax.plot(tm, pd.Series(b_db).rolling(10, center=True, min_periods=1).mean(), color='orange', label='Beta')
    ax.axvline(r['calib_sec']/60, color='gray', ls='--', alpha=0.5)
    ax.set_title(f'{sn}: Band Power'); ax.set_ylabel('dB'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Z-score分布
    ax = axes[4, col]
    ax.hist(r['med_zscore'], bins=80, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(best_cz['zc'], color='green', lw=2, label='Calm z')
    ax.axvline(-best_cz['za'], color='red', lw=2, label='Active z')
    ax.axvline(0, color='gray', ls=':', lw=2)
    ax.set_xlabel('Z-score'); ax.set_title(f'{sn}: Z-score Distribution')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cross_session_final.png', dpi=150, bbox_inches='tight')
print(f"グラフ保存: cross_session_final.png")
plt.close()


# ============================================================
# 最終結論
# ============================================================
print(f"\n{'='*70}")
print("  最終アルゴリズム推定")
print(f"{'='*70}")

print(f"""
■ キャリブレーション
  Session1(15min): median={r1['calib_median']:.3f}, MAD={r1['calib_mad']:.3f}, mean={r1['calib_mean']:.3f}, std={r1['calib_std']:.3f}
  Session2(10min):  median={r2['calib_median']:.3f}, MAD={r2['calib_mad']:.3f}, mean={r2['calib_mean']:.3f}, std={r2['calib_std']:.3f}

■ Mind State (2方式比較)
  MAD共通 (kc={best_cm['kc']:.2f}, ka={best_cm['ka']:.2f}): total_err={best_cm['total_error']}s
  Z-score共通 (zc={best_cz['zc']:.2f}, za={best_cz['za']:.2f}): total_err={best_cz['total_error']}s

  個別最適:
    S1 MAD: kc={r1['best_mad']['kc']:.2f}, ka={r1['best_mad']['ka']:.2f}, err={r1['best_mad']['error']}s
    S2 MAD: kc={r2['best_mad']['kc']:.2f}, ka={r2['best_mad']['ka']:.2f}, err={r2['best_mad']['error']}s
    S1 Z: zc={r1['best_z']['zc']:.2f}, za={r1['best_z']['za']:.2f}, err={r1['best_z']['error']}s
    S2 Z: zc={r2['best_z']['zc']:.2f}, za={r2['best_z']['za']:.2f}, err={r2['best_z']['error']}s

■ Birds (各方式比較)
  Z-score持続共通: z={best_cbz['z']:.1f}, dur={best_cbz['dur']}s, total_err={best_cbz['total_error']}
  MAD持続共通: k={best_cbm['k']:.1f}, dur={best_cbm['dur']}s, total_err={best_cbm['total_error']}
  Calm/N秒共通(各最適MS): base={best_cd['base']:.1f}s, total_err={best_cd['total_error']}
  Calm/N秒共通(共通MS): base={best_cd2['base']:.1f}s, total_err={best_cd2['total_error']}

  個別最適:
    S1 MAD: k={r1['best_B']['k']:.1f}, dur={r1['best_B']['dur']}s, err={r1['best_B']['error']}
    S2 MAD: k={r2['best_B']['k']:.1f}, dur={r2['best_B']['dur']}s, err={r2['best_B']['error']}
    S1 Calm/N: base={r1['best_D']['base']:.1f}s, calm={r1['best_D']['calm_total']:.0f}s
    S2 Calm/N: base={r2['best_D']['base']:.1f}s, calm={r2['best_D']['calm_total']:.0f}s

■ Recoveries
  Z-score共通: lb={best_crz['lb']}s, lf={best_crz['lf']}s, drop_z={best_crz['d_z']:.1f}, rise_z={best_crz['r_z']:.1f}, total_err={best_crz['total_error']}
""")
