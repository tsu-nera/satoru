#!/usr/bin/env python
"""
クロスセッション分析 v3
- Session1, 2, 3を含む3セッション横断分析
- キャリブレーション = 60秒
- アンチエイリアスフィルタ付きダウンサンプル
- Z-score正規化
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
    'session3': {
        'csv': 'session3/muse_app_2026-03-06--08-05-43.csv',
        'target': {
            'birds': 87, 'recoveries': 8,
            'active_sec': 5, 'neutral_sec': 261, 'calm_sec': 890,
            'calm_pct': 76, 'score': 76, 'duration_min': 20,
        },
    },
}

CALIBRATION_SEC = 60
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
            in_event = True; event_start = i
        elif not above[i] and in_event:
            in_event = False
            dur = time_vec[i - 1] - time_vec[event_start]
            if dur >= min_duration_sec:
                events.append({'start_sec': time_vec[event_start], 'end_sec': time_vec[i-1], 'duration': dur})
    if in_event:
        dur = time_vec[-1] - time_vec[event_start]
        if dur >= min_duration_sec:
            events.append({'start_sec': time_vec[event_start], 'end_sec': time_vec[-1], 'duration': dur})
    return events


def classify_mind_state(ratio_arr, calm_th, active_th):
    states = np.full(len(ratio_arr), 'Neutral', dtype=object)
    states[ratio_arr >= calm_th] = 'Calm'
    states[ratio_arr <= active_th] = 'Active'
    return states


def get_calm_segments(states, step=1.0):
    calm_binary = (states == 'Calm').astype(int)
    segs = []
    in_s = False; st = None
    for i in range(len(calm_binary)):
        if calm_binary[i] and not in_s: in_s = True; st = i
        elif not calm_binary[i] and in_s:
            in_s = False; segs.append((i - st) * step)
    if in_s: segs.append((len(calm_binary) - st) * step)
    return segs


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
    seconds_with_data = df['TimeStamp'].apply(lambda x: x.floor('s')).nunique()
    print(f"  元: {len(df)} samples, {duration_s:.1f}s, {orig_sfreq:.1f} Hz")
    print(f"  データ品質: {seconds_with_data}/{int(duration_s)}秒 ({100*seconds_with_data/int(duration_s):.0f}%)")

    ch_data = {}
    actual_sfreq = None
    for ch in frontal_channels:
        raw = df[ch].values.astype(float)
        ds, sf = downsample_with_aa(raw, orig_sfreq, TARGET_SFREQ)
        ch_data[ch] = ds
        actual_sfreq = sf

    print(f"  ダウンサンプル後: {len(ch_data[frontal_channels[0]])} samples, {actual_sfreq:.1f} Hz")

    print("  バンドパワー計算中...")
    ch_alpha = [compute_band_power(ch_data[ch], actual_sfreq, (8, 13)) for ch in frontal_channels]
    ch_beta  = [compute_band_power(ch_data[ch], actual_sfreq, (13, 25)) for ch in frontal_channels]

    min_len = min(len(ch_alpha[0]), len(ch_alpha[1]), len(ch_beta[0]), len(ch_beta[1]))
    alpha = np.mean([a[:min_len] for a in ch_alpha], axis=0)
    beta  = np.mean([b[:min_len] for b in ch_beta],  axis=0)
    time_sec = np.arange(min_len) * 1.0 + 1.0

    ratio = alpha / (beta + 1e-12)
    ratio_smooth = pd.Series(ratio).rolling(4, center=True, min_periods=1).mean().values

    calib_mask   = time_sec <= CALIBRATION_SEC
    calib_ratio  = ratio_smooth[calib_mask]
    calib_median = np.median(calib_ratio)
    calib_mad    = np.median(np.abs(calib_ratio - calib_median))
    calib_mean   = np.mean(calib_ratio)
    calib_std    = np.std(calib_ratio)

    print(f"  calib: median={calib_median:.3f}, MAD={calib_mad:.3f}, mean={calib_mean:.3f}, std={calib_std:.3f}")

    ratio_zscore = (ratio_smooth - calib_mean) / (calib_std + 1e-12)

    med_mask   = time_sec > CALIBRATION_SEC
    med_time   = time_sec[med_mask]
    med_ratio  = ratio_smooth[med_mask]
    med_zscore = ratio_zscore[med_mask]

    print(f"  瞑想期間 zscore: median={np.median(med_zscore):.2f}, mean={np.mean(med_zscore):.2f}, std={np.std(med_zscore):.2f}")

    # --- Mind State 個別最適化 (MAD) ---
    best_ms_mad = {'error': float('inf')}
    for kc in np.arange(-2.0, 3.0, 0.05):
        for ka in np.arange(0.5, 6.0, 0.1):
            c_th = calib_median + kc * calib_mad
            a_th = calib_median - ka * calib_mad
            if c_th <= a_th: continue
            states = classify_mind_state(med_ratio, c_th, a_th)
            a_s = np.sum(states == 'Active')
            n_s = np.sum(states == 'Neutral')
            c_s = np.sum(states == 'Calm')
            err = abs(a_s - t['active_sec']) + abs(n_s - t['neutral_sec']) + abs(c_s - t['calm_sec'])
            if err < best_ms_mad['error']:
                best_ms_mad = {'error': err, 'kc': kc, 'ka': ka, 'calm_th': c_th, 'active_th': a_th,
                               'active': a_s, 'neutral': n_s, 'calm': c_s}

    # --- Mind State 個別最適化 (Z-score) ---
    best_ms_z = {'error': float('inf')}
    for zc in np.arange(-3.0, 3.0, 0.05):
        for za in np.arange(0.1, 5.0, 0.1):
            states = classify_mind_state(med_zscore, zc, -za)
            a_s = np.sum(states == 'Active')
            n_s = np.sum(states == 'Neutral')
            c_s = np.sum(states == 'Calm')
            err = abs(a_s - t['active_sec']) + abs(n_s - t['neutral_sec']) + abs(c_s - t['calm_sec'])
            if err < best_ms_z['error']:
                best_ms_z = {'error': err, 'zc': zc, 'za': za, 'active': a_s, 'neutral': n_s, 'calm': c_s}

    print(f"  Mind State: MAD kc={best_ms_mad['kc']:.2f},ka={best_ms_mad['ka']:.2f} err={best_ms_mad['error']}s | "
          f"Z zc={best_ms_z['zc']:.2f},za={best_ms_z['za']:.2f} err={best_ms_z['error']}s")

    # --- Birds 個別最適化 ---
    # 仮説A: パーセンタイル + 持続時間
    best_A = {'error': float('inf')}
    for pct in np.arange(20, 95, 1):
        th = np.percentile(med_ratio, pct)
        for dur in np.arange(1, 30, 0.5):
            evts = count_sustained_events(med_ratio, th, med_time, dur)
            err = abs(len(evts) - t['birds'])
            if err < best_A['error'] or (err == best_A['error'] and dur > best_A.get('dur', 0)):
                best_A = {'error': err, 'pct': pct, 'th': th, 'dur': dur, 'n': len(evts)}

    # 仮説B: キャリブレーション基準 + 持続時間
    best_B = {'error': float('inf')}
    for k in np.arange(-3.0, 5.0, 0.1):
        th = calib_median + k * calib_mad
        for dur in np.arange(1, 30, 0.5):
            evts = count_sustained_events(med_ratio, th, med_time, dur)
            err = abs(len(evts) - t['birds'])
            if err < best_B['error'] or (err == best_B['error'] and dur > best_B.get('dur', 0)):
                best_B = {'error': err, 'k': k, 'th': th, 'dur': dur, 'n': len(evts)}

    # 仮説Bz: Z-score基準 + 持続時間
    best_Bz = {'error': float('inf')}
    for z_th in np.arange(-3.0, 5.0, 0.1):
        for dur in np.arange(1, 30, 0.5):
            evts = count_sustained_events(med_zscore, z_th, med_time, dur)
            err = abs(len(evts) - t['birds'])
            if err < best_Bz['error'] or (err == best_Bz['error'] and dur > best_Bz.get('dur', 0)):
                best_Bz = {'error': err, 'z': z_th, 'dur': dur, 'n': len(evts)}

    # 仮説D: Calm内 N秒/bird (MAD最適Calm状態を使用)
    states_mad = classify_mind_state(med_ratio, best_ms_mad['calm_th'], best_ms_mad['active_th'])
    segs = get_calm_segments(states_mad)
    best_D = {'error': float('inf')}
    for base in np.arange(3, 80, 0.5):
        nb = sum(int(s / base) for s in segs)
        err = abs(nb - t['birds'])
        if err < best_D['error']:
            best_D = {'error': err, 'base': base, 'n': nb}

    print(f"  Birds(t={t['birds']}): A={best_A['n']}(err={best_A['error']}) | "
          f"B={best_B['n']}(err={best_B['error']}) | "
          f"Bz={best_Bz['n']}(err={best_Bz['error']}) | "
          f"D={best_D['n']}(err={best_D['error']},base={best_D['base']:.1f}s)")

    # --- Recoveries 個別最適化 ---
    best_rec_z = {'error': float('inf')}
    for lb in [3, 5, 7, 10]:
        for lf in [3, 5, 7, 10]:
            for d_z in np.arange(-3.0, 0.5, 0.2):
                for r_z in np.arange(-0.5, 3.0, 0.2):
                    recs = []
                    for i in range(lb, len(med_zscore) - lf):
                        pre  = np.mean(med_zscore[i-lb:i])
                        post = np.mean(med_zscore[i:i+lf])
                        if pre <= d_z and post >= r_z and r_z > d_z:
                            if not recs or (i - recs[-1]) >= 30:
                                recs.append(i)
                    err = abs(len(recs) - t['recoveries'])
                    if err < best_rec_z['error']:
                        best_rec_z = {'error': err, 'lb': lb, 'lf': lf, 'd_z': d_z, 'r_z': r_z,
                                      'n': len(recs), 'indices': recs}

    print(f"  Rec(t={t['recoveries']}): Z lb={best_rec_z['lb']}s,lf={best_rec_z['lf']}s,"
          f"drop={best_rec_z['d_z']:.1f},rise={best_rec_z['r_z']:.1f} → {best_rec_z['n']} (err={best_rec_z['error']})")

    results[sn] = {
        'calib_median': calib_median, 'calib_mad': calib_mad,
        'calib_mean': calib_mean, 'calib_std': calib_std,
        'med_ratio': med_ratio, 'med_zscore': med_zscore, 'med_time': med_time,
        'ratio_smooth': ratio_smooth, 'time_sec': time_sec,
        'alpha': alpha, 'beta': beta,
        'best_ms_mad': best_ms_mad, 'best_ms_z': best_ms_z,
        'best_birds_A': best_A, 'best_birds_B': best_B, 'best_birds_Bz': best_Bz,
        'best_birds_D': best_D,
        'best_rec_z': best_rec_z,
        'target': t,
    }


# ============================================================
# 共通パラメータ探索
# ============================================================
print(f"\n{'='*70}")
print("  共通パラメータ探索 (3セッション横断)")
print(f"{'='*70}")

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

# ■ Birds: 仮説D (Calm内 N秒/bird) - 共通パラメータ
print("\n■ Birds 仮説D (Z-score共通Calm内 N秒/bird)")
best_common_dz = {'total_error': float('inf')}
zc_common = best_common_z['zc']
za_common = best_common_z['za']
for base in np.arange(3, 80, 0.5):
    total_err = 0
    sr = {}
    for sn, r in results.items():
        states = classify_mind_state(r['med_zscore'], zc_common, -za_common)
        segs = get_calm_segments(states)
        nb = sum(int(s / base) for s in segs)
        err = abs(nb - r['target']['birds'])
        total_err += err
        sr[sn] = nb
    if total_err < best_common_dz['total_error']:
        best_common_dz = {'total_error': total_err, 'base': base, 'sessions': sr}

print(f"  base={best_common_dz['base']:.1f}s/bird, total_err={best_common_dz['total_error']}")
for sn, n in best_common_dz['sessions'].items():
    print(f"  {sn}: {n} birds (target={results[sn]['target']['birds']})")

# ■ Birds: 仮説D - MADベース共通Calm使用
print("\n■ Birds 仮説D (MADベース共通Calm内 N秒/bird)")
best_common_d_mad = {'total_error': float('inf')}
kc_mad = best_common_mad['kc']
ka_mad = best_common_mad['ka']
for base in np.arange(3, 80, 0.5):
    total_err = 0
    sr = {}
    for sn, r in results.items():
        c_th = r['calib_median'] + kc_mad * r['calib_mad']
        a_th = r['calib_median'] - ka_mad * r['calib_mad']
        states = classify_mind_state(r['med_ratio'], c_th, a_th)
        segs = get_calm_segments(states)
        nb = sum(int(s / base) for s in segs)
        err = abs(nb - r['target']['birds'])
        total_err += err
        sr[sn] = nb
    if total_err < best_common_d_mad['total_error']:
        best_common_d_mad = {'total_error': total_err, 'base': base, 'sessions': sr}

print(f"  base={best_common_d_mad['base']:.1f}s/bird, total_err={best_common_d_mad['total_error']}")
for sn, n in best_common_d_mad['sessions'].items():
    print(f"  {sn}: {n} birds (target={results[sn]['target']['birds']})")

# ■ Birds: 仮説Bz (Z-score + 持続時間) - 共通
print("\n■ Birds 仮説Bz (Z-score + 持続時間 共通)")
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

# ■ Recoveries: Z-score共通
print("\n■ Recoveries (Z-score共通パラメータ)")
best_common_rz = {'total_error': float('inf')}
for lb in [3, 5, 7, 10]:
    for lf in [3, 5, 7, 10]:
        for d_z in np.arange(-3.0, 0.5, 0.2):
            for r_z in np.arange(-0.5, 3.0, 0.2):
                if r_z <= d_z:
                    continue
                total_err = 0
                sr = {}
                for sn, r in results.items():
                    recs = []
                    for i in range(lb, len(r['med_zscore']) - lf):
                        pre  = np.mean(r['med_zscore'][i-lb:i])
                        post = np.mean(r['med_zscore'][i:i+lf])
                        if pre <= d_z and post >= r_z:
                            if not recs or (i - recs[-1]) >= 30:
                                recs.append(i)
                    err = abs(len(recs) - r['target']['recoveries'])
                    total_err += err
                    sr[sn] = len(recs)
                if total_err < best_common_rz['total_error']:
                    best_common_rz = {'total_error': total_err, 'lb': lb, 'lf': lf,
                                      'd_z': d_z, 'r_z': r_z, 'sessions': sr}

print(f"  lb={best_common_rz['lb']}s, lf={best_common_rz['lf']}s, "
      f"drop={best_common_rz['d_z']:.1f}, rise={best_common_rz['r_z']:.1f}, "
      f"total_err={best_common_rz['total_error']}")
for sn, n in best_common_rz['sessions'].items():
    print(f"  {sn}: {n} recoveries (target={results[sn]['target']['recoveries']})")

# ============================================================
# 個別最適パラメータ比較
# ============================================================
print(f"\n{'='*70}")
print("  個別最適パラメータ比較")
print(f"{'='*70}")

print("\n■ Mind State Z-score個別最適")
print(f"{'Session':<10} {'calm_z':>8} {'active_z':>10} {'err':>6}")
for sn, r in results.items():
    ms = r['best_ms_z']
    print(f"  {sn:<10} {ms['zc']:>8.2f} {-ms['za']:>10.2f} {ms['error']:>6}s")

print("\n■ Mind State MAD個別最適")
print(f"{'Session':<10} {'kc':>6} {'ka':>6} {'err':>6}")
for sn, r in results.items():
    ms = r['best_ms_mad']
    print(f"  {sn:<10} {ms['kc']:>6.2f} {ms['ka']:>6.2f} {ms['error']:>6}s")

print("\n■ Birds 個別最適 (仮説D base_sec)")
print(f"{'Session':<10} {'base':>6} {'n':>5} {'target':>8} {'err':>6}")
for sn, r in results.items():
    bd = r['best_birds_D']
    print(f"  {sn:<10} {bd['base']:>6.1f} {bd['n']:>5} {r['target']['birds']:>8} {bd['error']:>6}")

print("\n■ Recoveries Z-score個別最適")
print(f"{'Session':<10} {'lb':>4} {'lf':>4} {'drop':>6} {'rise':>6} {'n':>4} {'target':>8} {'err':>6}")
for sn, r in results.items():
    rc = r['best_rec_z']
    print(f"  {sn:<10} {rc['lb']:>4} {rc['lf']:>4} {rc['d_z']:>6.1f} {rc['r_z']:>6.1f} {rc['n']:>4} {r['target']['recoveries']:>8} {rc['error']:>6}")

# ============================================================
# 比較グラフ
# ============================================================
print("\n比較グラフ生成中...")
fig, axes = plt.subplots(4, 3, figsize=(24, 16))

for col, (sn, r) in enumerate(results.items()):
    t = r['target']
    med_time_min = r['med_time'] / 60

    # Z-score時系列
    ax = axes[0, col]
    smoothed = pd.Series(r['med_zscore']).rolling(5, center=True, min_periods=1).mean()
    ax.plot(med_time_min, smoothed, color='blue', linewidth=1)
    ax.axhline(best_common_z['zc'], color='green', ls='--', lw=1.5, label=f'Calm z={best_common_z["zc"]:.2f}')
    ax.axhline(-best_common_z['za'], color='red', ls='--', lw=1.5, label=f'Active z=-{best_common_z["za"]:.2f}')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)

    # 共通パラメータでのRecoveries
    recs_common = []
    lb_c = best_common_rz['lb']; lf_c = best_common_rz['lf']
    for i in range(lb_c, len(r['med_zscore']) - lf_c):
        pre  = np.mean(r['med_zscore'][i-lb_c:i])
        post = np.mean(r['med_zscore'][i:i+lf_c])
        if pre <= best_common_rz['d_z'] and post >= best_common_rz['r_z']:
            if not recs_common or (i - recs_common[-1]) >= 30:
                recs_common.append(i)
    for idx in recs_common:
        ax.axvline(r['med_time'][idx]/60, color='gold', lw=1.5, alpha=0.7)

    ax.set_title(f'{sn}: Z-score (score={t["score"]}, rec={len(recs_common)}/t={t["recoveries"]})')
    ax.set_ylabel('Z-score')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Mind State (共通Zパラメータ)
    ax = axes[1, col]
    states = classify_mind_state(r['med_zscore'], best_common_z['zc'], -best_common_z['za'])
    for state, color in [('Active', 'red'), ('Neutral', 'gray'), ('Calm', 'green')]:
        ax.fill_between(med_time_min, 0, 1, where=(states == state), alpha=0.5, color=color, label=state)
    a_s = np.sum(states == 'Active'); n_s = np.sum(states == 'Neutral'); c_s = np.sum(states == 'Calm')
    ax.set_title(f'{sn}: MindState(A={a_s}/t={t["active_sec"]}, N={n_s}/t={t["neutral_sec"]}, C={c_s}/t={t["calm_sec"]})')
    ax.set_yticks([]); ax.legend(fontsize=7)

    # Birds (仮説D 共通)
    ax = axes[2, col]
    segs_d = get_calm_segments(states)
    nb_d = sum(int(s / best_common_dz['base']) for s in segs_d)
    raw_smoothed = pd.Series(r['med_ratio']).rolling(5, center=True, min_periods=1).mean()
    ax.plot(med_time_min, raw_smoothed, color='purple', linewidth=1)
    ax.axhline(r['best_ms_mad']['calm_th'], color='green', ls='--', alpha=0.6, label='Calm th')
    ax.axhline(r['best_ms_mad']['active_th'], color='red', ls='--', alpha=0.6, label='Active th')
    ax.set_title(f'{sn}: Raw Ratio (Birds D={nb_d}/t={t["birds"]}, base={best_common_dz["base"]:.1f}s)')
    ax.set_ylabel('Alpha/Beta Ratio'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Z-score分布
    ax = axes[3, col]
    ax.hist(r['med_zscore'], bins=80, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(best_common_z['zc'], color='green', lw=2, label=f'Calm z={best_common_z["zc"]:.2f}')
    ax.axvline(-best_common_z['za'], color='red', lw=2, label=f'Active z=-{best_common_z["za"]:.2f}')
    ax.axvline(0, color='gray', ls=':', lw=2)
    ax.set_title(f'{sn}: Z-score Distribution (median={np.median(r["med_zscore"]):.2f})')
    ax.set_xlabel('Z-score'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = OUTPUT_DIR / 'cross_session_v3.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"グラフ保存: {fig_path}")
plt.close()


# ============================================================
# 最終結論
# ============================================================
print(f"\n{'='*70}")
print("  最終アルゴリズム推定 (v3, 3セッション横断)")
print(f"{'='*70}")

print(f"""
=== データ品質 ===
  Session1: 低品質 (35%のみ) - パケットロス大
  Session2: 高品質 (84%)
  Session3: 中品質 (79%)

=== Mind State ===
Z-score共通: calm_z={best_common_z['zc']:.2f}, active_z=-{best_common_z['za']:.2f}, total_err={best_common_z['total_error']}s
MAD共通:     kc={best_common_mad['kc']:.2f}, ka={best_common_mad['ka']:.2f}, total_err={best_common_mad['total_error']}s

個別最適Z-score:
  S1: zc={results['session1']['best_ms_z']['zc']:.2f}, za={results['session1']['best_ms_z']['za']:.2f}
  S2: zc={results['session2']['best_ms_z']['zc']:.2f}, za={results['session2']['best_ms_z']['za']:.2f}
  S3: zc={results['session3']['best_ms_z']['zc']:.2f}, za={results['session3']['best_ms_z']['za']:.2f}

=== Birds ===
仮説D (Calm内N秒/bird):
  Z共通Calm base={best_common_dz['base']:.1f}s/bird, total_err={best_common_dz['total_error']}
  MAD共通Calm base={best_common_d_mad['base']:.1f}s/bird, total_err={best_common_d_mad['total_error']}
  仮説Bz (Z-score+持続): z_th={best_common_bz['z']:.1f}, dur={best_common_bz['dur']}s, total_err={best_common_bz['total_error']}

個別最適仮説D:
  S1: base={results['session1']['best_birds_D']['base']:.1f}s → {results['session1']['best_birds_D']['n']} (target=5)
  S2: base={results['session2']['best_birds_D']['base']:.1f}s → {results['session2']['best_birds_D']['n']} (target=29)
  S3: base={results['session3']['best_birds_D']['base']:.1f}s → {results['session3']['best_birds_D']['n']} (target=87)

=== Recoveries ===
Z-score共通: lb={best_common_rz['lb']}s, lf={best_common_rz['lf']}s, drop={best_common_rz['d_z']:.1f}, rise={best_common_rz['r_z']:.1f}, total_err={best_common_rz['total_error']}

個別最適:
  S1: lb={results['session1']['best_rec_z']['lb']}s, lf={results['session1']['best_rec_z']['lf']}s, drop={results['session1']['best_rec_z']['d_z']:.1f}, rise={results['session1']['best_rec_z']['r_z']:.1f}
  S2: lb={results['session2']['best_rec_z']['lb']}s, lf={results['session2']['best_rec_z']['lf']}s, drop={results['session2']['best_rec_z']['d_z']:.1f}, rise={results['session2']['best_rec_z']['r_z']:.1f}
  S3: lb={results['session3']['best_rec_z']['lb']}s, lf={results['session3']['best_rec_z']['lf']}s, drop={results['session3']['best_rec_z']['d_z']:.1f}, rise={results['session3']['best_rec_z']['r_z']:.1f}
""")
