#!/usr/bin/env python
"""
クロスセッション分析: Session1とSession2を統一アルゴリズムで分析

Session2を~51.5Hzにダウンサンプルし、同一条件で比較。
最終的にMuseアプリのアルゴリズムを推定する。
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

# ============================================================
# セッション定義
# ============================================================
SESSIONS = {
    'session1': {
        'csv': 'session1/muse_app_2026-03-04--08-05-52.csv',
        'target': {
            'birds': 5, 'recoveries': 3,
            'active_sec': 19, 'neutral_sec': 398, 'calm_sec': 369,
            'calm_pct': 46, 'score': 46,
            'duration_min': 15,
        },
    },
    'session2': {
        'csv': 'session2/muse_app_2026-03-04--17-46-58.csv',
        'target': {
            'birds': 29, 'recoveries': 3,
            'active_sec': 5, 'neutral_sec': 139, 'calm_sec': 403,
            'calm_pct': 73, 'score': 73,
            'duration_min': 10,
        },
    },
}

TARGET_SFREQ = 51.5  # 統一サンプリングレート

frontal_channels = ['RAW_AF7', 'RAW_AF8']


def compute_linear_band_power(data_array, sfreq, band, window_sec=2.0, step_sec=1.0):
    """単一チャンネルのリニアバンドパワーを計算"""
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    powers = []
    for start in range(0, len(data_array) - window_samples, step_samples):
        seg = data_array[start:start + window_samples]
        nperseg = min(window_samples, int(sfreq))
        freqs, psd = signal.welch(seg, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        powers.append(np.trapz(psd[mask], freqs[mask]))
    return np.array(powers)


def compute_frontal_band_power(df, sfreq, band, window_sec=2.0, step_sec=1.0):
    """前頭部チャンネル平均のバンドパワー"""
    ch_powers = []
    for ch in frontal_channels:
        data = df[ch].values.astype(float)
        ch_powers.append(compute_linear_band_power(data, sfreq, band, window_sec, step_sec))
    return np.mean(ch_powers, axis=0)


def downsample_df(df, original_sfreq, target_sfreq):
    """DataFrameをダウンサンプル"""
    ratio = int(original_sfreq / target_sfreq)
    if ratio <= 1:
        return df, original_sfreq
    # 単純間引き（anti-aliasing filterは省略 - Welch法で処理）
    df_down = df.iloc[::ratio].reset_index(drop=True)
    actual_sfreq = original_sfreq / ratio
    return df_down, actual_sfreq


def count_sustained_events(values, threshold, time_vec, min_duration_sec):
    """閾値を超える持続的なイベントを検出"""
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
            event_duration = time_vec[i - 1] - time_vec[event_start]
            if event_duration >= min_duration_sec:
                events.append({
                    'start_idx': event_start,
                    'end_idx': i - 1,
                    'start_sec': time_vec[event_start],
                    'end_sec': time_vec[i - 1],
                    'duration': event_duration,
                    'peak_ratio': np.max(values[event_start:i]),
                })
    if in_event:
        event_duration = time_vec[-1] - time_vec[event_start]
        if event_duration >= min_duration_sec:
            events.append({
                'start_idx': event_start,
                'end_idx': len(above) - 1,
                'start_sec': time_vec[event_start],
                'end_sec': time_vec[-1],
                'duration': event_duration,
                'peak_ratio': np.max(values[event_start:]),
            })
    return events


def classify_mind_state(ratio, calm_th, active_th):
    states = np.full(len(ratio), 'Neutral', dtype=object)
    states[ratio >= calm_th] = 'Calm'
    states[ratio <= active_th] = 'Active'
    return states


# ============================================================
# 各セッション分析
# ============================================================
results = {}

for session_name, session_info in SESSIONS.items():
    print(f"\n{'='*70}")
    print(f"  {session_name.upper()} 分析")
    print(f"{'='*70}")

    csv_path = OUTPUT_DIR / session_info['csv']
    target = session_info['target']

    df = pd.read_csv(csv_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

    duration_s = (df['TimeStamp'].iloc[-1] - df['TimeStamp'].iloc[0]).total_seconds()
    original_sfreq = len(df) / duration_s
    print(f"元データ: {len(df)} samples, {duration_s:.1f}s, {original_sfreq:.1f} Hz")

    # ダウンサンプル
    df_ds, actual_sfreq = downsample_df(df, original_sfreq, TARGET_SFREQ)
    print(f"ダウンサンプル後: {len(df_ds)} samples, {actual_sfreq:.1f} Hz")

    # バンドパワー計算
    print("バンドパワー計算中...")
    alpha = compute_frontal_band_power(df_ds, actual_sfreq, (8, 13))
    beta = compute_frontal_band_power(df_ds, actual_sfreq, (13, 25))
    theta = compute_frontal_band_power(df_ds, actual_sfreq, (4, 8))

    # タイムベクトル
    time_sec = np.arange(len(alpha)) * 1.0 + 1.0

    # Alpha/Beta比
    ratio = alpha / (beta + 1e-12)
    ratio_smooth = pd.Series(ratio).rolling(4, center=True, min_periods=1).mean().values

    # キャリブレーション (120秒)
    CALIBRATION_SEC = 120
    calib_mask = time_sec <= CALIBRATION_SEC
    calib_ratio = ratio_smooth[calib_mask]
    calib_median = np.median(calib_ratio)
    calib_mad = np.median(np.abs(calib_ratio - calib_median))

    print(f"\nキャリブレーション:")
    print(f"  median: {calib_median:.3f}")
    print(f"  MAD:    {calib_mad:.3f}")

    # セッション期間
    med_mask = time_sec > CALIBRATION_SEC
    med_time = time_sec[med_mask]
    med_ratio = ratio_smooth[med_mask]
    step = 1.0

    # --------------------------------------------------
    # Mind State: パラメータ最適化
    # --------------------------------------------------
    best_ms = {'error': float('inf')}
    for kc in np.arange(-1.0, 2.0, 0.05):
        for ka in np.arange(0.5, 5.0, 0.1):
            c_th = calib_median + kc * calib_mad
            a_th = calib_median - ka * calib_mad
            if c_th <= a_th:
                continue

            states = classify_mind_state(med_ratio, c_th, a_th)
            a_s = np.sum(states == 'Active') * step
            n_s = np.sum(states == 'Neutral') * step
            c_s = np.sum(states == 'Calm') * step
            err = abs(a_s - target['active_sec']) + abs(n_s - target['neutral_sec']) + abs(c_s - target['calm_sec'])
            if err < best_ms['error']:
                best_ms = {
                    'error': err, 'kc': kc, 'ka': ka,
                    'calm_th': c_th, 'active_th': a_th,
                    'active': a_s, 'neutral': n_s, 'calm': c_s,
                }

    print(f"\nMind State最適パラメータ:")
    print(f"  kc={best_ms['kc']:.2f}, ka={best_ms['ka']:.2f}")
    print(f"  Calm th={best_ms['calm_th']:.3f}, Active th={best_ms['active_th']:.3f}")
    print(f"  Active:  {best_ms['active']:.0f}s (target: {target['active_sec']}s)")
    print(f"  Neutral: {best_ms['neutral']:.0f}s (target: {target['neutral_sec']}s)")
    print(f"  Calm:    {best_ms['calm']:.0f}s (target: {target['calm_sec']}s)")
    print(f"  Error:   {best_ms['error']:.0f}s")

    # Session1のパラメータ (kc=0.05, ka=1.8) での結果
    s1_calm_th = calib_median + 0.05 * calib_mad
    s1_active_th = calib_median - 1.8 * calib_mad
    s1_states = classify_mind_state(med_ratio, s1_calm_th, s1_active_th)
    s1_a = np.sum(s1_states == 'Active') * step
    s1_n = np.sum(s1_states == 'Neutral') * step
    s1_c = np.sum(s1_states == 'Calm') * step
    s1_err = abs(s1_a - target['active_sec']) + abs(s1_n - target['neutral_sec']) + abs(s1_c - target['calm_sec'])
    print(f"\nSession1パラメータ (kc=0.05, ka=1.8) で:")
    print(f"  Active={s1_a:.0f}s, Neutral={s1_n:.0f}s, Calm={s1_c:.0f}s, Error={s1_err:.0f}s")

    # --------------------------------------------------
    # Birds: パラメータ探索
    # --------------------------------------------------
    print(f"\n--- Birds検出 (target={target['birds']}) ---")

    # 仮説A: パーセンタイル + 持続時間
    best_birds_A = {'error': float('inf')}
    for pct in np.arange(30, 95, 1):
        th = np.percentile(med_ratio, pct)
        for min_dur in np.arange(1, 25, 0.5):
            events = count_sustained_events(med_ratio, th, med_time, min_dur)
            err = abs(len(events) - target['birds'])
            if err < best_birds_A['error'] or (
                err == best_birds_A['error'] and min_dur > best_birds_A.get('min_dur', 0)
            ):
                best_birds_A = {
                    'error': err, 'pct': pct, 'threshold': th,
                    'min_dur': min_dur, 'n_events': len(events),
                }

    print(f"\n  仮説A (パーセンタイル+持続時間):")
    print(f"    pct={best_birds_A['pct']}, th={best_birds_A['threshold']:.3f}, "
          f"min_dur={best_birds_A['min_dur']}s → {best_birds_A['n_events']} birds (err={best_birds_A['error']})")

    # 仮説B: キャリブレーション基準 + 持続時間
    best_birds_B = {'error': float('inf')}
    for k in np.arange(-1.0, 3.0, 0.1):
        th = calib_median + k * calib_mad
        for min_dur in np.arange(1, 25, 0.5):
            events = count_sustained_events(med_ratio, th, med_time, min_dur)
            err = abs(len(events) - target['birds'])
            if err < best_birds_B['error'] or (
                err == best_birds_B['error'] and min_dur > best_birds_B.get('min_dur', 0)
            ):
                best_birds_B = {
                    'error': err, 'k': k, 'threshold': th,
                    'min_dur': min_dur, 'n_events': len(events),
                }

    print(f"\n  仮説B (キャリブレーション基準):")
    print(f"    k={best_birds_B['k']:.1f}, th={best_birds_B['threshold']:.3f}, "
          f"min_dur={best_birds_B['min_dur']}s → {best_birds_B['n_events']} birds (err={best_birds_B['error']})")

    # 仮説C: Calm状態中のDeep Calmスパイク(crossing up)
    best_birds_C = {'error': float('inf')}
    final_states = classify_mind_state(med_ratio, best_ms['calm_th'], best_ms['active_th'])
    calm_indices = np.where(final_states == 'Calm')[0]
    if len(calm_indices) > 0:
        calm_ratio = med_ratio[calm_indices]
        for spike_pct in np.arange(40, 95, 1):
            spike_th = np.percentile(calm_ratio, spike_pct)
            above = calm_ratio >= spike_th
            crossings = int(np.sum(np.diff(above.astype(int)) == 1))
            err = abs(crossings - target['birds'])
            if err < best_birds_C['error']:
                best_birds_C = {
                    'error': err, 'pct': spike_pct, 'threshold': spike_th,
                    'n_events': crossings,
                }

    print(f"\n  仮説C (Calm中のDeep Calmスパイク):")
    print(f"    pct={best_birds_C.get('pct', 'N/A')}, th={best_birds_C.get('threshold', 0):.3f} "
          f"→ {best_birds_C.get('n_events', 0)} birds (err={best_birds_C.get('error', 'N/A')})")

    # 仮説D: Calm状態 N秒ごとに1 bird
    best_birds_D = {'error': float('inf')}
    calm_binary = (final_states == 'Calm').astype(int)
    calm_segs = []
    in_seg = False
    seg_st = None
    for i in range(len(calm_binary)):
        if calm_binary[i] and not in_seg:
            in_seg = True
            seg_st = i
        elif not calm_binary[i] and in_seg:
            in_seg = False
            calm_segs.append((i - seg_st) * step)
    if in_seg:
        calm_segs.append((len(calm_binary) - seg_st) * step)

    for base_sec in np.arange(3, 80, 0.5):
        n_birds = sum(int(s / base_sec) for s in calm_segs)
        err = abs(n_birds - target['birds'])
        if err < best_birds_D['error']:
            best_birds_D = {'error': err, 'base_sec': base_sec, 'n_events': n_birds}

    print(f"\n  仮説D (Calmセグメント内 N秒/bird):")
    print(f"    base_sec={best_birds_D['base_sec']:.1f}s → {best_birds_D['n_events']} birds (err={best_birds_D['error']})")

    # --------------------------------------------------
    # Recoveries: パラメータ探索
    # --------------------------------------------------
    print(f"\n--- Recoveries検出 (target={target['recoveries']}) ---")

    best_rec = {'error': float('inf')}
    for lookback in [3, 5, 7, 10]:
        for lookforward in [3, 5, 7, 10]:
            for drop_pct in np.arange(10, 50, 5):
                for rise_pct in np.arange(50, 90, 5):
                    drop_th = np.percentile(med_ratio, drop_pct)
                    rise_th = np.percentile(med_ratio, rise_pct)
                    if rise_th <= drop_th:
                        continue

                    lb_idx = int(lookback / step)
                    lf_idx = int(lookforward / step)
                    recs = []
                    min_gap_idx = int(30 / step)

                    for i in range(lb_idx, len(med_ratio) - lf_idx):
                        pre = np.mean(med_ratio[i - lb_idx:i])
                        post = np.mean(med_ratio[i:i + lf_idx])
                        if pre <= drop_th and post >= rise_th:
                            if not recs or (i - recs[-1]) >= min_gap_idx:
                                recs.append(i)

                    err = abs(len(recs) - target['recoveries'])
                    if err < best_rec['error']:
                        best_rec = {
                            'error': err, 'lookback': lookback, 'lookforward': lookforward,
                            'drop_pct': drop_pct, 'rise_pct': rise_pct,
                            'drop_th': drop_th, 'rise_th': rise_th,
                            'n_recoveries': len(recs), 'indices': recs,
                        }

    print(f"  最適パラメータ: lb={best_rec['lookback']}s, lf={best_rec['lookforward']}s")
    print(f"  Drop: {best_rec['drop_pct']}%ile ({best_rec['drop_th']:.3f})")
    print(f"  Rise: {best_rec['rise_pct']}%ile ({best_rec['rise_th']:.3f})")
    print(f"  検出数: {best_rec['n_recoveries']} (target: {target['recoveries']})")

    for i, idx in enumerate(best_rec['indices']):
        t_min = med_time[idx] / 60
        lb_idx = int(best_rec['lookback'] / step)
        lf_idx = int(best_rec['lookforward'] / step)
        pre = np.mean(med_ratio[max(0, idx - lb_idx):idx])
        post = np.mean(med_ratio[idx:min(len(med_ratio), idx + lf_idx)])
        print(f"    Recovery {i+1}: {t_min:.1f}min (ratio: {pre:.2f} → {post:.2f})")

    # 結果保存
    results[session_name] = {
        'calib_median': calib_median,
        'calib_mad': calib_mad,
        'sfreq': actual_sfreq,
        'best_ms': best_ms,
        's1_ms_error': s1_err,
        'best_birds_A': best_birds_A,
        'best_birds_B': best_birds_B,
        'best_birds_C': best_birds_C,
        'best_birds_D': best_birds_D,
        'best_rec': best_rec,
        'target': target,
        'med_ratio': med_ratio,
        'med_time': med_time,
        'final_states': final_states,
        'ratio_smooth': ratio_smooth,
        'time_sec': time_sec,
        'alpha': alpha,
        'beta': beta,
    }


# ============================================================
# クロスセッション比較
# ============================================================
print(f"\n{'='*70}")
print("  クロスセッション比較")
print(f"{'='*70}")

r1 = results['session1']
r2 = results['session2']

print(f"""
■ キャリブレーション値
  Session1: median={r1['calib_median']:.3f}, MAD={r1['calib_mad']:.3f}
  Session2: median={r2['calib_median']:.3f}, MAD={r2['calib_mad']:.3f}

■ Mind State最適パラメータ
  Session1: kc={r1['best_ms']['kc']:.2f}, ka={r1['best_ms']['ka']:.2f}, err={r1['best_ms']['error']:.0f}s
  Session2: kc={r2['best_ms']['kc']:.2f}, ka={r2['best_ms']['ka']:.2f}, err={r2['best_ms']['error']:.0f}s
  Session1パラメータ(kc=0.05, ka=1.8)をSession2に適用: err={r2['s1_ms_error']:.0f}s

■ Birds最適パラメータ比較
  仮説A (パーセンタイル):
    S1: pct={r1['best_birds_A']['pct']}, dur={r1['best_birds_A']['min_dur']}s → {r1['best_birds_A']['n_events']}  (target={r1['target']['birds']})
    S2: pct={r2['best_birds_A']['pct']}, dur={r2['best_birds_A']['min_dur']}s → {r2['best_birds_A']['n_events']}  (target={r2['target']['birds']})

  仮説B (キャリブレーション基準):
    S1: k={r1['best_birds_B']['k']:.1f}, dur={r1['best_birds_B']['min_dur']}s → {r1['best_birds_B']['n_events']}  (target={r1['target']['birds']})
    S2: k={r2['best_birds_B']['k']:.1f}, dur={r2['best_birds_B']['min_dur']}s → {r2['best_birds_B']['n_events']}  (target={r2['target']['birds']})

  仮説C (Calm中スパイク):
    S1: pct={r1['best_birds_C'].get('pct', 'N/A')} → {r1['best_birds_C'].get('n_events', 0)}  (target={r1['target']['birds']})
    S2: pct={r2['best_birds_C'].get('pct', 'N/A')} → {r2['best_birds_C'].get('n_events', 0)}  (target={r2['target']['birds']})

  仮説D (Calm内 N秒/bird):
    S1: base={r1['best_birds_D']['base_sec']:.1f}s → {r1['best_birds_D']['n_events']}  (target={r1['target']['birds']})
    S2: base={r2['best_birds_D']['base_sec']:.1f}s → {r2['best_birds_D']['n_events']}  (target={r2['target']['birds']})

■ Recoveries最適パラメータ
  Session1: lb={r1['best_rec']['lookback']}s, lf={r1['best_rec']['lookforward']}s, drop={r1['best_rec']['drop_pct']}%ile, rise={r1['best_rec']['rise_pct']}%ile → {r1['best_rec']['n_recoveries']}
  Session2: lb={r2['best_rec']['lookback']}s, lf={r2['best_rec']['lookforward']}s, drop={r2['best_rec']['drop_pct']}%ile, rise={r2['best_rec']['rise_pct']}%ile → {r2['best_rec']['n_recoveries']}
""")

# ============================================================
# 共通パラメータの検証
# ============================================================
print("=" * 70)
print("  共通パラメータでの検証")
print("=" * 70)

# Mind State: 両セッションに共通するkc, ka範囲を探索
print("\n■ Mind State共通パラメータ探索")
best_common_ms = {'total_error': float('inf')}
for kc in np.arange(-1.0, 2.0, 0.05):
    for ka in np.arange(0.5, 5.0, 0.1):
        total_err = 0
        session_results = {}
        for sn, r in results.items():
            c_th = r['calib_median'] + kc * r['calib_mad']
            a_th = r['calib_median'] - ka * r['calib_mad']
            if c_th <= a_th:
                total_err = float('inf')
                break
            states = classify_mind_state(r['med_ratio'], c_th, a_th)
            step = 1.0
            a_s = np.sum(states == 'Active') * step
            n_s = np.sum(states == 'Neutral') * step
            c_s = np.sum(states == 'Calm') * step
            t = r['target']
            err = abs(a_s - t['active_sec']) + abs(n_s - t['neutral_sec']) + abs(c_s - t['calm_sec'])
            total_err += err
            session_results[sn] = {'active': a_s, 'neutral': n_s, 'calm': c_s, 'error': err}

        if total_err < best_common_ms['total_error']:
            best_common_ms = {
                'total_error': total_err, 'kc': kc, 'ka': ka,
                'sessions': session_results,
            }

print(f"  最適共通パラメータ: kc={best_common_ms['kc']:.2f}, ka={best_common_ms['ka']:.2f}")
print(f"  Total error: {best_common_ms['total_error']:.0f}s")
for sn, sr in best_common_ms['sessions'].items():
    t = results[sn]['target']
    print(f"  {sn}: Active={sr['active']:.0f}s(t={t['active_sec']}), "
          f"Neutral={sr['neutral']:.0f}s(t={t['neutral_sec']}), "
          f"Calm={sr['calm']:.0f}s(t={t['calm_sec']}), err={sr['error']:.0f}s")

# Birds: 共通パラメータ探索 (仮説B: キャリブレーション基準)
print("\n■ Birds共通パラメータ探索 (仮説B: k * MAD + median)")
best_common_birds = {'total_error': float('inf')}
for k in np.arange(-2.0, 4.0, 0.1):
    for min_dur in np.arange(1, 25, 0.5):
        total_err = 0
        session_bird_counts = {}
        for sn, r in results.items():
            th = r['calib_median'] + k * r['calib_mad']
            events = count_sustained_events(r['med_ratio'], th, r['med_time'], min_dur)
            err = abs(len(events) - r['target']['birds'])
            total_err += err
            session_bird_counts[sn] = len(events)

        if total_err < best_common_birds['total_error'] or (
            total_err == best_common_birds['total_error'] and
            min_dur > best_common_birds.get('min_dur', 0)
        ):
            best_common_birds = {
                'total_error': total_err, 'k': k, 'min_dur': min_dur,
                'sessions': session_bird_counts,
            }

print(f"  最適共通パラメータ: k={best_common_birds['k']:.1f}, min_dur={best_common_birds['min_dur']}s")
print(f"  Total error: {best_common_birds['total_error']}")
for sn, count in best_common_birds['sessions'].items():
    print(f"  {sn}: {count} birds (target={results[sn]['target']['birds']})")

# Birds: 共通パラメータ探索 (仮説D: Calm内 N秒/bird)
print("\n■ Birds共通パラメータ探索 (仮説D: Calm N秒ごとに1 bird)")
# 各セッションでCalm stateを再計算 (共通MS使用)
common_kc = best_common_ms['kc']
common_ka = best_common_ms['ka']

best_common_birds_D = {'total_error': float('inf')}
for base_sec in np.arange(3, 80, 0.5):
    total_err = 0
    session_bird_counts = {}
    for sn, r in results.items():
        c_th = r['calib_median'] + common_kc * r['calib_mad']
        a_th = r['calib_median'] - common_ka * r['calib_mad']
        states = classify_mind_state(r['med_ratio'], c_th, a_th)
        calm_binary = (states == 'Calm').astype(int)

        # Calmセグメント
        segs = []
        in_seg = False
        seg_st = None
        for i in range(len(calm_binary)):
            if calm_binary[i] and not in_seg:
                in_seg = True
                seg_st = i
            elif not calm_binary[i] and in_seg:
                in_seg = False
                segs.append((i - seg_st) * 1.0)
        if in_seg:
            segs.append((len(calm_binary) - seg_st) * 1.0)

        n_birds = sum(int(s / base_sec) for s in segs)
        err = abs(n_birds - r['target']['birds'])
        total_err += err
        session_bird_counts[sn] = n_birds

    if total_err < best_common_birds_D['total_error']:
        best_common_birds_D = {
            'total_error': total_err, 'base_sec': base_sec,
            'sessions': session_bird_counts,
        }

print(f"  最適共通パラメータ: base_sec={best_common_birds_D['base_sec']:.1f}s")
print(f"  Total error: {best_common_birds_D['total_error']}")
for sn, count in best_common_birds_D['sessions'].items():
    print(f"  {sn}: {count} birds (target={results[sn]['target']['birds']})")

# Recoveries: 共通パラメータ探索
print("\n■ Recoveries共通パラメータ探索")
best_common_rec = {'total_error': float('inf')}
for lb in [3, 5, 7, 10]:
    for lf in [3, 5, 7, 10]:
        for drop_pct in np.arange(10, 50, 5):
            for rise_pct in np.arange(50, 90, 5):
                total_err = 0
                session_rec_counts = {}
                valid = True
                for sn, r in results.items():
                    drop_th = np.percentile(r['med_ratio'], drop_pct)
                    rise_th = np.percentile(r['med_ratio'], rise_pct)
                    if rise_th <= drop_th:
                        valid = False
                        break

                    lb_idx = int(lb / 1.0)
                    lf_idx = int(lf / 1.0)
                    recs = []
                    min_gap_idx = int(30 / 1.0)
                    for i in range(lb_idx, len(r['med_ratio']) - lf_idx):
                        pre = np.mean(r['med_ratio'][i - lb_idx:i])
                        post = np.mean(r['med_ratio'][i:i + lf_idx])
                        if pre <= drop_th and post >= rise_th:
                            if not recs or (i - recs[-1]) >= min_gap_idx:
                                recs.append(i)
                    err = abs(len(recs) - r['target']['recoveries'])
                    total_err += err
                    session_rec_counts[sn] = len(recs)

                if not valid:
                    continue

                if total_err < best_common_rec['total_error']:
                    best_common_rec = {
                        'total_error': total_err, 'lb': lb, 'lf': lf,
                        'drop_pct': drop_pct, 'rise_pct': rise_pct,
                        'sessions': session_rec_counts,
                    }

print(f"  最適共通パラメータ: lb={best_common_rec['lb']}s, lf={best_common_rec['lf']}s, "
      f"drop={best_common_rec['drop_pct']}%ile, rise={best_common_rec['rise_pct']}%ile")
print(f"  Total error: {best_common_rec['total_error']}")
for sn, count in best_common_rec['sessions'].items():
    print(f"  {sn}: {count} recoveries (target={results[sn]['target']['recoveries']})")


# ============================================================
# 可視化: 2セッション比較
# ============================================================
print("\n比較グラフ生成中...")

fig, axes = plt.subplots(4, 2, figsize=(20, 16))

for col, (sn, r) in enumerate(results.items()):
    t = r['target']
    med_time_min = r['med_time'] / 60

    # Alpha/Beta比
    ax = axes[0, col]
    c_th = r['calib_median'] + best_common_ms['kc'] * r['calib_mad']
    a_th = r['calib_median'] - best_common_ms['ka'] * r['calib_mad']
    smoothed = pd.Series(r['med_ratio']).rolling(5, center=True, min_periods=1).mean()
    ax.plot(med_time_min, smoothed, color='blue', linewidth=1)
    ax.axhline(c_th, color='green', linestyle='--', alpha=0.7, label=f'Calm th')
    ax.axhline(a_th, color='red', linestyle='--', alpha=0.7, label=f'Active th')
    ax.set_title(f'{sn}: Alpha/Beta Ratio (score={t["score"]})')
    ax.set_ylabel('Ratio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mind State
    ax = axes[1, col]
    states = classify_mind_state(r['med_ratio'], c_th, a_th)
    colors_map = {'Active': 'red', 'Neutral': 'gray', 'Calm': 'green'}
    for state in ['Active', 'Neutral', 'Calm']:
        mask = states == state
        ax.fill_between(med_time_min, 0, 1, where=mask,
                        alpha=0.5, color=colors_map[state], label=state)
    a_s = np.sum(states == 'Active')
    n_s = np.sum(states == 'Neutral')
    c_s = np.sum(states == 'Calm')
    ax.set_title(f'{sn}: Mind State (A={a_s}s/t={t["active_sec"]}, '
                 f'N={n_s}s/t={t["neutral_sec"]}, C={c_s}s/t={t["calm_sec"]})')
    ax.set_yticks([])
    ax.legend(fontsize=8)

    # バンドパワー
    ax = axes[2, col]
    alpha_db = 10 * np.log10(r['alpha'] + 1e-12)
    beta_db = 10 * np.log10(r['beta'] + 1e-12)
    time_min = r['time_sec'] / 60
    ax.plot(time_min, pd.Series(alpha_db).rolling(10, center=True, min_periods=1).mean(),
            color='green', label='Alpha')
    ax.plot(time_min, pd.Series(beta_db).rolling(10, center=True, min_periods=1).mean(),
            color='orange', label='Beta')
    ax.axvline(120/60, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'{sn}: Band Power')
    ax.set_ylabel('dB')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 比率の分布
    ax = axes[3, col]
    ax.hist(r['med_ratio'], bins=80, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(c_th, color='green', linewidth=2, label=f'Calm th')
    ax.axvline(a_th, color='red', linewidth=2, label=f'Active th')
    ax.axvline(r['calib_median'], color='blue', linestyle=':', linewidth=2, label='Calib median')
    ax.set_title(f'{sn}: Ratio Distribution')
    ax.set_xlabel('Alpha/Beta Ratio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = OUTPUT_DIR / 'cross_session_comparison.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"グラフ保存: {fig_path}")
plt.close()


# ============================================================
# 最終結論
# ============================================================
print(f"\n{'='*70}")
print("  最終アルゴリズム推定")
print(f"{'='*70}")

print(f"""
■ 共通Mind Stateパラメータ: kc={best_common_ms['kc']:.2f}, ka={best_common_ms['ka']:.2f}
  - Calm閾値  = calib_median + {best_common_ms['kc']:.2f} × calib_MAD
  - Active閾値 = calib_median − {best_common_ms['ka']:.2f} × calib_MAD

■ 共通Birdsパラメータ (仮説B): k={best_common_birds['k']:.1f}, min_dur={best_common_birds['min_dur']}s
  - Deep Calm閾値 = calib_median + {best_common_birds['k']:.1f} × calib_MAD
  - 最小持続時間 = {best_common_birds['min_dur']}s

■ 共通Birdsパラメータ (仮説D): base_sec={best_common_birds_D['base_sec']:.1f}s
  - Calmが{best_common_birds_D['base_sec']:.1f}秒続くごとに1 bird

■ 共通Recoveriesパラメータ:
  - lookback={best_common_rec['lb']}s, lookforward={best_common_rec['lf']}s
  - drop < {best_common_rec['drop_pct']}%ile, rise > {best_common_rec['rise_pct']}%ile
  - 最小間隔 30秒
""")
