#!/usr/bin/env python
"""
Session3 分析スクリプト

Session1/2で推定したアルゴリズムをSession3に適用し検証する。

アプリ表示値 (Session3):
- Score: 76
- Muse Points: 2931
- Recoveries: 8
- Birds: 87
- Calm: 76%, Active: 5s, Neutral: 4m21s(261s), Calm: 14m50s(890s)
- Duration: 20 mins
- Date: 2026-03-06 08:05
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
CSV_PATH = Path(__file__).parent / 'session3' / 'muse_app_2026-03-06--08-05-43.csv'

TARGET = {
    'birds': 87,
    'recoveries': 8,
    'active_sec': 5,
    'neutral_sec': 261,
    'calm_sec': 890,
    'calm_pct': 76,
    'score': 76,
    'duration_min': 20,
}

CALIBRATION_SEC = 60
TARGET_SFREQ = 51.5
frontal_channels = ['RAW_AF7', 'RAW_AF8']

# ============================================================
# 1. データ読み込み
# ============================================================
print("=" * 60)
print("SESSION3 分析")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

duration_s = (df['TimeStamp'].iloc[-1] - df['TimeStamp'].iloc[0]).total_seconds()
orig_sfreq = len(df) / duration_s
print(f"データ: {len(df)} samples, {duration_s:.1f}s ({duration_s/60:.1f}min), {orig_sfreq:.1f} Hz")

# データ品質確認
seconds_with_data = df['TimeStamp'].apply(lambda x: x.floor('s')).nunique()
total_seconds = int(duration_s)
print(f"データ品質: {seconds_with_data}/{total_seconds}秒にデータあり ({100*seconds_with_data/total_seconds:.0f}%)")

# ============================================================
# 2. アンチエイリアスフィルタ付きダウンサンプル (Session2と同じ手法)
# ============================================================
def downsample_with_aa(data, original_sfreq, target_sfreq):
    ratio = int(round(original_sfreq / target_sfreq))
    if ratio <= 1:
        return data, original_sfreq
    nyq = original_sfreq / 2
    cutoff = (target_sfreq / 2) * 0.9
    sos = signal.butter(8, cutoff / nyq, btype='low', output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    downsampled = filtered[::ratio]
    return downsampled, original_sfreq / ratio

ch_data = {}
actual_sfreq = None
for ch in frontal_channels:
    raw = df[ch].values.astype(float)
    ds, sf = downsample_with_aa(raw, orig_sfreq, TARGET_SFREQ)
    ch_data[ch] = ds
    actual_sfreq = sf
print(f"ダウンサンプル後: {len(ch_data[frontal_channels[0]])} samples, {actual_sfreq:.1f} Hz")

# ============================================================
# 3. バンドパワー計算
# ============================================================
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

print("\nバンドパワー計算中...")
ch_alpha = [compute_band_power(ch_data[ch], actual_sfreq, (8, 13)) for ch in frontal_channels]
ch_beta  = [compute_band_power(ch_data[ch], actual_sfreq, (13, 25)) for ch in frontal_channels]

min_len = min(len(ch_alpha[0]), len(ch_alpha[1]), len(ch_beta[0]), len(ch_beta[1]))
alpha = np.mean([a[:min_len] for a in ch_alpha], axis=0)
beta  = np.mean([b[:min_len] for b in ch_beta],  axis=0)
time_sec = np.arange(min_len) * 1.0 + 1.0
print(f"バンドパワーポイント数: {min_len}, 時間範囲: {time_sec[0]:.1f}s - {time_sec[-1]:.1f}s")

# Alpha/Beta比 + スムージング
ratio = alpha / (beta + 1e-12)
ratio_smooth = pd.Series(ratio).rolling(4, center=True, min_periods=1).mean().values

# ============================================================
# 4. キャリブレーション
# ============================================================
calib_mask = time_sec <= CALIBRATION_SEC
calib_ratio = ratio_smooth[calib_mask]
calib_median = np.median(calib_ratio)
calib_mad    = np.median(np.abs(calib_ratio - calib_median))
calib_mean   = np.mean(calib_ratio)
calib_std    = np.std(calib_ratio)

print(f"\nキャリブレーション (0-{CALIBRATION_SEC}s):")
print(f"  median={calib_median:.3f}, MAD={calib_mad:.3f}")
print(f"  mean={calib_mean:.3f},   std={calib_std:.3f}")

# Z-score正規化
ratio_zscore = (ratio_smooth - calib_mean) / (calib_std + 1e-12)

# セッション期間
med_mask   = time_sec > CALIBRATION_SEC
med_time   = time_sec[med_mask]
med_ratio  = ratio_smooth[med_mask]
med_zscore = ratio_zscore[med_mask]
step = 1.0

print(f"\n瞑想期間: {med_time[0]:.0f}s - {med_time[-1]:.0f}s")
print(f"  ratio  : median={np.median(med_ratio):.3f}, mean={np.mean(med_ratio):.3f}, std={np.std(med_ratio):.3f}")
print(f"  zscore : median={np.median(med_zscore):.3f}, mean={np.mean(med_zscore):.3f}, std={np.std(med_zscore):.3f}")

# ============================================================
# 5. Mind State 分類
# ============================================================
def classify_mind_state(ratio_arr, calm_th, active_th):
    states = np.full(len(ratio_arr), 'Neutral', dtype=object)
    states[ratio_arr >= calm_th] = 'Calm'
    states[ratio_arr <= active_th] = 'Active'
    return states

print("\n" + "=" * 60)
print("Mind State 分析")
print("=" * 60)

# S1/S2共通パラメータ (Z-score: calm_z=0.05, active_z=-1.60 from cross_session_v2)
# Cross-session v2で得られた共通パラメータを適用
COMMON_ZC = 0.05    # S1/S2 共通値 (実際の値はanalyze_cross_session_v2.pyの出力を参照)
COMMON_ZA = 1.60

states_common_z = classify_mind_state(med_zscore, COMMON_ZC, -COMMON_ZA)
a_cz = np.sum(states_common_z == 'Active')
n_cz = np.sum(states_common_z == 'Neutral')
c_cz = np.sum(states_common_z == 'Calm')
print(f"\nS1/S2共通Zパラメータ (calm_z={COMMON_ZC}, active_z=-{COMMON_ZA}) の適用結果:")
print(f"  Active:  {a_cz}s (target: {TARGET['active_sec']}s, diff: {a_cz-TARGET['active_sec']:+d}s)")
print(f"  Neutral: {n_cz}s (target: {TARGET['neutral_sec']}s, diff: {n_cz-TARGET['neutral_sec']:+d}s)")
print(f"  Calm:    {c_cz}s (target: {TARGET['calm_sec']}s, diff: {c_cz-TARGET['calm_sec']:+d}s)")

# 個別最適化 (MADベース)
print("\n--- Mind State MADベース最適化 ---")
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
        err = abs(a_s - TARGET['active_sec']) + abs(n_s - TARGET['neutral_sec']) + abs(c_s - TARGET['calm_sec'])
        if err < best_ms_mad['error']:
            best_ms_mad = {'error': err, 'kc': kc, 'ka': ka, 'calm_th': c_th, 'active_th': a_th,
                           'active': a_s, 'neutral': n_s, 'calm': c_s}

print(f"  最適: kc={best_ms_mad['kc']:.2f}, ka={best_ms_mad['ka']:.2f}, err={best_ms_mad['error']}s")
print(f"  A={best_ms_mad['active']}s(t={TARGET['active_sec']}), "
      f"N={best_ms_mad['neutral']}s(t={TARGET['neutral_sec']}), "
      f"C={best_ms_mad['calm']}s(t={TARGET['calm_sec']})")

# 個別最適化 (Z-scoreベース)
print("\n--- Mind State Z-scoreベース最適化 ---")
best_ms_z = {'error': float('inf')}
for zc in np.arange(-3.0, 3.0, 0.05):
    for za in np.arange(0.1, 5.0, 0.1):
        states = classify_mind_state(med_zscore, zc, -za)
        a_s = np.sum(states == 'Active')
        n_s = np.sum(states == 'Neutral')
        c_s = np.sum(states == 'Calm')
        err = abs(a_s - TARGET['active_sec']) + abs(n_s - TARGET['neutral_sec']) + abs(c_s - TARGET['calm_sec'])
        if err < best_ms_z['error']:
            best_ms_z = {'error': err, 'zc': zc, 'za': za, 'active': a_s, 'neutral': n_s, 'calm': c_s}

print(f"  最適: calm_z={best_ms_z['zc']:.2f}, active_z=-{best_ms_z['za']:.2f}, err={best_ms_z['error']}s")
print(f"  A={best_ms_z['active']}s(t={TARGET['active_sec']}), "
      f"N={best_ms_z['neutral']}s(t={TARGET['neutral_sec']}), "
      f"C={best_ms_z['calm']}s(t={TARGET['calm_sec']})")

# ============================================================
# 6. Birds 検出
# ============================================================
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
                events.append({'start_sec': time_vec[event_start], 'end_sec': time_vec[i-1], 'duration': dur})
    if in_event:
        dur = time_vec[-1] - time_vec[event_start]
        if dur >= min_duration_sec:
            events.append({'start_sec': time_vec[event_start], 'end_sec': time_vec[-1], 'duration': dur})
    return events

print("\n" + "=" * 60)
print(f"Birds 分析 (target={TARGET['birds']})")
print("=" * 60)

# 仮説A: パーセンタイル + 持続時間
best_A = {'error': float('inf')}
for pct in np.arange(20, 95, 1):
    th = np.percentile(med_ratio, pct)
    for dur in np.arange(1, 30, 0.5):
        evts = count_sustained_events(med_ratio, th, med_time, dur)
        err = abs(len(evts) - TARGET['birds'])
        if err < best_A['error'] or (err == best_A['error'] and dur > best_A.get('dur', 0)):
            best_A = {'error': err, 'pct': pct, 'th': th, 'dur': dur, 'n': len(evts)}
print(f"  仮説A (pct+持続): pct={best_A['pct']}, dur={best_A['dur']}s → {best_A['n']} (err={best_A['error']})")

# 仮説B: キャリブレーション基準 + 持続時間
best_B = {'error': float('inf')}
for k in np.arange(-3.0, 5.0, 0.1):
    th = calib_median + k * calib_mad
    for dur in np.arange(1, 30, 0.5):
        evts = count_sustained_events(med_ratio, th, med_time, dur)
        err = abs(len(evts) - TARGET['birds'])
        if err < best_B['error'] or (err == best_B['error'] and dur > best_B.get('dur', 0)):
            best_B = {'error': err, 'k': k, 'th': th, 'dur': dur, 'n': len(evts)}
print(f"  仮説B (calib基準+持続): k={best_B['k']:.1f}, dur={best_B['dur']}s → {best_B['n']} (err={best_B['error']})")

# 仮説Bz: Z-score基準 + 持続時間
best_Bz = {'error': float('inf')}
for z_th in np.arange(-3.0, 5.0, 0.1):
    for dur in np.arange(1, 30, 0.5):
        evts = count_sustained_events(med_zscore, z_th, med_time, dur)
        err = abs(len(evts) - TARGET['birds'])
        if err < best_Bz['error'] or (err == best_Bz['error'] and dur > best_Bz.get('dur', 0)):
            best_Bz = {'error': err, 'z': z_th, 'dur': dur, 'n': len(evts)}
print(f"  仮説Bz (Z-score+持続): z={best_Bz['z']:.1f}, dur={best_Bz['dur']}s → {best_Bz['n']} (err={best_Bz['error']})")

# 仮説D: Calm内 N秒/bird
states_for_d = classify_mind_state(med_ratio, best_ms_mad['calm_th'], best_ms_mad['active_th'])
calm_binary = (states_for_d == 'Calm').astype(int)
segs = []
in_s = False; st = None
for i in range(len(calm_binary)):
    if calm_binary[i] and not in_s: in_s = True; st = i
    elif not calm_binary[i] and in_s:
        in_s = False; segs.append((i - st) * step)
if in_s: segs.append((len(calm_binary) - st) * step)

best_D = {'error': float('inf')}
for base in np.arange(3, 80, 0.5):
    nb = sum(int(s / base) for s in segs)
    err = abs(nb - TARGET['birds'])
    if err < best_D['error']:
        best_D = {'error': err, 'base': base, 'n': nb}
print(f"  仮説D (Calm内N秒/bird): base={best_D['base']:.1f}s → {best_D['n']} (err={best_D['error']})")

# ============================================================
# 7. Recoveries 検出
# ============================================================
print("\n" + "=" * 60)
print(f"Recoveries 分析 (target={TARGET['recoveries']})")
print("=" * 60)

# S1/S2共通パラメータ: lb=3s, lf=7s, drop_z=-1.6, rise_z=0.2
COMMON_LB = 3
COMMON_LF = 7
COMMON_DROP_Z = -1.6
COMMON_RISE_Z = 0.2
MIN_GAP = 30

recs_common = []
for i in range(COMMON_LB, len(med_zscore) - COMMON_LF):
    pre  = np.mean(med_zscore[i-COMMON_LB:i])
    post = np.mean(med_zscore[i:i+COMMON_LF])
    if pre <= COMMON_DROP_Z and post >= COMMON_RISE_Z:
        if not recs_common or (i - recs_common[-1]) >= MIN_GAP:
            recs_common.append(i)

print(f"\nS1/S2共通Zパラメータ (lb={COMMON_LB}s, lf={COMMON_LF}s, drop={COMMON_DROP_Z}, rise={COMMON_RISE_Z}) 適用:")
print(f"  検出数: {len(recs_common)} (target: {TARGET['recoveries']})")
for i, idx in enumerate(recs_common):
    t_min = med_time[idx] / 60
    pre_v  = np.mean(med_zscore[max(0,idx-COMMON_LB):idx])
    post_v = np.mean(med_zscore[idx:min(len(med_zscore),idx+COMMON_LF)])
    print(f"  Recovery {i+1}: {t_min:.1f}min (z: {pre_v:.2f} → {post_v:.2f})")

# 個別最適化 (Z-score)
print("\n--- Recoveries Z-score個別最適化 ---")
best_rec_z = {'error': float('inf')}
for lb in [3, 5, 7, 10]:
    for lf in [3, 5, 7, 10]:
        for d_z in np.arange(-3.0, 0.0, 0.2):
            for r_z in np.arange(0.0, 3.0, 0.2):
                recs = []
                for i in range(lb, len(med_zscore) - lf):
                    pre = np.mean(med_zscore[i-lb:i])
                    post = np.mean(med_zscore[i:i+lf])
                    if pre <= d_z and post >= r_z:
                        if not recs or (i - recs[-1]) >= MIN_GAP:
                            recs.append(i)
                err = abs(len(recs) - TARGET['recoveries'])
                if err < best_rec_z['error']:
                    best_rec_z = {'error': err, 'lb': lb, 'lf': lf, 'd_z': d_z, 'r_z': r_z,
                                  'n': len(recs), 'indices': recs}

print(f"  最適: lb={best_rec_z['lb']}s, lf={best_rec_z['lf']}s, "
      f"drop={best_rec_z['d_z']:.1f}, rise={best_rec_z['r_z']:.1f}, err={best_rec_z['error']}")
print(f"  検出数: {best_rec_z['n']} (target: {TARGET['recoveries']})")
for i, idx in enumerate(best_rec_z['indices']):
    t_min = med_time[idx] / 60
    print(f"  Recovery {i+1}: {t_min:.1f}min")

# ============================================================
# 8. 可視化
# ============================================================
print("\nグラフ生成中...")

fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
med_time_min = med_time / 60

# (a) Z-score時系列
ax = axes[0]
smoothed_z = pd.Series(med_zscore).rolling(5, center=True, min_periods=1).mean()
ax.plot(med_time_min, smoothed_z, color='blue', linewidth=1, alpha=0.8, label='Z-score')
ax.axhline(COMMON_ZC, color='green', ls='--', label=f'Calm z={COMMON_ZC}')
ax.axhline(-COMMON_ZA, color='red', ls='--', label=f'Active z=-{COMMON_ZA}')
ax.axhline(best_ms_z['zc'], color='lime', ls=':', label=f'Opt Calm z={best_ms_z["zc"]:.2f}')
ax.axhline(-best_ms_z['za'], color='salmon', ls=':', label=f'Opt Active z=-{best_ms_z["za"]:.2f}')
ax.axhline(0, color='gray', ls=':', alpha=0.5)
for idx in recs_common:
    ax.axvline(med_time[idx]/60, color='gold', lw=2, alpha=0.7)
ax.set_ylabel('Z-score')
ax.set_title(f'Session3: Z-score Alpha/Beta (Score={TARGET["score"]}, S1/S2共通パラメータ適用)')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# (b) Mind State比較
ax = axes[1]
# 共通パラメータ
for state, color in [('Active', 'red'), ('Neutral', 'gray'), ('Calm', 'green')]:
    ax.fill_between(med_time_min, 0, 0.5, where=(states_common_z == state),
                    alpha=0.5, color=color, label=f'{state} (common)')
# 個別最適
states_opt = classify_mind_state(med_zscore, best_ms_z['zc'], -best_ms_z['za'])
for state, color in [('Active', 'darkred'), ('Neutral', 'darkgray'), ('Calm', 'darkgreen')]:
    ax.fill_between(med_time_min, 0.5, 1.0, where=(states_opt == state),
                    alpha=0.5, color=color, label=f'{state} (opt)')
ax.set_title(f'Mind State: 上=共通パラメータ(A={a_cz}/t={TARGET["active_sec"]}, N={n_cz}/t={TARGET["neutral_sec"]}, C={c_cz}/t={TARGET["calm_sec"]}), '
             f'下=個別最適(err={best_ms_z["error"]}s)')
ax.set_yticks([0.25, 0.75])
ax.set_yticklabels(['共通', '個別最適'])
ax.legend(fontsize=7, loc='upper right', ncol=3)

# (c) Raw Alpha/Beta比
ax = axes[2]
smoothed_r = pd.Series(med_ratio).rolling(5, center=True, min_periods=1).mean()
ax.plot(med_time_min, smoothed_r, color='purple', linewidth=1, alpha=0.8)
ax.axhline(best_ms_mad['calm_th'],   color='green', ls='--', label=f'Calm th (MAD k={best_ms_mad["kc"]:.2f})')
ax.axhline(best_ms_mad['active_th'], color='red',   ls='--', label=f'Active th (MAD k={best_ms_mad["ka"]:.2f})')
ax.axhline(calib_median, color='blue', ls=':', label=f'calib median={calib_median:.2f}')
ax.set_ylabel('Alpha/Beta Ratio')
ax.set_title(f'Session3: Alpha/Beta Ratio (calib median={calib_median:.2f}, MAD={calib_mad:.2f})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (d) Z-score分布
ax = axes[3]
ax.hist(med_zscore, bins=100, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(COMMON_ZC, color='green', lw=2, label=f'S1/S2共通 Calm z={COMMON_ZC}')
ax.axvline(-COMMON_ZA, color='red', lw=2, label=f'S1/S2共通 Active z=-{COMMON_ZA}')
ax.axvline(best_ms_z['zc'], color='lime', lw=2, ls='--', label=f'個別最適 Calm z={best_ms_z["zc"]:.2f}')
ax.axvline(-best_ms_z['za'], color='salmon', lw=2, ls='--', label=f'個別最適 Active z=-{best_ms_z["za"]:.2f}')
ax.axvline(0, color='gray', ls=':', lw=2)
ax.set_xlabel('Z-score')
ax.set_ylabel('Count')
ax.set_title('Session3: Z-score Distribution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax.set_xlabel('Time (min)')
plt.tight_layout()
fig_path = OUTPUT_DIR / 'session3_analysis.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"グラフ保存: {fig_path}")
plt.close()

# ============================================================
# 9. サマリー
# ============================================================
print("\n" + "=" * 60)
print("Session3 サマリー")
print("=" * 60)
print(f"""
アプリ値:
  Score={TARGET['score']}, Birds={TARGET['birds']}, Recoveries={TARGET['recoveries']}
  Active={TARGET['active_sec']}s, Neutral={TARGET['neutral_sec']}s, Calm={TARGET['calm_sec']}s

キャリブレーション:
  median={calib_median:.3f}, MAD={calib_mad:.3f}
  mean={calib_mean:.3f}, std={calib_std:.3f}

Mind State推定精度:
  S1/S2共通Zパラメータ: A={a_cz}s/t={TARGET['active_sec']}, N={n_cz}s/t={TARGET['neutral_sec']}, C={c_cz}s/t={TARGET['calm_sec']}, err={abs(a_cz-TARGET['active_sec'])+abs(n_cz-TARGET['neutral_sec'])+abs(c_cz-TARGET['calm_sec'])}s
  MAD個別最適: kc={best_ms_mad['kc']:.2f}, ka={best_ms_mad['ka']:.2f}, err={best_ms_mad['error']}s
  Z個別最適:  zc={best_ms_z['zc']:.2f}, za=-{best_ms_z['za']:.2f}, err={best_ms_z['error']}s

Birds推定精度:
  仮説A: pct={best_A['pct']}, dur={best_A['dur']}s → {best_A['n']} (err={best_A['error']})
  仮説B: k={best_B['k']:.1f}, dur={best_B['dur']}s → {best_B['n']} (err={best_B['error']})
  仮説Bz: z={best_Bz['z']:.1f}, dur={best_Bz['dur']}s → {best_Bz['n']} (err={best_Bz['error']})
  仮説D: base={best_D['base']:.1f}s → {best_D['n']} (err={best_D['error']})

Recoveries推定精度:
  S1/S2共通Zパラメータ: {len(recs_common)} (target: {TARGET['recoveries']}, err={abs(len(recs_common)-TARGET['recoveries'])})
  Z個別最適: lb={best_rec_z['lb']}s, lf={best_rec_z['lf']}s, drop={best_rec_z['d_z']:.1f}, rise={best_rec_z['r_z']:.1f}, err={best_rec_z['error']}
""")

# 結果をpickle保存 (クロスセッション分析用)
import json
session3_results = {
    'calib_median': float(calib_median),
    'calib_mad': float(calib_mad),
    'calib_mean': float(calib_mean),
    'calib_std': float(calib_std),
    'best_ms_mad': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                    for k, v in best_ms_mad.items() if k != 'indices'},
    'best_ms_z': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                  for k, v in best_ms_z.items() if k != 'indices'},
    'best_birds_A': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                     for k, v in best_A.items()},
    'best_birds_B': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                     for k, v in best_B.items()},
    'best_birds_Bz': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                      for k, v in best_Bz.items()},
    'best_birds_D': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                     for k, v in best_D.items()},
    'recoveries_common_n': len(recs_common),
    'best_rec_z': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                   for k, v in best_rec_z.items() if k != 'indices'},
    'target': TARGET,
}
with open(OUTPUT_DIR / 'session3_results.json', 'w') as f:
    json.dump(session3_results, f, indent=2)
print(f"結果保存: {OUTPUT_DIR / 'session3_results.json'}")
