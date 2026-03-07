#!/usr/bin/env python
"""
Session2 分析スクリプト

Session1で推定したアルゴリズムをSession2に適用し、
クロスバリデーションを行い、アルゴリズムを精緻化する。

アプリ表示値 (Session2):
- Score: 73
- Muse Points: 1348
- Recoveries: 3
- Birds: 29
- Calm: 73%, Active: 5s, Neutral: 2m19s(139s), Calm: 6m43s(403s)
- Duration: 10 mins
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
# 1. データ読み込み
# ============================================================
CSV_PATH = Path(__file__).parent / 'session2' / 'muse_app_2026-03-04--17-46-58.csv'

df = pd.read_csv(CSV_PATH)
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

duration_s = (df['TimeStamp'].iloc[-1] - df['TimeStamp'].iloc[0]).total_seconds()
effective_sfreq = len(df) / duration_s
print(f"Session2 データ: {len(df)} samples, {duration_s:.1f}s ({duration_s/60:.1f}min), effective {effective_sfreq:.1f} Hz")

# ターゲット値
TARGET = {
    'birds': 29,
    'recoveries': 3,
    'active_sec': 5,
    'neutral_sec': 139,
    'calm_sec': 403,
    'calm_pct': 73,
    'score': 73,
}
SESSION_DURATION_MIN = 10

frontal_channels = ['RAW_AF7', 'RAW_AF8']
SFREQ = effective_sfreq

# 等間隔タイムスタンプ
start_time = df['TimeStamp'].iloc[0]
uniform_timestamps = pd.date_range(
    start=start_time,
    periods=len(df),
    freq=pd.Timedelta(seconds=1.0 / effective_sfreq)
)

# ============================================================
# 2. バンドパワー計算 (Session1と同じ手法)
# ============================================================

def compute_linear_band_power(df, channels, sfreq, band, window_sec=2.0, step_sec=1.0):
    """チャンネル平均のリニアバンドパワーを計算"""
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    ch_powers = []
    for ch in channels:
        data = df[ch].values.astype(float)
        powers = []
        for start in range(0, len(data) - window_samples, step_samples):
            seg = data[start:start + window_samples]
            nperseg = min(window_samples, int(sfreq))
            freqs, psd = signal.welch(seg, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
            mask = (freqs >= band[0]) & (freqs <= band[1])
            powers.append(np.trapz(psd[mask], freqs[mask]))
        ch_powers.append(powers)
    return np.mean(ch_powers, axis=0)

print("\nバンドパワー計算中...")
alpha_linear = compute_linear_band_power(df, frontal_channels, SFREQ, (8, 13))
beta_linear = compute_linear_band_power(df, frontal_channels, SFREQ, (13, 25))
theta_linear = compute_linear_band_power(df, frontal_channels, SFREQ, (4, 8))
delta_linear = compute_linear_band_power(df, frontal_channels, SFREQ, (1, 4))

# dBスケール
alpha_db = 10 * np.log10(alpha_linear + 1e-12)
beta_db = 10 * np.log10(beta_linear + 1e-12)
theta_db = 10 * np.log10(theta_linear + 1e-12)

# 秒単位のタイムベクトル
time_sec = np.arange(len(alpha_linear)) * 1.0 + 1.0

print(f"バンドパワーポイント数: {len(alpha_linear)}")
print(f"タイム範囲: {time_sec[0]:.1f}s - {time_sec[-1]:.1f}s")

# ============================================================
# 3. Session1アルゴリズムをそのまま適用
# ============================================================
print("\n" + "=" * 60)
print("Session1アルゴリズムの適用")
print("=" * 60)

# Alpha/Beta比 (リニアスケール)
alpha_beta_ratio = alpha_linear / (beta_linear + 1e-12)

# スムージング (移動平均 4秒)
smooth_window = 4
alpha_beta_smooth = pd.Series(alpha_beta_ratio).rolling(
    smooth_window, center=True, min_periods=1
).mean().values

# キャリブレーション期間 (最初の120秒)
CALIBRATION_SEC = 120
calib_mask = time_sec <= CALIBRATION_SEC
calib_ratio = alpha_beta_smooth[calib_mask]

calib_median = np.median(calib_ratio)
calib_mad = np.median(np.abs(calib_ratio - calib_median))

print(f"\nキャリブレーション (0-{CALIBRATION_SEC}s):")
print(f"  median: {calib_median:.3f}")
print(f"  MAD:    {calib_mad:.3f}")

# Session1のパラメータ: kc=0.05, ka=1.8
calm_th = calib_median + 0.05 * calib_mad
active_th = calib_median - 1.8 * calib_mad

print(f"\n閾値 (Session1パラメータ):")
print(f"  Calm threshold  (median + 0.05*MAD): {calm_th:.3f}")
print(f"  Active threshold (median - 1.8*MAD): {active_th:.3f}")

# セッション期間
med_mask = time_sec > CALIBRATION_SEC
meditation_time_sec = time_sec[med_mask]
meditation_ratio = alpha_beta_smooth[med_mask]

session_duration = meditation_time_sec[-1] - meditation_time_sec[0]
print(f"\n瞑想セッション: {CALIBRATION_SEC}s〜{time_sec[-1]:.0f}s "
      f"(有効 {session_duration:.0f}s = {session_duration/60:.1f}min)")

# Mind State分類
def classify_mind_state(ratio, calm_th, active_th):
    states = np.full(len(ratio), 'Neutral', dtype=object)
    states[ratio >= calm_th] = 'Calm'
    states[ratio <= active_th] = 'Active'
    return states

meditation_states = classify_mind_state(meditation_ratio, calm_th, active_th)

step = meditation_time_sec[1] - meditation_time_sec[0] if len(meditation_time_sec) > 1 else 1.0
active_sec = np.sum(meditation_states == 'Active') * step
neutral_sec = np.sum(meditation_states == 'Neutral') * step
calm_sec = np.sum(meditation_states == 'Calm') * step
calm_pct = calm_sec / (active_sec + neutral_sec + calm_sec) * 100

print(f"\nMind State分類結果 (Session1アルゴリズム):")
print(f"  Active:  {active_sec:.0f}s  (アプリ: {TARGET['active_sec']}s, 差: {active_sec - TARGET['active_sec']:+.0f}s)")
print(f"  Neutral: {neutral_sec:.0f}s  (アプリ: {TARGET['neutral_sec']}s, 差: {neutral_sec - TARGET['neutral_sec']:+.0f}s)")
print(f"  Calm:    {calm_sec:.0f}s  (アプリ: {TARGET['calm_sec']}s, 差: {calm_sec - TARGET['calm_sec']:+.0f}s)")
print(f"  Calm%:   {calm_pct:.0f}%  (アプリ: {TARGET['calm_pct']}%)")

# ============================================================
# 4. Mind State パラメータの最適化探索
# ============================================================
print("\n" + "=" * 60)
print("Mind State パラメータ最適化")
print("=" * 60)

best_ms_error = float('inf')
best_ms_params = None

for kc in np.arange(-0.5, 1.5, 0.05):
    for ka in np.arange(0.5, 4.0, 0.1):
        c_th = calib_median + kc * calib_mad
        a_th = calib_median - ka * calib_mad
        if c_th <= a_th:
            continue

        states = classify_mind_state(meditation_ratio, c_th, a_th)
        a_s = np.sum(states == 'Active') * step
        n_s = np.sum(states == 'Neutral') * step
        c_s = np.sum(states == 'Calm') * step

        err = abs(a_s - TARGET['active_sec']) + abs(n_s - TARGET['neutral_sec']) + abs(c_s - TARGET['calm_sec'])
        if err < best_ms_error:
            best_ms_error = err
            best_ms_params = {
                'kc': kc, 'ka': ka,
                'calm_th': c_th, 'active_th': a_th,
                'active_sec': a_s, 'neutral_sec': n_s, 'calm_sec': c_s,
            }

print(f"\n最適パラメータ (Session2):")
print(f"  kc = {best_ms_params['kc']:.2f}, ka = {best_ms_params['ka']:.2f}")
print(f"  Calm threshold:   {best_ms_params['calm_th']:.3f}")
print(f"  Active threshold: {best_ms_params['active_th']:.3f}")
print(f"  Active:  {best_ms_params['active_sec']:.0f}s (target: {TARGET['active_sec']}s)")
print(f"  Neutral: {best_ms_params['neutral_sec']:.0f}s (target: {TARGET['neutral_sec']}s)")
print(f"  Calm:    {best_ms_params['calm_sec']:.0f}s (target: {TARGET['calm_sec']}s)")
print(f"  Total error: {best_ms_error:.0f}s")

# Session1のパラメータ (kc=0.05, ka=1.8) での結果
print(f"\nSession1パラメータ (kc=0.05, ka=1.8) での結果:")
print(f"  Active:  {active_sec:.0f}s, Neutral: {neutral_sec:.0f}s, Calm: {calm_sec:.0f}s")
s1_err = abs(active_sec - TARGET['active_sec']) + abs(neutral_sec - TARGET['neutral_sec']) + abs(calm_sec - TARGET['calm_sec'])
print(f"  Total error: {s1_err:.0f}s")

# 最終的に使用するパラメータの決定
# Session1のパラメータで十分なら、Session1を採用
USE_OPTIMIZED = s1_err > 20  # 20秒以上のずれがあれば最適化版を使用
if USE_OPTIMIZED:
    final_calm_th = best_ms_params['calm_th']
    final_active_th = best_ms_params['active_th']
    print(f"\n→ Session1パラメータのずれが大きいため最適化版を使用")
else:
    final_calm_th = calm_th
    final_active_th = active_th
    print(f"\n→ Session1パラメータで十分な精度")

final_states = classify_mind_state(meditation_ratio, final_calm_th, final_active_th)
final_active = np.sum(final_states == 'Active') * step
final_neutral = np.sum(final_states == 'Neutral') * step
final_calm = np.sum(final_states == 'Calm') * step

# ============================================================
# 5. Birds検出
# ============================================================
print("\n" + "=" * 60)
print("Birds検出分析")
print("=" * 60)
print(f"ターゲット: {TARGET['birds']}個")

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

# Session1アルゴリズム: Deep Calm パーセンタイル + 持続時間
# Session1では上位38%タイル、最小10-13秒だった
# しかしBirds=29は非常に多い → パーセンタイルか持続時間が異なるはず

# まずSession1パラメータで試す
deep_pct_s1 = 62  # 上位38%タイル
min_dur_s1 = 10
deep_th_s1 = np.percentile(meditation_ratio, deep_pct_s1)
birds_s1 = count_sustained_events(meditation_ratio, deep_th_s1, meditation_time_sec, min_dur_s1)
print(f"\nSession1パラメータ (pct={deep_pct_s1}, dur={min_dur_s1}s):")
print(f"  Deep Calm threshold: {deep_th_s1:.3f}")
print(f"  検出数: {len(birds_s1)} (target: {TARGET['birds']})")

# パラメータ探索
print(f"\nパラメータ探索...")
best_bird_error = float('inf')
best_bird_params = None

for deep_pct in np.arange(30, 90, 1):
    deep_th = np.percentile(meditation_ratio, deep_pct)
    for min_dur in np.arange(1, 20, 0.5):
        events = count_sustained_events(
            meditation_ratio, deep_th, meditation_time_sec, min_dur
        )
        n_events = len(events)
        error = abs(n_events - TARGET['birds'])
        if error < best_bird_error or (
            error == best_bird_error and
            best_bird_params and
            abs(min_dur - 5) < abs(best_bird_params['min_duration'] - 5)  # 5秒に近い方を優先
        ):
            best_bird_error = error
            best_bird_params = {
                'deep_pct': deep_pct,
                'deep_threshold': deep_th,
                'min_duration': min_dur,
                'n_events': n_events,
            }

print(f"\n最適Birds パラメータ (Session2):")
print(f"  Deep Calm閾値: Alpha/Beta ratio >= {best_bird_params['deep_threshold']:.3f} "
      f"(上位{100-best_bird_params['deep_pct']:.0f}%)")
print(f"  最小持続時間: {best_bird_params['min_duration']}秒")
print(f"  検出数: {best_bird_params['n_events']} (ターゲット: {TARGET['birds']})")

bird_events = count_sustained_events(
    meditation_ratio,
    best_bird_params['deep_threshold'],
    meditation_time_sec,
    best_bird_params['min_duration'],
)

# Birds代替仮説: Calmの秒数ベース (Calm状態N秒ごとに1 Bird)
# Birds=29, Calm=403s → 403/29 ≈ 13.9s per bird
print(f"\n--- Birds代替仮説: Calm秒数ベース ---")
calm_mask_med = final_states == 'Calm'
calm_secs = np.sum(calm_mask_med) * step
print(f"  Calm合計: {calm_secs:.0f}s")
print(f"  Calm / Birds = {calm_secs / TARGET['birds']:.1f}s per bird")

# Session1: Calm=369s, Birds=5 → 73.8s per bird (全然違う)
print(f"  Session1: Calm=369s / Birds=5 = 73.8s per bird")
print(f"  → 秒数ベースは不整合")

# Birds代替仮説2: Calm状態中のDeep Calmスパイク数
# 閾値を超えるたびにカウント（連続性は不要）
print(f"\n--- Birds代替仮説2: Calm中のDeep Calmスパイク ---")
# meditation_ratioがCalm閾値を超えている区間で、さらに高い閾値を超えるスパイク回数
calm_indices = np.where(calm_mask_med)[0]
if len(calm_indices) > 0:
    calm_ratio = meditation_ratio[calm_indices]
    for spike_pct in [60, 65, 70, 75, 80, 85, 90]:
        spike_th = np.percentile(calm_ratio, spike_pct)
        # スパイク = 閾値を下から上に crossing
        above = calm_ratio >= spike_th
        crossings = np.sum(np.diff(above.astype(int)) == 1)
        print(f"  pct={spike_pct} (th={spike_th:.2f}): crossings={crossings}")

# Birds代替仮説3: Calm状態のセグメント数（中断で分割）
print(f"\n--- Birds代替仮説3: Calmセグメント数 ---")
calm_binary = (final_states == 'Calm').astype(int)
# 各Calmセグメントを検出
calm_segments = []
in_calm = False
seg_start = None
for i in range(len(calm_binary)):
    if calm_binary[i] and not in_calm:
        in_calm = True
        seg_start = i
    elif not calm_binary[i] and in_calm:
        in_calm = False
        seg_dur = (i - seg_start) * step
        calm_segments.append({'start': seg_start, 'end': i-1, 'duration': seg_dur})
if in_calm:
    calm_segments.append({'start': seg_start, 'end': len(calm_binary)-1,
                          'duration': (len(calm_binary) - seg_start) * step})

print(f"  Calmセグメント数: {len(calm_segments)}")
for i, seg in enumerate(calm_segments[:10]):
    t_start = meditation_time_sec[seg['start']] / 60
    print(f"    Seg {i+1}: {t_start:.1f}min, {seg['duration']:.1f}s")
if len(calm_segments) > 10:
    print(f"    ... (残り{len(calm_segments)-10}セグメント)")

# Birds代替仮説4: Muse Appの基準秒数（例: 10秒Calm = 1 Bird）
print(f"\n--- Birds代替仮説4: N秒Calm = 1 Bird ---")
for base_sec in [5, 8, 10, 12, 13, 14, 15, 20]:
    total_birds = 0
    for seg in calm_segments:
        total_birds += int(seg['duration'] / base_sec)
    print(f"  {base_sec}秒/bird: {total_birds} birds (target: {TARGET['birds']})")

# Session1も同じ計算 (後で比較用)
print(f"\n  Session1参考: Calm=369s / N秒, Birds=5")
for base_sec in [5, 8, 10, 12, 13, 14, 15, 20, 60, 70, 75]:
    print(f"    {base_sec}秒/bird: {int(369 / base_sec)} birds (target: 5)")

# ============================================================
# 5b. Birds新仮説: キャリブレーション基準のDeep Calm
# ============================================================
print(f"\n--- Birds新仮説5: キャリブレーション基準Deep Calm閾値 ---")

# Calmではなく、キャリブレーション基準でDeep Calmを定義
# Deep Calm = median + k * MAD (kを探索)
for k_deep in np.arange(0.0, 3.0, 0.1):
    deep_th_calib = calib_median + k_deep * calib_mad
    deep_calm_mask = meditation_ratio >= deep_th_calib

    # deep calm の持続セグメント
    segments = []
    in_seg = False
    seg_start = None
    for i in range(len(deep_calm_mask)):
        if deep_calm_mask[i] and not in_seg:
            in_seg = True
            seg_start = i
        elif not deep_calm_mask[i] and in_seg:
            in_seg = False
            seg_dur = (i - seg_start) * step
            segments.append(seg_dur)
    if in_seg:
        segments.append((len(deep_calm_mask) - seg_start) * step)

    # N秒持続を1 birdとしてカウント
    for min_d in [3, 5, 7, 10]:
        n_birds = sum(1 for s in segments if s >= min_d)
        if abs(n_birds - TARGET['birds']) <= 2:
            print(f"  k={k_deep:.1f} (th={deep_th_calib:.2f}), min_dur={min_d}s: {n_birds} birds ★")

# ============================================================
# 5c. Birds: 持続時間を足し合わせる方式
# ============================================================
print(f"\n--- Birds仮説6: DeepCalm累積時間 / base_sec ---")
for k_deep in np.arange(0.0, 2.0, 0.2):
    deep_th_calib = calib_median + k_deep * calib_mad
    deep_calm_time = np.sum(meditation_ratio >= deep_th_calib) * step
    for base_sec in [10, 12, 13, 14, 15, 20]:
        n_birds = int(deep_calm_time / base_sec)
        if abs(n_birds - TARGET['birds']) <= 2:
            print(f"  k={k_deep:.1f} (th={deep_th_calib:.2f}), deep_calm_time={deep_calm_time:.0f}s, "
                  f"base={base_sec}s: {n_birds} birds ★")


# ============================================================
# 6. Recoveries検出
# ============================================================
print("\n" + "=" * 60)
print("Recoveries検出分析")
print("=" * 60)
print(f"ターゲット: {TARGET['recoveries']}個")

# Session1アルゴリズム: ステップ遷移検出
best_recovery_error = float('inf')
best_recovery_params = None

for lookback_sec in [3, 5, 7, 10, 15]:
    for lookforward_sec in [3, 5, 7, 10, 15]:
        for drop_pct in np.arange(15, 55, 5):
            for rise_pct in np.arange(50, 90, 5):
                drop_th = np.percentile(meditation_ratio, drop_pct)
                rise_th = np.percentile(meditation_ratio, rise_pct)

                if rise_th <= drop_th:
                    continue

                lookback_idx = int(lookback_sec / step)
                lookforward_idx = int(lookforward_sec / step)

                recoveries = []
                min_gap = 30
                min_gap_idx = int(min_gap / step)

                for i in range(lookback_idx, len(meditation_ratio) - lookforward_idx):
                    pre_window = meditation_ratio[i - lookback_idx:i]
                    post_window = meditation_ratio[i:i + lookforward_idx]

                    if np.mean(pre_window) <= drop_th and np.mean(post_window) >= rise_th:
                        if not recoveries or (i - recoveries[-1]) >= min_gap_idx:
                            recoveries.append(i)

                n_rec = len(recoveries)
                error = abs(n_rec - TARGET['recoveries'])

                if error < best_recovery_error:
                    best_recovery_error = error
                    best_recovery_params = {
                        'lookback_sec': lookback_sec,
                        'lookforward_sec': lookforward_sec,
                        'drop_pct': drop_pct,
                        'rise_pct': rise_pct,
                        'drop_th': drop_th,
                        'rise_th': rise_th,
                        'n_recoveries': n_rec,
                        'recovery_indices': recoveries,
                    }

print(f"\n最適Recovery パラメータ:")
print(f"  Lookback: {best_recovery_params['lookback_sec']}s")
print(f"  Lookforward: {best_recovery_params['lookforward_sec']}s")
print(f"  Drop閾値: ratio <= {best_recovery_params['drop_th']:.3f} "
      f"(下位{best_recovery_params['drop_pct']:.0f}%)")
print(f"  Rise閾値: ratio >= {best_recovery_params['rise_th']:.3f} "
      f"(上位{100-best_recovery_params['rise_pct']:.0f}%)")
print(f"  検出数: {best_recovery_params['n_recoveries']} (ターゲット: {TARGET['recoveries']})")

print(f"\nRecoveryイベント詳細:")
for i, idx in enumerate(best_recovery_params['recovery_indices']):
    t_min = meditation_time_sec[idx] / 60
    lb_idx = int(best_recovery_params['lookback_sec'] / step)
    lf_idx = int(best_recovery_params['lookforward_sec'] / step)
    ratio_before = np.mean(meditation_ratio[max(0, idx - lb_idx):idx])
    ratio_after = np.mean(meditation_ratio[idx:min(len(meditation_ratio), idx + lf_idx)])
    print(f"  Recovery {i+1}: {t_min:.1f}min (ratio: {ratio_before:.2f} → {ratio_after:.2f})")

# Recoveries代替: Active状態からNeutral/Calm状態への遷移回数
print(f"\n--- Recovery代替: Active→Calm/Neutral 遷移 ---")
recovery_count = 0
recovery_times = []
for i in range(1, len(final_states)):
    if final_states[i-1] == 'Active' and final_states[i] in ('Calm', 'Neutral'):
        t = meditation_time_sec[i] / 60
        if not recovery_times or (t - recovery_times[-1]) > 0.5:  # 30秒間隔
            recovery_count += 1
            recovery_times.append(t)
print(f"  Active→Calm/Neutral遷移回数: {recovery_count} (target: {TARGET['recoveries']})")
for i, t in enumerate(recovery_times):
    print(f"    Recovery {i+1}: {t:.1f}min")

# ============================================================
# 7. 可視化
# ============================================================
print("\nグラフ生成中...")

fig, axes = plt.subplots(5, 1, figsize=(16, 22), sharex=True)

time_min = time_sec / 60
med_time_min = meditation_time_sec / 60

# (a) バンドパワー
ax = axes[0]
band_db_map = {'Alpha': alpha_db, 'Beta': beta_db, 'Theta': theta_db}
colors = {'Theta': 'blue', 'Alpha': 'green', 'Beta': 'orange'}
for band_name in ['Alpha', 'Beta', 'Theta']:
    ax.plot(time_min, band_db_map[band_name], label=band_name, color=colors[band_name], alpha=0.7)
ax.set_ylabel('Power (dB)')
ax.set_title('Session2: Band Power Time Series (Frontal: AF7, AF8)')
ax.legend()
ax.axvline(CALIBRATION_SEC / 60, color='gray', linestyle='--', alpha=0.5, label='Calibration end')
ax.grid(True, alpha=0.3)

# (b) Alpha/Beta比
ax = axes[1]
ax.plot(med_time_min, meditation_ratio, alpha=0.3, color='gray', label='Raw')
smoothed = pd.Series(meditation_ratio).rolling(5, center=True, min_periods=1).mean()
ax.plot(med_time_min, smoothed, color='blue', linewidth=1.5, label='Smoothed (4s)')
ax.axhline(final_calm_th, color='green', linestyle='--', alpha=0.7, label=f'Calm threshold ({final_calm_th:.2f})')
ax.axhline(final_active_th, color='red', linestyle='--', alpha=0.7, label=f'Active threshold ({final_active_th:.2f})')

# Birds マーカー
for ev in bird_events:
    mid = (ev['start_sec'] + ev['end_sec']) / 2 / 60
    ax.axvspan(ev['start_sec'] / 60, ev['end_sec'] / 60, alpha=0.15, color='cyan', zorder=0)

# Recoveries マーカー
for idx in best_recovery_params['recovery_indices']:
    t = meditation_time_sec[idx] / 60
    ax.axvline(t, color='gold', linewidth=2, alpha=0.7, linestyle='-')
    ax.plot(t, meditation_ratio[idx], '*', color='gold',
            markersize=15, markeredgecolor='darkorange', markeredgewidth=1)

ax.set_ylabel('Alpha/Beta Ratio')
ax.set_title(f'Session2: Alpha/Beta Ratio (Birds detected={len(bird_events)}, target={TARGET["birds"]})')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# (c) Mind State分類
ax = axes[2]
state_colors = {'Active': 'red', 'Neutral': 'gray', 'Calm': 'green'}
for state in ['Active', 'Neutral', 'Calm']:
    mask = final_states == state
    ax.fill_between(med_time_min, 0, 1, where=mask,
                    alpha=0.5, color=state_colors[state], label=state)
ax.set_ylabel('Mind State')
ax.set_title(f'Session2: Mind State (Active={final_active:.0f}s, Neutral={final_neutral:.0f}s, '
             f'Calm={final_calm:.0f}s)')
ax.legend()
ax.set_yticks([])

# (d) Alpha vs Beta
ax = axes[3]
med_alpha_db = alpha_db[med_mask]
med_beta_db = beta_db[med_mask]
alpha_sm = pd.Series(med_alpha_db).rolling(10, center=True, min_periods=1).mean()
beta_sm = pd.Series(med_beta_db).rolling(10, center=True, min_periods=1).mean()
ax.plot(med_time_min, alpha_sm, color='green', linewidth=1.5, label='Alpha (8-13 Hz)')
ax.plot(med_time_min, beta_sm, color='orange', linewidth=1.5, label='Beta (13-25 Hz)')
ax.set_ylabel('Power (dB)')
ax.set_title('Session2: Alpha vs Beta Power (Smoothed)')
ax.legend()
ax.grid(True, alpha=0.3)

# (e) Alpha/Beta比のヒストグラム + 閾値
ax = axes[4]
ax.hist(meditation_ratio, bins=100, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(final_calm_th, color='green', linewidth=2, label=f'Calm th={final_calm_th:.2f}')
ax.axvline(final_active_th, color='red', linewidth=2, label=f'Active th={final_active_th:.2f}')
ax.axvline(calib_median, color='blue', linewidth=2, linestyle=':', label=f'Calib median={calib_median:.2f}')
ax.set_xlabel('Alpha/Beta Ratio')
ax.set_ylabel('Count')
ax.set_title('Session2: Alpha/Beta Ratio Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = OUTPUT_DIR / 'session2_analysis.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"グラフ保存: {fig_path}")
plt.close()

# ============================================================
# 8. Session1 vs Session2 比較サマリー
# ============================================================
print("\n" + "=" * 60)
print("Session1 vs Session2 比較サマリー")
print("=" * 60)

print(f"""
| 項目                  | Session1        | Session2        |
|-----------------------|-----------------|-----------------|
| Duration              | 15min           | 10min           |
| Score                 | 46              | 73              |
| sfreq                 | ~51.5 Hz        | ~{effective_sfreq:.1f} Hz       |
| Calib median          | 20.98           | {calib_median:.2f}          |
| Calib MAD             | 2.20            | {calib_mad:.2f}           |
| Calm th               | 21.09           | {final_calm_th:.2f}          |
| Active th             | 17.02           | {final_active_th:.2f}          |
| Active (target/est)   | 19/19           | {TARGET['active_sec']}/{final_active:.0f}             |
| Neutral (target/est)  | 398/401         | {TARGET['neutral_sec']}/{final_neutral:.0f}           |
| Calm (target/est)     | 369/367         | {TARGET['calm_sec']}/{final_calm:.0f}           |
| Birds (target/est)    | 5/{len(bird_events)}             | {TARGET['birds']}/{len(bird_events)}             |
| Recoveries (target)   | 3/{best_recovery_params['n_recoveries']}             | {TARGET['recoveries']}/{best_recovery_params['n_recoveries']}             |
""")
