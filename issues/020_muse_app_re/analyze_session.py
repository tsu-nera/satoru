#!/usr/bin/env python
"""
Muse App リバースエンジニアリング分析スクリプト

Muse AppのOSCストリーミングデータから、以下を分析:
- Birds (鳥) イベントの検出アルゴリズム推定
- Recoveries (回復) イベントの検出アルゴリズム推定
- Mind State (Active/Neutral/Calm) 分類

アプリ表示値:
- Birds: 5
- Recoveries: 3
- Calm: 46%, Active: 19s, Neutral: 6m38s, Calm: 6m9s
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import signal

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# ============================================================
# 1. データ読み込み
# ============================================================
CSV_PATH = Path(__file__).parent / 'session' / 'muse_app_2026-03-04--08-05-52.csv'
OUTPUT_DIR = Path(__file__).parent

df = pd.read_csv(CSV_PATH)
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

# 実効サンプリングレート
duration_s = (df['TimeStamp'].iloc[-1] - df['TimeStamp'].iloc[0]).total_seconds()
effective_sfreq = len(df) / duration_s
print(f"データ: {len(df)} samples, {duration_s:.1f}s, effective {effective_sfreq:.1f} Hz")

# RAW EEGチャンネル
raw_channels = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
frontal_channels = ['RAW_AF7', 'RAW_AF8']

# 等間隔タイムスタンプを生成（バッチ送信のため元のタイムスタンプは不均一）
start_time = df['TimeStamp'].iloc[0]
uniform_timestamps = pd.date_range(
    start=start_time,
    periods=len(df),
    freq=pd.Timedelta(seconds=1.0 / effective_sfreq)
)

# ============================================================
# 2. バンドパワー計算 (Welch法, スライディングウィンドウ)
# ============================================================

SFREQ = effective_sfreq  # ~51.5 Hz

# ============================================================
# 2. バンドパワー計算 (Welch PSD → チャンネル平均, リニアスケール)
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

print("バンドパワー計算中...")
alpha_linear = compute_linear_band_power(df, frontal_channels, SFREQ, (8, 13))
beta_linear = compute_linear_band_power(df, frontal_channels, SFREQ, (13, 25))
theta_linear = compute_linear_band_power(df, frontal_channels, SFREQ, (4, 8))

# dB変換（グラフ用）
alpha_db = 10 * np.log10(alpha_linear + 1e-12)
beta_db = 10 * np.log10(beta_linear + 1e-12)
theta_db = 10 * np.log10(theta_linear + 1e-12)

# 秒単位のタイムベクトル
time_sec = np.arange(len(alpha_linear)) * 1.0 + 1.0

# ============================================================
# 3. Alpha/Beta比によるMind State分類
# ============================================================

# Alpha/Beta比 (リニアスケール)
alpha_beta_ratio = alpha_linear / (beta_linear + 1e-12)

# スムージング (移動平均 4秒)
smooth_window = 4
alpha_beta_smooth = pd.Series(alpha_beta_ratio).rolling(
    smooth_window, center=True, min_periods=1
).mean().values

# ============================================================
# キャリブレーション期間 (最初の120秒) から閾値を算出
# ============================================================
CALIBRATION_SEC = 120
calib_mask = time_sec <= CALIBRATION_SEC
calib_ratio = alpha_beta_smooth[calib_mask]

calib_median = np.median(calib_ratio)
calib_mad = np.median(np.abs(calib_ratio - calib_median))

print(f"\nキャリブレーション (0-{CALIBRATION_SEC}s):")
print(f"  median: {calib_median:.3f}")
print(f"  MAD:    {calib_mad:.3f}")

# 閾値設定: Calm ≈ median + 0.05*MAD, Active = median - 1.8*MAD
calm_th = calib_median + 0.05 * calib_mad  # kc ≈ 0
active_th = calib_median - 1.8 * calib_mad

print(f"\n閾値:")
print(f"  Calm threshold  (≈ median):          {calm_th:.3f}")
print(f"  Active threshold (median - 1.8*MAD): {active_th:.3f}")

# セッション期間 (キャリブレーション後)
med_mask = time_sec > CALIBRATION_SEC
meditation_time_sec = time_sec[med_mask]
meditation_ratio = alpha_beta_smooth[med_mask]
meditation_alpha_db = alpha_db[med_mask]
meditation_beta_db = beta_db[med_mask]
meditation_theta_db = theta_db[med_mask]

session_duration = meditation_time_sec[-1] - meditation_time_sec[0]
print(f"\n瞑想セッション: {CALIBRATION_SEC}s～{time_sec[-1]:.0f}s "
      f"(有効 {session_duration:.0f}s = {session_duration/60:.1f}min)")

# Mind State分類
def classify_mind_state(ratio, calm_th, active_th):
    states = np.full(len(ratio), 'Neutral', dtype=object)
    states[ratio >= calm_th] = 'Calm'
    states[ratio <= active_th] = 'Active'
    return states

meditation_states = classify_mind_state(meditation_ratio, calm_th, active_th)

# 状態ごとの時間を計算
step = meditation_time_sec[1] - meditation_time_sec[0] if len(meditation_time_sec) > 1 else 1.0
active_sec = np.sum(meditation_states == 'Active') * step
neutral_sec = np.sum(meditation_states == 'Neutral') * step
calm_sec = np.sum(meditation_states == 'Calm') * step

print(f"\nMind State分類結果:")
print(f"  Active:  {active_sec:.0f}s  (アプリ: 19s)")
print(f"  Neutral: {neutral_sec:.0f}s  (アプリ: 398s = 6m38s)")
print(f"  Calm:    {calm_sec:.0f}s  (アプリ: 369s = 6m9s)")
print(f"  Calm%:   {calm_sec/(active_sec+neutral_sec+calm_sec)*100:.0f}%  (アプリ: 46%)")

# ============================================================
# 4. Birds検出アルゴリズム推定
# ============================================================
# Birds = 持続的な深いCalm状態
# 仮説: Alpha/Beta比が高い状態が一定秒数以上続く

print("\n" + "=" * 60)
print("Birds検出分析")
print("=" * 60)

# Calmの上位層 (Deep Calm) を検出
# birdsは5回なので、特に深いcalm期間を探す

# Alpha/Beta比のさらに上位パーセンタイル
deep_calm_threshold_candidates = np.arange(60, 95, 1)
min_duration_candidates = np.arange(3, 15, 1)  # 最小持続秒数

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

    # 末尾チェック
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

# Birds = 5 に最も近い組み合わせを探索
TARGET_BIRDS = 5
best_bird_error = float('inf')
best_bird_params = None

for deep_pct in deep_calm_threshold_candidates:
    deep_th = np.percentile(meditation_ratio, deep_pct)
    for min_dur in min_duration_candidates:
        events = count_sustained_events(
            meditation_ratio, deep_th, meditation_time_sec, min_dur
        )
        n_events = len(events)
        error = abs(n_events - TARGET_BIRDS)
        if error < best_bird_error or (error == best_bird_error and min_dur > (best_bird_params or {}).get('min_duration', 0)):
            best_bird_error = error
            best_bird_params = {
                'deep_pct': deep_pct,
                'deep_threshold': deep_th,
                'min_duration': min_dur,
                'n_events': n_events,
            }

print(f"\n最適Birds パラメータ:")
print(f"  Deep Calm閾値: Alpha/Beta ratio >= {best_bird_params['deep_threshold']:.3f} "
      f"(上位{100-best_bird_params['deep_pct']:.0f}%)")
print(f"  最小持続時間: {best_bird_params['min_duration']}秒")
print(f"  検出数: {best_bird_params['n_events']} (ターゲット: {TARGET_BIRDS})")

bird_events = count_sustained_events(
    meditation_ratio,
    best_bird_params['deep_threshold'],
    meditation_time_sec,
    best_bird_params['min_duration'],
)

print(f"\nBirdイベント詳細:")
for i, ev in enumerate(bird_events):
    start_min = ev['start_sec'] / 60
    end_min = ev['end_sec'] / 60
    print(f"  Bird {i+1}: {start_min:.1f}min - {end_min:.1f}min "
          f"(duration: {ev['duration']:.1f}s, peak ratio: {ev['peak_ratio']:.2f})")


# ============================================================
# 5. Recoveries検出アルゴリズム推定
# ============================================================
# Recovery = Active/distracted → Calm への遷移

print("\n" + "=" * 60)
print("Recoveries検出分析")
print("=" * 60)

TARGET_RECOVERIES = 3

# 方法: Activeまたは低Alpha/Beta期間からCalmへの遷移を検出
# スムージングした比率の急激な上昇を検出

# Alpha/Beta比の変化率
ratio_diff = np.diff(meditation_ratio, prepend=meditation_ratio[0])

# スムージング
ratio_diff_smooth = pd.Series(ratio_diff).rolling(
    3, center=True, min_periods=1
).mean().values

# Recovery候補: 比率が急上昇し、直前がNeutral以下、直後がCalm
# 様々なパラメータで探索
best_recovery_error = float('inf')
best_recovery_params = None

for lookback_sec in [3, 5, 7, 10, 15]:
    for lookforward_sec in [3, 5, 7, 10, 15]:
        for drop_pct in np.arange(20, 60, 5):
            for rise_pct in np.arange(50, 90, 5):
                drop_th = np.percentile(meditation_ratio, drop_pct)
                rise_th = np.percentile(meditation_ratio, rise_pct)

                if rise_th <= drop_th:
                    continue

                lookback_idx = int(lookback_sec / step)
                lookforward_idx = int(lookforward_sec / step)

                recoveries = []
                min_gap = 30  # 最小イベント間隔 (秒)
                min_gap_idx = int(min_gap / step)

                for i in range(lookback_idx, len(meditation_ratio) - lookforward_idx):
                    # 直前がdrop_th以下
                    pre_window = meditation_ratio[i - lookback_idx:i]
                    # 直後がrise_th以上
                    post_window = meditation_ratio[i:i + lookforward_idx]

                    if np.mean(pre_window) <= drop_th and np.mean(post_window) >= rise_th:
                        # 最小間隔チェック
                        if not recoveries or (i - recoveries[-1]) >= min_gap_idx:
                            recoveries.append(i)

                n_rec = len(recoveries)
                error = abs(n_rec - TARGET_RECOVERIES)

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
print(f"  検出数: {best_recovery_params['n_recoveries']} (ターゲット: {TARGET_RECOVERIES})")

print(f"\nRecoveryイベント詳細:")
for i, idx in enumerate(best_recovery_params['recovery_indices']):
    t_min = meditation_time_sec[idx] / 60
    ratio_before = np.mean(
        meditation_ratio[max(0, idx - int(best_recovery_params['lookback_sec'] / step)):idx]
    )
    ratio_after = np.mean(
        meditation_ratio[idx:min(len(meditation_ratio),
                                  idx + int(best_recovery_params['lookforward_sec'] / step))]
    )
    print(f"  Recovery {i+1}: {t_min:.1f}min "
          f"(ratio: {ratio_before:.2f} → {ratio_after:.2f})")


# ============================================================
# 6. 可視化
# ============================================================
print("\nグラフ生成中...")

fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)

# タイムベクトル (分)
time_min = time_sec / 60
med_time_min = meditation_time_sec / 60

# (a) バンドパワー時系列
ax = axes[0]
band_db_map = {'Alpha': alpha_db, 'Beta': beta_db, 'Theta': theta_db}
colors = {'Theta': 'blue', 'Alpha': 'green', 'Beta': 'orange'}
for band_name in ['Alpha', 'Beta', 'Theta']:
    ax.plot(time_min, band_db_map[band_name], label=band_name, color=colors[band_name], alpha=0.7)
ax.set_ylabel('Power (dB)')
ax.set_title('Band Power Time Series (Frontal: AF7, AF8)')
ax.legend()
ax.axvline(CALIBRATION_SEC / 60, color='gray', linestyle='--', alpha=0.5, label='Calibration end')
ax.grid(True, alpha=0.3)

# (b) Alpha/Beta比
ax = axes[1]
ax.plot(med_time_min, meditation_ratio, alpha=0.3, color='gray', label='Raw')
ax.plot(med_time_min, pd.Series(meditation_ratio).rolling(5, center=True, min_periods=1).mean(),
        color='blue', linewidth=1.5, label='Smoothed (4s)')
ax.axhline(calm_th, color='green', linestyle='--', alpha=0.7, label=f'Calm threshold ({calm_th:.2f})')
ax.axhline(active_th, color='red', linestyle='--', alpha=0.7, label=f'Active threshold ({active_th:.2f})')

# Birds マーカー
for ev in bird_events:
    mid = (ev['start_sec'] + ev['end_sec']) / 2 / 60
    ax.axvspan(ev['start_sec'] / 60, ev['end_sec'] / 60,
               alpha=0.15, color='cyan', zorder=0)
    ax.plot(mid, np.max(meditation_ratio) * 1.02, 'v', color='cyan',
            markersize=12, markeredgecolor='darkblue', markeredgewidth=1.5)
    ax.text(mid, np.max(meditation_ratio) * 1.06, 'B', fontsize=8,
            ha='center', va='bottom', fontweight='bold', color='darkblue')

# Recoveries マーカー
for idx in best_recovery_params['recovery_indices']:
    t = meditation_time_sec[idx] / 60
    ax.axvline(t, color='gold', linewidth=2, alpha=0.7, linestyle='-')
    ax.plot(t, meditation_ratio[idx], '*', color='gold',
            markersize=15, markeredgecolor='darkorange', markeredgewidth=1)
    ax.text(t, meditation_ratio[idx] * 1.05, 'R', fontsize=8,
            ha='center', va='bottom', fontweight='bold', color='darkorange')

ax.set_ylabel('Alpha/Beta Ratio')
ax.set_title('Alpha/Beta Ratio with Birds & Recoveries')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# (c) Mind State分類
ax = axes[2]
state_colors = {'Active': 'red', 'Neutral': 'gray', 'Calm': 'green'}
for state in ['Active', 'Neutral', 'Calm']:
    mask = meditation_states == state
    ax.fill_between(med_time_min, 0, 1, where=mask,
                    alpha=0.5, color=state_colors[state], label=state)
ax.set_ylabel('Mind State')
ax.set_title(f'Mind State Classification '
             f'(Active={active_sec:.0f}s, Neutral={neutral_sec:.0f}s, Calm={calm_sec:.0f}s)')
ax.legend()
ax.set_yticks([])

# (d) Alpha power (dBスケール)
ax = axes[3]
alpha_smooth = pd.Series(meditation_alpha_db).rolling(10, center=True, min_periods=1).mean()
ax.plot(med_time_min, alpha_smooth, color='green', linewidth=1.5, label='Alpha (8-13 Hz)')
beta_smooth = pd.Series(meditation_beta_db).rolling(10, center=True, min_periods=1).mean()
ax.plot(med_time_min, beta_smooth, color='orange', linewidth=1.5, label='Beta (13-25 Hz)')

for ev in bird_events:
    ax.axvspan(ev['start_sec'] / 60, ev['end_sec'] / 60, alpha=0.15, color='cyan')

ax.set_ylabel('Power (dB)')
ax.set_title('Alpha vs Beta Power (Smoothed)')
ax.legend()
ax.grid(True, alpha=0.3)

# (e) Theta power & Theta/Alpha ratio
ax = axes[4]
theta_alpha_ratio = theta_linear[med_mask] / (alpha_linear[med_mask] + 1e-12)
theta_alpha_smooth = pd.Series(theta_alpha_ratio).rolling(
    10, center=True, min_periods=1
).mean()
ax.plot(med_time_min, theta_alpha_smooth, color='blue', linewidth=1.5,
        label='Theta/Alpha Ratio')
ax.set_ylabel('Theta/Alpha Ratio')
ax.set_title('Theta/Alpha Ratio (meditation depth indicator)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlabel('Time (minutes)')

plt.tight_layout()
fig_path = OUTPUT_DIR / 'analysis_results.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"グラフ保存: {fig_path}")
plt.close()

# ============================================================
# 7. 追加分析: Alphaパワーの絶対値ベースでのBirds検出
# ============================================================
print("\n" + "=" * 60)
print("追加分析: 絶対Alphaパワーベースの検出")
print("=" * 60)

# Alpha dBスムージング
alpha_smooth_arr = pd.Series(meditation_alpha_db).rolling(
    5, center=True, min_periods=1
).mean().values

# Alphaパワーの閾値でBirds検出
best_alpha_bird_error = float('inf')
best_alpha_bird_params = None

for pct in np.arange(55, 90, 1):
    alpha_th = np.percentile(alpha_smooth_arr, pct)
    for min_dur in [3, 5, 7, 8, 10, 12]:
        events = count_sustained_events(
            alpha_smooth_arr, alpha_th, meditation_time_sec, min_dur
        )
        error = abs(len(events) - TARGET_BIRDS)
        if error < best_alpha_bird_error:
            best_alpha_bird_error = error
            best_alpha_bird_params = {
                'pct': pct,
                'threshold': alpha_th,
                'min_duration': min_dur,
                'n_events': len(events),
                'events': events,
            }

print(f"\nAlphaパワーベースBirds:")
print(f"  閾値: {best_alpha_bird_params['threshold']:.1f} dB "
      f"(上位{100-best_alpha_bird_params['pct']:.0f}%)")
print(f"  最小持続時間: {best_alpha_bird_params['min_duration']}s")
print(f"  検出数: {best_alpha_bird_params['n_events']}")

for i, ev in enumerate(best_alpha_bird_params['events']):
    print(f"  Bird {i+1}: {ev['start_sec']/60:.1f}-{ev['end_sec']/60:.1f}min "
          f"(duration: {ev['duration']:.1f}s)")


# ============================================================
# 8. 結果サマリー
# ============================================================
print("\n" + "=" * 60)
print("推定アルゴリズムまとめ")
print("=" * 60)

print("""
■ Mind State分類 (Active/Neutral/Calm)
  - 前頭部 (AF7, AF8) のAlpha(8-13Hz)/Beta(13-25Hz)パワー比を使用
  - 最初の約2分間がキャリブレーション期間
  - 比率の閾値で3状態に分類
  - スムージング: 5秒移動平均

■ Birds (鳥) 検出
  - Alpha/Beta比が上位パーセンタイルを超える「Deep Calm」状態が
    一定秒数以上持続した場合にBirdイベント発生
  - 推定パラメータ: 上位{bird_top}%、最小{bird_dur}秒持続

■ Recoveries (回復) 検出
  - Alpha/Beta比が低い状態(distracted)から高い状態(calm)への遷移
  - 直前{lb}秒の平均が低閾値以下、直後{lf}秒の平均が高閾値以上
  - 最小間隔: 30秒
  - 推定パラメータ: drop<{dpct}%ile, rise>{rpct}%ile
""".format(
    bird_top=int(100 - best_bird_params['deep_pct']),
    bird_dur=best_bird_params['min_duration'],
    lb=best_recovery_params['lookback_sec'],
    lf=best_recovery_params['lookforward_sec'],
    dpct=int(best_recovery_params['drop_pct']),
    rpct=int(best_recovery_params['rise_pct']),
))
