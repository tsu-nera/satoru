#!/usr/bin/env python3
"""
バースト区間詳細分析スクリプト

15分と27分付近に見られる特徴的なバーストの詳細分析
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import (
    load_mind_monitor_csv,
    prepare_mne_raw,
    calculate_psd,
)

# データパス
DATA_PATH = project_root / "data" / "mindMonitor_2025-12-08--07-38-33_3000223989795832513.csv"
OUTPUT_DIR = Path(__file__).parent

# バースト区間定義（分単位）
BURST_WINDOWS = [
    {"name": "burst_15min", "start_min": 13.5, "end_min": 16.5, "label": "15分付近バースト"},
    {"name": "burst_27min", "start_min": 25.5, "end_min": 28.5, "label": "27分付近バースト"},
]

# 比較用の通常区間
NORMAL_WINDOWS = [
    {"name": "normal_9min", "start_min": 7.5, "end_min": 10.5, "label": "通常区間 (9分)"},
    {"name": "normal_21min", "start_min": 19.5, "end_min": 22.5, "label": "通常区間 (21分)"},
]


def load_data():
    """データ読み込み"""
    print(f"Loading: {DATA_PATH}")
    df = load_mind_monitor_csv(DATA_PATH, filter_headband=False, warmup_seconds=60)

    # 開始時刻からの経過時間（分）を計算
    start_time = df['TimeStamp'].iloc[0]
    df['elapsed_min'] = (df['TimeStamp'] - start_time).dt.total_seconds() / 60

    return df


def extract_window(df, start_min, end_min):
    """指定区間のデータを抽出"""
    mask = (df['elapsed_min'] >= start_min) & (df['elapsed_min'] <= end_min)
    return df[mask].copy()


def analyze_motion_hsi(df_window, window_info):
    """モーション・HSIデータの分析"""
    result = {"window": window_info["label"]}

    # HSI品質
    hsi_cols = ['HSI_TP9', 'HSI_AF7', 'HSI_AF8', 'HSI_TP10']
    if all(col in df_window.columns for col in hsi_cols):
        hsi_mean = df_window[hsi_cols].mean().mean()
        hsi_bad_ratio = (df_window[hsi_cols] >= 4).mean().mean() * 100
        result['hsi_mean'] = hsi_mean
        result['hsi_bad_ratio'] = hsi_bad_ratio

    # 加速度
    accel_cols = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
    if all(col in df_window.columns for col in accel_cols):
        accel_magnitude = np.sqrt(
            df_window['Accelerometer_X']**2 +
            df_window['Accelerometer_Y']**2 +
            df_window['Accelerometer_Z']**2
        )
        result['accel_mean'] = accel_magnitude.mean()
        result['accel_std'] = accel_magnitude.std()
        result['accel_max'] = accel_magnitude.max()

    # ジャイロ
    gyro_cols = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']
    if all(col in df_window.columns for col in gyro_cols):
        gyro_magnitude = np.sqrt(
            df_window['Gyro_X']**2 +
            df_window['Gyro_Y']**2 +
            df_window['Gyro_Z']**2
        )
        result['gyro_mean'] = gyro_magnitude.mean()
        result['gyro_std'] = gyro_magnitude.std()
        result['gyro_max'] = gyro_magnitude.max()

    return result


def plot_raw_eeg_comparison(df, windows, output_path):
    """生EEGデータの比較プロット"""
    raw_cols = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

    fig, axes = plt.subplots(len(windows), 1, figsize=(14, 4 * len(windows)))
    if len(windows) == 1:
        axes = [axes]

    for ax, window in zip(axes, windows):
        df_window = extract_window(df, window['start_min'], window['end_min'])

        for col in raw_cols:
            if col in df_window.columns:
                ax.plot(df_window['elapsed_min'], df_window[col],
                       label=col.replace('RAW_', ''), alpha=0.7, linewidth=0.5)

        ax.set_title(f"{window['label']} ({window['start_min']:.1f}-{window['end_min']:.1f}分)")
        ax.set_xlabel('経過時間 (分)')
        ax.set_ylabel('電圧 (μV)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_band_power_comparison(df, windows, output_path):
    """バンドパワーの比較プロット"""
    bands = {
        'Delta': ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10'],
        'Theta': ['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10'],
        'Alpha': ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10'],
        'Beta': ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10'],
        'Gamma': ['Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10'],
    }

    fig, axes = plt.subplots(len(windows), 1, figsize=(14, 5 * len(windows)))
    if len(windows) == 1:
        axes = [axes]

    colors = {'Delta': 'purple', 'Theta': 'blue', 'Alpha': 'green',
              'Beta': 'orange', 'Gamma': 'red'}

    for ax, window in zip(axes, windows):
        df_window = extract_window(df, window['start_min'], window['end_min'])

        for band_name, band_cols in bands.items():
            if all(col in df_window.columns for col in band_cols):
                band_mean = df_window[band_cols].mean(axis=1)
                ax.plot(df_window['elapsed_min'], band_mean,
                       label=band_name, color=colors[band_name], alpha=0.8)

        ax.set_title(f"{window['label']} - バンドパワー時系列")
        ax.set_xlabel('経過時間 (分)')
        ax.set_ylabel('パワー (dB)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_spectrogram_comparison(df, windows, output_path):
    """スペクトログラムの比較プロット"""
    raw_cols = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
    sfreq = 256  # サンプリング周波数

    fig, axes = plt.subplots(len(windows), 2, figsize=(16, 4 * len(windows)))
    if len(windows) == 1:
        axes = axes.reshape(1, -1)

    for row, window in enumerate(windows):
        df_window = extract_window(df, window['start_min'], window['end_min'])

        # 前頭（AF7+AF8平均）と側頭（TP9+TP10平均）
        for col_idx, (region_name, cols) in enumerate([
            ('前頭 (AF7+AF8)', ['RAW_AF7', 'RAW_AF8']),
            ('側頭 (TP9+TP10)', ['RAW_TP9', 'RAW_TP10'])
        ]):
            ax = axes[row, col_idx]

            if all(col in df_window.columns for col in cols):
                # 平均信号
                signal_data = df_window[cols].mean(axis=1).values

                # NaN除去
                signal_data = np.nan_to_num(signal_data, nan=0.0)

                if len(signal_data) > sfreq:
                    # スペクトログラム計算
                    f, t, Sxx = signal.spectrogram(
                        signal_data,
                        fs=sfreq,
                        nperseg=sfreq,
                        noverlap=sfreq//2,
                        scaling='density'
                    )

                    # 0-40Hz表示
                    freq_mask = f <= 40

                    # dB変換
                    Sxx_db = 10 * np.log10(Sxx[freq_mask, :] + 1e-10)

                    im = ax.pcolormesh(t, f[freq_mask], Sxx_db,
                                      shading='gouraud', cmap='viridis',
                                      vmin=-10, vmax=30)
                    plt.colorbar(im, ax=ax, label='Power (dB)')

            ax.set_title(f"{window['label']} - {region_name}")
            ax.set_xlabel('時間 (秒)')
            ax.set_ylabel('周波数 (Hz)')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_motion_during_burst(df, windows, output_path):
    """バースト時のモーションデータ"""
    fig, axes = plt.subplots(len(windows), 2, figsize=(14, 4 * len(windows)))
    if len(windows) == 1:
        axes = axes.reshape(1, -1)

    for row, window in enumerate(windows):
        df_window = extract_window(df, window['start_min'], window['end_min'])

        # 加速度
        ax_accel = axes[row, 0]
        accel_cols = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
        if all(col in df_window.columns for col in accel_cols):
            for col in accel_cols:
                ax_accel.plot(df_window['elapsed_min'], df_window[col],
                             label=col.split('_')[1], alpha=0.7)
            ax_accel.set_title(f"{window['label']} - 加速度")
            ax_accel.set_xlabel('経過時間 (分)')
            ax_accel.set_ylabel('加速度 (m/s²)')
            ax_accel.legend()
            ax_accel.grid(True, alpha=0.3)

        # ジャイロ
        ax_gyro = axes[row, 1]
        gyro_cols = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']
        if all(col in df_window.columns for col in gyro_cols):
            for col in gyro_cols:
                ax_gyro.plot(df_window['elapsed_min'], df_window[col],
                            label=col.split('_')[1], alpha=0.7)
            ax_gyro.set_title(f"{window['label']} - ジャイロ（頭部回転）")
            ax_gyro.set_xlabel('経過時間 (分)')
            ax_gyro.set_ylabel('角速度 (°/s)')
            ax_gyro.legend()
            ax_gyro.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def compute_psd_for_window(df_window, sfreq=256):
    """指定区間のPSD計算"""
    raw_cols = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

    psds = {}
    for col in raw_cols:
        if col in df_window.columns:
            data = df_window[col].dropna().values
            if len(data) > sfreq * 2:  # 最低2秒必要
                f, psd = signal.welch(data, fs=sfreq, nperseg=sfreq*2)
                psds[col] = {'freqs': f, 'psd': psd}

    return psds


def plot_psd_comparison(df, burst_windows, normal_windows, output_path):
    """バースト時と通常時のPSD比較"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    regions = [
        ('前頭 (AF7+AF8)', ['RAW_AF7', 'RAW_AF8']),
        ('側頭 (TP9+TP10)', ['RAW_TP9', 'RAW_TP10'])
    ]

    for row, (region_name, cols) in enumerate(regions):
        # バースト区間
        ax_burst = axes[row, 0]
        for window in burst_windows:
            df_window = extract_window(df, window['start_min'], window['end_min'])
            psds = compute_psd_for_window(df_window)

            # 領域平均PSD
            psd_list = []
            freqs = None
            for col in cols:
                if col in psds:
                    if freqs is None:
                        freqs = psds[col]['freqs']
                    psd_list.append(psds[col]['psd'])

            if psd_list:
                psd_mean = np.mean(psd_list, axis=0)
                psd_db = 10 * np.log10(psd_mean + 1e-10)
                ax_burst.plot(freqs, psd_db, label=window['label'], alpha=0.8)

        ax_burst.set_xlim(0, 40)
        ax_burst.set_title(f'{region_name} - バースト区間')
        ax_burst.set_xlabel('周波数 (Hz)')
        ax_burst.set_ylabel('パワー (dB)')
        ax_burst.legend()
        ax_burst.grid(True, alpha=0.3)
        ax_burst.axvspan(4, 8, alpha=0.2, color='blue', label='Theta')
        ax_burst.axvspan(8, 12, alpha=0.2, color='green', label='Alpha')

        # 通常区間
        ax_normal = axes[row, 1]
        for window in normal_windows:
            df_window = extract_window(df, window['start_min'], window['end_min'])
            psds = compute_psd_for_window(df_window)

            psd_list = []
            freqs = None
            for col in cols:
                if col in psds:
                    if freqs is None:
                        freqs = psds[col]['freqs']
                    psd_list.append(psds[col]['psd'])

            if psd_list:
                psd_mean = np.mean(psd_list, axis=0)
                psd_db = 10 * np.log10(psd_mean + 1e-10)
                ax_normal.plot(freqs, psd_db, label=window['label'], alpha=0.8)

        ax_normal.set_xlim(0, 40)
        ax_normal.set_title(f'{region_name} - 通常区間')
        ax_normal.set_xlabel('周波数 (Hz)')
        ax_normal.set_ylabel('パワー (dB)')
        ax_normal.legend()
        ax_normal.grid(True, alpha=0.3)
        ax_normal.axvspan(4, 8, alpha=0.2, color='blue')
        ax_normal.axvspan(8, 12, alpha=0.2, color='green')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def analyze_burst_characteristics(df, window):
    """バーストの詳細特性分析"""
    df_window = extract_window(df, window['start_min'], window['end_min'])

    result = {
        'window': window['label'],
        'duration_sec': len(df_window) / 256,
        'samples': len(df_window),
    }

    # バンドパワー統計
    bands = {
        'Delta': ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10'],
        'Theta': ['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10'],
        'Alpha': ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10'],
        'Beta': ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10'],
        'Gamma': ['Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10'],
    }

    for band_name, band_cols in bands.items():
        if all(col in df_window.columns for col in band_cols):
            band_mean = df_window[band_cols].mean(axis=1)
            result[f'{band_name}_mean'] = band_mean.mean()
            result[f'{band_name}_max'] = band_mean.max()
            result[f'{band_name}_std'] = band_mean.std()

    # Raw EEG振幅
    raw_cols = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
    if all(col in df_window.columns for col in raw_cols):
        raw_mean = df_window[raw_cols].mean(axis=1)
        result['raw_amplitude_mean'] = np.abs(raw_mean).mean()
        result['raw_amplitude_max'] = np.abs(raw_mean).max()
        result['raw_amplitude_std'] = raw_mean.std()

    return result


def plot_burst_zoom(df, window, output_path):
    """バースト部分の拡大プロット（10秒単位）"""
    df_window = extract_window(df, window['start_min'], window['end_min'])

    raw_cols = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

    # 10秒ごとに分割
    total_sec = (window['end_min'] - window['start_min']) * 60
    n_segments = int(total_sec // 10)

    fig, axes = plt.subplots(n_segments, 1, figsize=(14, 3 * n_segments))
    if n_segments == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        start_sec = i * 10
        end_sec = (i + 1) * 10

        # 秒単位でのフィルタ
        window_start_sec = window['start_min'] * 60
        mask = (
            ((df_window['elapsed_min'] * 60) >= (window_start_sec + start_sec)) &
            ((df_window['elapsed_min'] * 60) < (window_start_sec + end_sec))
        )
        df_seg = df_window[mask]

        if len(df_seg) > 0:
            time_sec = (df_seg['elapsed_min'] * 60) - window_start_sec

            for col in raw_cols:
                if col in df_seg.columns:
                    ax.plot(time_sec, df_seg[col],
                           label=col.replace('RAW_', ''), alpha=0.7, linewidth=0.5)

        ax.set_title(f"{start_sec}-{end_sec}秒")
        ax.set_xlabel('時間 (秒)')
        ax.set_ylabel('電圧 (μV)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(start_sec, end_sec)

    plt.suptitle(f"{window['label']} - 10秒ごとの拡大", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    """メイン処理"""
    print("=" * 60)
    print("バースト区間詳細分析")
    print("=" * 60)

    # データ読み込み
    df = load_data()
    print(f"データ形状: {df.shape}")
    print(f"時間範囲: {df['elapsed_min'].min():.1f} - {df['elapsed_min'].max():.1f} 分")

    all_windows = BURST_WINDOWS + NORMAL_WINDOWS

    # 1. モーション・HSI分析
    print("\n1. モーション・HSI分析...")
    motion_results = []
    for window in all_windows:
        df_window = extract_window(df, window['start_min'], window['end_min'])
        result = analyze_motion_hsi(df_window, window)
        motion_results.append(result)

    motion_df = pd.DataFrame(motion_results)
    print(motion_df.to_string())
    motion_df.to_csv(OUTPUT_DIR / "motion_hsi_comparison.csv", index=False)

    # 2. 生EEGプロット
    print("\n2. 生EEGプロット...")
    plot_raw_eeg_comparison(df, BURST_WINDOWS, OUTPUT_DIR / "raw_eeg_burst.png")
    plot_raw_eeg_comparison(df, NORMAL_WINDOWS, OUTPUT_DIR / "raw_eeg_normal.png")

    # 3. バンドパワー比較
    print("\n3. バンドパワー比較...")
    plot_band_power_comparison(df, all_windows, OUTPUT_DIR / "band_power_comparison.png")

    # 4. スペクトログラム
    print("\n4. スペクトログラム比較...")
    plot_spectrogram_comparison(df, BURST_WINDOWS, OUTPUT_DIR / "spectrogram_burst.png")
    plot_spectrogram_comparison(df, NORMAL_WINDOWS, OUTPUT_DIR / "spectrogram_normal.png")

    # 5. モーションプロット
    print("\n5. モーションプロット...")
    plot_motion_during_burst(df, BURST_WINDOWS, OUTPUT_DIR / "motion_burst.png")

    # 6. PSD比較
    print("\n6. PSD比較...")
    plot_psd_comparison(df, BURST_WINDOWS, NORMAL_WINDOWS, OUTPUT_DIR / "psd_comparison.png")

    # 7. バースト特性分析
    print("\n7. バースト特性分析...")
    burst_results = []
    for window in all_windows:
        result = analyze_burst_characteristics(df, window)
        burst_results.append(result)

    burst_df = pd.DataFrame(burst_results)
    print(burst_df.to_string())
    burst_df.to_csv(OUTPUT_DIR / "burst_characteristics.csv", index=False)

    # 8. バースト拡大プロット
    print("\n8. バースト拡大プロット...")
    for window in BURST_WINDOWS:
        safe_name = window['name']
        plot_burst_zoom(df, window, OUTPUT_DIR / f"zoom_{safe_name}.png")

    print("\n" + "=" * 60)
    print("分析完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
