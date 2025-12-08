#!/usr/bin/env python3
"""
EOG（眼球運動）アーティファクト詳細分析

眼球運動アーティファクトの特徴:
- 前頭部（AF7/AF8）で最大振幅
- 側頭部（TP9/TP10）では減衰
- 低周波成分が支配的だが、広帯域に影響
- 瞬き: 急峻な負のスパイク後に正の回復
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import load_mind_monitor_csv

# データパス
DATA_PATH = project_root / "data" / "mindMonitor_2025-12-08--07-38-33_3000223989795832513.csv"
OUTPUT_DIR = Path(__file__).parent

# バースト区間
BURST_WINDOWS = [
    {"name": "burst_15min", "start_min": 14.0, "end_min": 15.0, "label": "15分バースト（ピーク1分間）"},
    {"name": "burst_27min", "start_min": 27.0, "end_min": 28.0, "label": "27分バースト（ピーク1分間）"},
]

NORMAL_WINDOW = {"name": "normal", "start_min": 20.0, "end_min": 21.0, "label": "通常区間（20-21分）"}


def load_data():
    """データ読み込み"""
    print(f"Loading: {DATA_PATH}")
    df = load_mind_monitor_csv(DATA_PATH, filter_headband=False, warmup_seconds=60)
    start_time = df['TimeStamp'].iloc[0]
    df['elapsed_min'] = (df['TimeStamp'] - start_time).dt.total_seconds() / 60
    return df


def extract_window(df, start_min, end_min):
    """指定区間のデータを抽出"""
    mask = (df['elapsed_min'] >= start_min) & (df['elapsed_min'] <= end_min)
    return df[mask].copy()


def analyze_channel_amplitude_ratio(df_window):
    """
    チャネル間の振幅比を分析
    EOGの場合: AF7/AF8 >> TP9/TP10
    """
    raw_cols = {
        'frontal': ['RAW_AF7', 'RAW_AF8'],
        'temporal': ['RAW_TP9', 'RAW_TP10']
    }

    result = {}

    # 各領域の振幅（標準偏差）
    for region, cols in raw_cols.items():
        if all(col in df_window.columns for col in cols):
            amplitudes = []
            for col in cols:
                data = df_window[col].dropna()
                # DCオフセット除去
                data_centered = data - data.mean()
                amplitudes.append(data_centered.std())
            result[f'{region}_amplitude'] = np.mean(amplitudes)

    # 前頭/側頭比
    if 'frontal_amplitude' in result and 'temporal_amplitude' in result:
        result['frontal_temporal_ratio'] = result['frontal_amplitude'] / result['temporal_amplitude']

    return result


def analyze_channel_correlation(df_window):
    """
    チャネル間相関分析
    EOGの場合:
    - AF7とAF8は高相関（同側に動く）
    - または逆相関（左右に動く）
    """
    raw_cols = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

    if not all(col in df_window.columns for col in raw_cols):
        return {}

    # 相関行列
    corr_matrix = df_window[raw_cols].corr()

    result = {
        'AF7_AF8_corr': corr_matrix.loc['RAW_AF7', 'RAW_AF8'],
        'TP9_TP10_corr': corr_matrix.loc['RAW_TP9', 'RAW_TP10'],
        'AF7_TP9_corr': corr_matrix.loc['RAW_AF7', 'RAW_TP9'],
        'AF8_TP10_corr': corr_matrix.loc['RAW_AF8', 'RAW_TP10'],
    }

    return result


def detect_blink_events(df_window, threshold_std=3.0):
    """
    瞬きイベントの検出
    瞬きの特徴: 急峻な負のスパイク → 正の回復
    """
    frontal_cols = ['RAW_AF7', 'RAW_AF8']

    if not all(col in df_window.columns for col in frontal_cols):
        return []

    # 前頭部平均
    frontal_mean = df_window[frontal_cols].mean(axis=1)
    frontal_centered = frontal_mean - frontal_mean.mean()

    # 閾値: 標準偏差のN倍
    threshold = threshold_std * frontal_centered.std()

    # 大きな変動を検出
    events = []
    above_threshold = np.abs(frontal_centered) > threshold

    # イベント数をカウント
    event_count = 0
    in_event = False
    for i, is_above in enumerate(above_threshold):
        if is_above and not in_event:
            in_event = True
            event_count += 1
        elif not is_above:
            in_event = False

    return event_count


def plot_frontal_vs_temporal(df, windows, output_path):
    """前頭部 vs 側頭部の振幅比較"""
    fig, axes = plt.subplots(len(windows), 2, figsize=(16, 5 * len(windows)))
    if len(windows) == 1:
        axes = axes.reshape(1, -1)

    for row, window in enumerate(windows):
        df_window = extract_window(df, window['start_min'], window['end_min'])

        # 時間軸（秒）
        time_sec = (df_window['elapsed_min'] - window['start_min']) * 60

        # 前頭部
        ax_frontal = axes[row, 0]
        for col, color in [('RAW_AF7', 'orange'), ('RAW_AF8', 'green')]:
            if col in df_window.columns:
                data = df_window[col] - df_window[col].mean()  # DCオフセット除去
                ax_frontal.plot(time_sec, data, label=col.replace('RAW_', ''),
                               color=color, alpha=0.7, linewidth=0.5)
        ax_frontal.set_title(f"{window['label']} - 前頭部（AF7/AF8）")
        ax_frontal.set_xlabel('時間 (秒)')
        ax_frontal.set_ylabel('電圧 (μV, DCオフセット除去)')
        ax_frontal.legend()
        ax_frontal.grid(True, alpha=0.3)
        ax_frontal.set_ylim(-500, 500)

        # 側頭部
        ax_temporal = axes[row, 1]
        for col, color in [('RAW_TP9', 'blue'), ('RAW_TP10', 'purple')]:
            if col in df_window.columns:
                data = df_window[col] - df_window[col].mean()
                ax_temporal.plot(time_sec, data, label=col.replace('RAW_', ''),
                                color=color, alpha=0.7, linewidth=0.5)
        ax_temporal.set_title(f"{window['label']} - 側頭部（TP9/TP10）")
        ax_temporal.set_xlabel('時間 (秒)')
        ax_temporal.set_ylabel('電圧 (μV, DCオフセット除去)')
        ax_temporal.legend()
        ax_temporal.grid(True, alpha=0.3)
        ax_temporal.set_ylim(-500, 500)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_channel_difference(df, windows, output_path):
    """
    チャネル差分プロット
    AF7-AF8: 水平眼球運動の検出
    (AF7+AF8)/2 - (TP9+TP10)/2: EOGとEEGの分離
    """
    fig, axes = plt.subplots(len(windows), 2, figsize=(16, 5 * len(windows)))
    if len(windows) == 1:
        axes = axes.reshape(1, -1)

    for row, window in enumerate(windows):
        df_window = extract_window(df, window['start_min'], window['end_min'])
        time_sec = (df_window['elapsed_min'] - window['start_min']) * 60

        # AF7-AF8（水平眼球運動）
        ax_heog = axes[row, 0]
        if 'RAW_AF7' in df_window.columns and 'RAW_AF8' in df_window.columns:
            heog = df_window['RAW_AF7'] - df_window['RAW_AF8']
            ax_heog.plot(time_sec, heog, color='red', alpha=0.7, linewidth=0.5)
            ax_heog.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_heog.set_title(f"{window['label']} - HEOG (AF7-AF8)")
        ax_heog.set_xlabel('時間 (秒)')
        ax_heog.set_ylabel('電圧差 (μV)')
        ax_heog.grid(True, alpha=0.3)

        # 前頭-側頭差分（VEOG proxy）
        ax_veog = axes[row, 1]
        frontal_cols = ['RAW_AF7', 'RAW_AF8']
        temporal_cols = ['RAW_TP9', 'RAW_TP10']
        if all(col in df_window.columns for col in frontal_cols + temporal_cols):
            frontal_mean = df_window[frontal_cols].mean(axis=1)
            temporal_mean = df_window[temporal_cols].mean(axis=1)
            veog_proxy = frontal_mean - temporal_mean
            ax_veog.plot(time_sec, veog_proxy, color='blue', alpha=0.7, linewidth=0.5)
            ax_veog.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_veog.set_title(f"{window['label']} - 前頭-側頭差分")
        ax_veog.set_xlabel('時間 (秒)')
        ax_veog.set_ylabel('電圧差 (μV)')
        ax_veog.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_burst_waveform_detail(df, window, output_path):
    """
    バースト波形の詳細（5秒単位）
    瞬きパターンの確認用
    """
    df_window = extract_window(df, window['start_min'], window['end_min'])

    # 5秒ごとに分割
    total_sec = (window['end_min'] - window['start_min']) * 60
    n_segments = int(total_sec // 5)

    fig, axes = plt.subplots(n_segments, 1, figsize=(14, 2.5 * n_segments))
    if n_segments == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        start_sec = i * 5
        end_sec = (i + 1) * 5

        window_start_sec = window['start_min'] * 60
        mask = (
            ((df_window['elapsed_min'] * 60) >= (window_start_sec + start_sec)) &
            ((df_window['elapsed_min'] * 60) < (window_start_sec + end_sec))
        )
        df_seg = df_window[mask]

        if len(df_seg) > 0:
            time_sec = (df_seg['elapsed_min'] * 60) - window_start_sec

            # 前頭部平均（EOG成分が強い）
            if 'RAW_AF7' in df_seg.columns and 'RAW_AF8' in df_seg.columns:
                frontal = (df_seg['RAW_AF7'] + df_seg['RAW_AF8']) / 2
                frontal_centered = frontal - frontal.mean()
                ax.plot(time_sec, frontal_centered, label='前頭 (AF7+AF8)/2',
                       color='red', alpha=0.8, linewidth=0.8)

            # 側頭部平均（EEG成分が強い）
            if 'RAW_TP9' in df_seg.columns and 'RAW_TP10' in df_seg.columns:
                temporal = (df_seg['RAW_TP9'] + df_seg['RAW_TP10']) / 2
                temporal_centered = temporal - temporal.mean()
                ax.plot(time_sec, temporal_centered, label='側頭 (TP9+TP10)/2',
                       color='blue', alpha=0.8, linewidth=0.8)

        ax.set_title(f"{start_sec}-{end_sec}秒")
        ax.set_xlabel('時間 (秒)')
        ax.set_ylabel('電圧 (μV)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(start_sec, end_sec)
        ax.set_ylim(-400, 400)

    plt.suptitle(f"{window['label']} - 前頭部 vs 側頭部（5秒詳細）", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def create_eog_summary_table(df, windows):
    """EOG分析サマリーテーブル"""
    results = []

    for window in windows:
        df_window = extract_window(df, window['start_min'], window['end_min'])

        row = {'区間': window['label']}

        # 振幅比分析
        amp_result = analyze_channel_amplitude_ratio(df_window)
        row.update(amp_result)

        # 相関分析
        corr_result = analyze_channel_correlation(df_window)
        row.update(corr_result)

        # 瞬きイベント数
        blink_count = detect_blink_events(df_window)
        row['blink_events'] = blink_count

        results.append(row)

    return pd.DataFrame(results)


def main():
    """メイン処理"""
    print("=" * 60)
    print("EOGアーティファクト詳細分析")
    print("=" * 60)

    df = load_data()

    all_windows = BURST_WINDOWS + [NORMAL_WINDOW]

    # 1. EOGサマリー
    print("\n1. EOG特性サマリー...")
    eog_summary = create_eog_summary_table(df, all_windows)
    print(eog_summary.to_string())
    eog_summary.to_csv(OUTPUT_DIR / "eog_summary.csv", index=False)

    # 2. 前頭部 vs 側頭部プロット
    print("\n2. 前頭部 vs 側頭部比較...")
    plot_frontal_vs_temporal(df, all_windows, OUTPUT_DIR / "frontal_vs_temporal.png")

    # 3. チャネル差分プロット
    print("\n3. チャネル差分（HEOG/VEOG proxy）...")
    plot_channel_difference(df, all_windows, OUTPUT_DIR / "channel_difference.png")

    # 4. バースト波形詳細
    print("\n4. バースト波形詳細...")
    for window in BURST_WINDOWS:
        plot_burst_waveform_detail(df, window,
                                   OUTPUT_DIR / f"waveform_detail_{window['name']}.png")

    print("\n" + "=" * 60)
    print("EOG分析完了!")
    print("=" * 60)

    # 結果の解釈
    print("\n【EOGアーティファクトの判定基準】")
    print("- 前頭/側頭振幅比 > 1.5: EOGの可能性高い")
    print("- AF7-AF8相関が高い: 垂直眼球運動/瞬き")
    print("- AF7-AF8相関が負: 水平眼球運動")
    print("")

    for _, row in eog_summary.iterrows():
        ratio = row.get('frontal_temporal_ratio', 0)
        print(f"\n{row['区間']}:")
        print(f"  前頭/側頭比: {ratio:.2f}")
        if ratio > 1.5:
            print("  → EOGアーティファクトの可能性が高い")
        else:
            print("  → 脳由来の活動の可能性")


if __name__ == "__main__":
    main()
