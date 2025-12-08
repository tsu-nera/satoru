#!/usr/bin/env python3
"""
EOG（眼電図）アーティファクトの影響度デモンストレーション

閉眼中の眼球運動がEEGに与える影響を定量化
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import load_mind_monitor_csv

DATA_PATH = project_root / "data" / "mindMonitor_2025-12-08--07-38-33_3000223989795832513.csv"
OUTPUT_DIR = Path(__file__).parent


def load_data():
    df = load_mind_monitor_csv(DATA_PATH, filter_headband=False, warmup_seconds=60)
    start_time = df['TimeStamp'].iloc[0]
    df['elapsed_min'] = (df['TimeStamp'] - start_time).dt.total_seconds() / 60
    return df


def analyze_eog_magnitude():
    """EOGアーティファクトの振幅を分析"""
    df = load_data()

    # バースト区間と通常区間
    burst_15 = df[(df['elapsed_min'] >= 14.0) & (df['elapsed_min'] <= 15.0)]
    burst_27 = df[(df['elapsed_min'] >= 27.0) & (df['elapsed_min'] <= 28.0)]
    normal = df[(df['elapsed_min'] >= 20.0) & (df['elapsed_min'] <= 21.0)]

    results = []

    for name, data in [('15分バースト', burst_15), ('27分バースト', burst_27), ('通常区間', normal)]:
        # 前頭部（EOGの影響大）
        af7 = data['RAW_AF7'].dropna()
        af8 = data['RAW_AF8'].dropna()
        frontal_amplitude = (af7.std() + af8.std()) / 2

        # 側頭部（EOGの影響小）
        tp9 = data['RAW_TP9'].dropna()
        tp10 = data['RAW_TP10'].dropna()
        temporal_amplitude = (tp9.std() + tp10.std()) / 2

        # HEOG (水平眼球運動)
        heog = (data['RAW_AF7'] - data['RAW_AF8']).dropna()
        heog_amplitude = heog.std()

        # 最大振幅
        frontal_max = max(af7.max() - af7.min(), af8.max() - af8.min())
        heog_max = heog.max() - heog.min()

        results.append({
            '区間': name,
            '前頭部振幅 (μV)': frontal_amplitude,
            '側頭部振幅 (μV)': temporal_amplitude,
            '前頭/側頭比': frontal_amplitude / temporal_amplitude,
            'HEOG振幅 (μV)': heog_amplitude,
            '前頭部最大変動 (μV)': frontal_max,
            'HEOG最大変動 (μV)': heog_max,
        })

    return pd.DataFrame(results)


def plot_eog_effect():
    """EOGの影響を可視化"""
    df = load_data()

    # 15分バーストの30秒間
    data = df[(df['elapsed_min'] >= 14.5) & (df['elapsed_min'] <= 15.0)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    time = (data['elapsed_min'] - 14.5) * 60  # 秒に変換

    # 1. HEOG (AF7 - AF8): 水平眼球運動
    ax1 = axes[0]
    heog = data['RAW_AF7'] - data['RAW_AF8']
    ax1.plot(time, heog, color='red', linewidth=0.5)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('HEOG (μV)\n(AF7-AF8)')
    ax1.set_title('水平眼球運動 (HEOG): 左右に目を動かすと大きく振れる')
    ax1.set_ylim(-800, 800)
    ax1.grid(True, alpha=0.3)

    # 2. 前頭部平均
    ax2 = axes[1]
    frontal = (data['RAW_AF7'] + data['RAW_AF8']) / 2
    frontal_centered = frontal - frontal.mean()
    ax2.plot(time, frontal_centered, color='orange', linewidth=0.5)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('前頭部 (μV)\n(AF7+AF8)/2')
    ax2.set_title('前頭部EEG: 眼球運動の影響を強く受ける')
    ax2.set_ylim(-400, 400)
    ax2.grid(True, alpha=0.3)

    # 3. 側頭部平均
    ax3 = axes[2]
    temporal = (data['RAW_TP9'] + data['RAW_TP10']) / 2
    temporal_centered = temporal - temporal.mean()
    ax3.plot(time, temporal_centered, color='blue', linewidth=0.5)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('側頭部 (μV)\n(TP9+TP10)/2')
    ax3.set_title('側頭部EEG: 眼球運動の影響は小さい（が、ゼロではない）')
    ax3.set_ylim(-400, 400)
    ax3.set_xlabel('時間 (秒)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'eog_effect_demonstration.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'eog_effect_demonstration.png'}")


def main():
    print("=" * 60)
    print("EOGアーティファクトの影響度分析")
    print("=" * 60)

    # 振幅分析
    print("\n【EOGアーティファクトの振幅】\n")
    results = analyze_eog_magnitude()
    print(results.to_string(index=False))

    # 可視化
    print("\n【可視化】")
    plot_eog_effect()

    # 解説
    print("\n" + "=" * 60)
    print("【結論】")
    print("=" * 60)
    print("""
閉眼中でも眼球を動かすだけで:
- 前頭部EEG: 数百μVの変動が発生
- HEOG: 最大800μV以上の変動
- 側頭部EEG: 影響は小さいが完全にはゼロではない

これは通常のEEG信号（数十μV）の10倍以上の大きさ。
→ 眼球運動は最も強力なEEGアーティファクト源の一つ。

【瞑想中の対策】
- 眼球を意識的に固定する（例：眉間に集中）
- 目を軽く閉じる（強く閉じると筋電図アーティファクト）
- EOG除去アルゴリズム（ICA等）を事後処理で適用
""")


if __name__ == "__main__":
    main()
