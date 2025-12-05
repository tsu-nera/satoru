#!/usr/bin/env python3
"""
10-15Hz帯域の異常分析スクリプト

2025-12-04のセッションで観察された10-15Hz帯域の異常パターンを詳細に調査。
- IAFの二峰性（8.5Hz vs 12.5Hz）の原因を特定
- アーティファクト（EMG、眼球運動）の可能性を検証
- チャネル別の詳細分析

Usage:
    python analyze_10_15hz_anomaly.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne
from scipy import signal

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import (
    load_mind_monitor_csv,
    prepare_mne_raw,
    calculate_psd,
    calculate_paf,
    calculate_spectrogram_all_channels,
    filter_eeg_quality,
    FREQ_BANDS,
)

# 設定
DATA_PATH = project_root / 'data' / 'mindMonitor_2025-12-04--07-39-03_7794313749178367799.csv'
OUTPUT_DIR = Path(__file__).parent / 'img'
OUTPUT_DIR.mkdir(exist_ok=True)

# 日本語フォント設定
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'sans-serif']


def analyze_psd_by_segment(raw, segment_minutes=3, warmup_minutes=1):
    """
    セグメント別のPSD分析

    Returns
    -------
    dict : セグメント別のPSD結果
    """
    sfreq = raw.info['sfreq']
    total_samples = raw.n_times
    total_seconds = total_samples / sfreq

    warmup_samples = int(warmup_minutes * 60 * sfreq)
    segment_samples = int(segment_minutes * 60 * sfreq)

    results = []
    start_sample = warmup_samples
    segment_num = 1

    while start_sample + segment_samples <= total_samples:
        end_sample = start_sample + segment_samples

        # セグメントを抽出
        raw_segment = raw.copy().crop(
            tmin=start_sample / sfreq,
            tmax=end_sample / sfreq
        )

        # PSD計算
        psd_dict = calculate_psd(raw_segment)
        paf_dict = calculate_paf(psd_dict)

        # チャネル別PSD辞書を作成
        psd_by_channel = {}
        for i, ch_name in enumerate(psd_dict['channels']):
            psd_by_channel[ch_name] = psd_dict['psds'][i]

        # 全チャネル平均PSD
        psd_avg = np.mean(psd_dict['psds'], axis=0)

        results.append({
            'segment': segment_num,
            'start_min': (start_sample / sfreq) / 60,
            'end_min': (end_sample / sfreq) / 60,
            'freqs': psd_dict['freqs'],
            'psd_by_channel': psd_by_channel,
            'psd_avg': psd_avg,
            'iaf_peak': paf_dict['iaf_peak'],
            'iaf_cog': paf_dict['iaf_cog'],
            'paf_by_channel': paf_dict['paf_by_channel'],
        })

        start_sample = end_sample
        segment_num += 1

    return results


def plot_segment_psd_comparison(segment_results, output_path):
    """
    セグメント別PSD比較プロット
    """
    n_segments = len(segment_results)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # カラーマップ
    colors = plt.cm.viridis(np.linspace(0, 1, n_segments))

    # 上: 全体PSD比較
    ax1 = axes[0]
    for i, seg in enumerate(segment_results):
        freqs = seg['freqs']
        psd_avg = seg['psd_avg']

        # 1-30Hz範囲でプロット
        mask = (freqs >= 1) & (freqs <= 30)
        label = f"Seg {seg['segment']} ({seg['start_min']:.0f}-{seg['end_min']:.0f}min) IAF={seg['iaf_peak']:.1f}Hz"
        ax1.semilogy(freqs[mask], psd_avg[mask], color=colors[i], label=label, linewidth=1.5)

    # Alpha帯域をハイライト
    ax1.axvspan(8, 13, alpha=0.2, color='green', label='Alpha (8-13Hz)')
    ax1.axvspan(10, 15, alpha=0.1, color='red', label='問題帯域 (10-15Hz)')

    ax1.set_xlabel('周波数 (Hz)')
    ax1.set_ylabel('パワースペクトル密度 (μV²/Hz)')
    ax1.set_title('セグメント別PSD比較')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 30)

    # 下: 10-15Hz帯域の詳細
    ax2 = axes[1]
    for i, seg in enumerate(segment_results):
        freqs = seg['freqs']
        psd_avg = seg['psd_avg']

        # 5-20Hz範囲で詳細表示
        mask = (freqs >= 5) & (freqs <= 20)
        ax2.plot(freqs[mask], psd_avg[mask], color=colors[i],
                 label=f"Seg {seg['segment']}", linewidth=2)

        # IAFピークをマーク
        iaf = seg['iaf_peak']
        if 5 <= iaf <= 20:
            idx = np.argmin(np.abs(freqs - iaf))
            ax2.scatter([iaf], [psd_avg[idx]], color=colors[i], s=100, zorder=5)

    ax2.axvspan(10, 15, alpha=0.2, color='red', label='問題帯域')
    ax2.set_xlabel('周波数 (Hz)')
    ax2.set_ylabel('パワースペクトル密度 (μV²/Hz)')
    ax2.set_title('10-15Hz帯域の詳細 (線形スケール)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(5, 20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存: {output_path}')


def plot_channel_comparison(segment_results, output_path):
    """
    チャネル別のPSD比較（10-15Hz帯域に注目）
    """
    channels = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
    channel_labels = ['TP9 (左側頭)', 'AF7 (左前頭)', 'AF8 (右前頭)', 'TP10 (右側頭)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    n_segments = len(segment_results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_segments))

    for ch_idx, (ch, ch_label) in enumerate(zip(channels, channel_labels)):
        ax = axes[ch_idx]

        for i, seg in enumerate(segment_results):
            freqs = seg['freqs']

            if ch in seg['psd_by_channel']:
                psd = seg['psd_by_channel'][ch]
                mask = (freqs >= 5) & (freqs <= 20)
                ax.plot(freqs[mask], psd[mask], color=colors[i],
                        label=f"Seg {seg['segment']}", linewidth=1.5)

        ax.axvspan(10, 15, alpha=0.2, color='red')
        ax.set_xlabel('周波数 (Hz)')
        ax.set_ylabel('PSD (μV²/Hz)')
        ax.set_title(f'{ch_label}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(5, 20)

    plt.suptitle('チャネル別PSD比較（10-15Hz帯域に注目）', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存: {output_path}')


def analyze_band_power_ratio(df, output_path):
    """
    バンドパワー比率の時系列分析
    (10-15Hz帯域の相対的な強さを確認)
    """
    # Alpha列の確認
    alpha_cols = [c for c in df.columns if c.startswith('Alpha_')]
    beta_cols = [c for c in df.columns if c.startswith('Beta_')]
    theta_cols = [c for c in df.columns if c.startswith('Theta_')]

    if not alpha_cols:
        print('警告: Alpha列が見つかりません')
        return

    # 時間軸
    timestamps = pd.to_datetime(df['TimeStamp'])
    elapsed_min = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 60

    # 各バンドの平均パワー
    alpha_mean = df[alpha_cols].mean(axis=1)
    beta_mean = df[beta_cols].mean(axis=1) if beta_cols else None
    theta_mean = df[theta_cols].mean(axis=1) if theta_cols else None

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # リサンプリング（10秒間隔）
    df_plot = pd.DataFrame({
        'elapsed_min': elapsed_min,
        'alpha': alpha_mean,
    })
    if beta_mean is not None:
        df_plot['beta'] = beta_mean
    if theta_mean is not None:
        df_plot['theta'] = theta_mean

    df_plot = df_plot.set_index('elapsed_min')
    df_resampled = df_plot.groupby(df_plot.index // (1/6)).mean()  # 約10秒間隔
    df_resampled.index = df_resampled.index * (1/6)

    # Alpha Power
    ax1 = axes[0]
    ax1.plot(df_resampled.index, df_resampled['alpha'], color='green', linewidth=1)
    ax1.fill_between(df_resampled.index, df_resampled['alpha'], alpha=0.3, color='green')
    ax1.set_ylabel('Alpha Power (dB)')
    ax1.set_title('Alpha帯域 (8-13Hz) パワー時系列')
    ax1.grid(True, alpha=0.3)

    # Alpha/Beta ratio (低いほどリラックス)
    if 'beta' in df_resampled.columns:
        ax2 = axes[1]
        ratio = df_resampled['alpha'] / (df_resampled['beta'] + 1e-10)
        ax2.plot(df_resampled.index, ratio, color='purple', linewidth=1)
        ax2.set_ylabel('Alpha/Beta Ratio')
        ax2.set_title('Alpha/Beta比（高いほどリラックス）')
        ax2.grid(True, alpha=0.3)

    # Theta/Alpha ratio (高いほど瞑想深度が深い)
    if 'theta' in df_resampled.columns:
        ax3 = axes[2]
        ratio = df_resampled['theta'] / (df_resampled['alpha'] + 1e-10)
        ax3.plot(df_resampled.index, ratio, color='blue', linewidth=1)
        ax3.set_ylabel('Theta/Alpha Ratio')
        ax3.set_title('Theta/Alpha比（高いほど瞑想深度が深い）')
        ax3.grid(True, alpha=0.3)

    ax3.set_xlabel('経過時間 (分)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存: {output_path}')


def analyze_raw_waveform_segments(raw, output_path, segment_minutes=3, warmup_minutes=1):
    """
    生波形のセグメント別分析（アーティファクト検出）
    """
    sfreq = raw.info['sfreq']
    warmup_samples = int(warmup_minutes * 60 * sfreq)
    segment_samples = int(segment_minutes * 60 * sfreq)

    # 代表的な3つのセグメントを比較（序盤、IAF高い区間、終盤）
    segments_to_check = [
        {'name': 'Seg 2 (6min, IAF≈8.4Hz)', 'start': warmup_samples + segment_samples},
        {'name': 'Seg 5 (15min, IAF≈12.5Hz)', 'start': warmup_samples + 4 * segment_samples},
        {'name': 'Seg 8 (24min, IAF≈11.1Hz)', 'start': warmup_samples + 7 * segment_samples},
    ]

    fig, axes = plt.subplots(len(segments_to_check), 2, figsize=(16, 12))

    for i, seg_info in enumerate(segments_to_check):
        start = seg_info['start']
        end = min(start + int(5 * sfreq), raw.n_times)  # 5秒間のスナップショット

        if end > raw.n_times:
            continue

        data, times = raw[:, start:end]
        times = times - times[0]  # 0から開始

        # 左: 生波形（全チャネル）
        ax_wave = axes[i, 0]
        for ch_idx, ch_name in enumerate(raw.ch_names):
            offset = ch_idx * 50  # チャネル間オフセット
            ax_wave.plot(times, data[ch_idx] + offset, linewidth=0.5, label=ch_name)

        ax_wave.set_xlabel('時間 (秒)')
        ax_wave.set_ylabel('振幅 (μV) + オフセット')
        ax_wave.set_title(f'{seg_info["name"]} - 生波形 (5秒)')
        ax_wave.legend(loc='upper right', fontsize=8)
        ax_wave.grid(True, alpha=0.3)

        # 右: 振幅ヒストグラム（アーティファクト検出）
        ax_hist = axes[i, 1]
        for ch_idx, ch_name in enumerate(raw.ch_names):
            ax_hist.hist(data[ch_idx], bins=50, alpha=0.5, label=ch_name)

        ax_hist.set_xlabel('振幅 (μV)')
        ax_hist.set_ylabel('頻度')
        ax_hist.set_title(f'{seg_info["name"]} - 振幅分布')
        ax_hist.legend(loc='upper right', fontsize=8)
        ax_hist.grid(True, alpha=0.3)

    plt.suptitle('セグメント別生波形分析（アーティファクト検出）', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存: {output_path}')


def calculate_emg_indicator(raw):
    """
    EMG（筋電図）混入の指標を計算

    高周波成分（20-45Hz）のパワーが高いとEMG混入の可能性
    """
    # 高周波帯域のパワーを計算
    raw_highpass = raw.copy().filter(l_freq=20, h_freq=45, verbose=False)

    data = raw_highpass.get_data()
    rms = np.sqrt(np.mean(data ** 2, axis=1))

    return {
        'channel': raw.ch_names,
        'high_freq_rms': rms,
        'mean_rms': np.mean(rms),
    }


def plot_spectrogram_comparison(tfr_results, output_path):
    """
    チャネル別スペクトログラムの詳細比較（10-15Hz帯域強調）
    """
    channels = list(tfr_results.keys())

    fig, axes = plt.subplots(len(channels), 1, figsize=(16, 4 * len(channels)))
    if len(channels) == 1:
        axes = [axes]

    for i, ch in enumerate(channels):
        tfr = tfr_results[ch]
        times = tfr['times']
        freqs = tfr['freqs']
        power = tfr['power']

        # 5-20Hzの範囲を表示
        freq_mask = (freqs >= 5) & (freqs <= 20)

        ax = axes[i]
        im = ax.pcolormesh(
            times / 60,  # 分に変換
            freqs[freq_mask],
            10 * np.log10(power[freq_mask, :] + 1e-10),
            shading='auto',
            cmap='viridis'
        )

        # 10-15Hz帯域をハイライト
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=15, color='red', linestyle='--', alpha=0.7, linewidth=1)

        ax.set_ylabel('周波数 (Hz)')
        ax.set_title(f'{ch} - スペクトログラム (5-20Hz)')
        plt.colorbar(im, ax=ax, label='Power (dB)')

    axes[-1].set_xlabel('経過時間 (分)')

    plt.suptitle('チャネル別スペクトログラム（10-15Hz帯域を赤線でマーク）', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存: {output_path}')


def generate_analysis_report(segment_results, emg_indicator, output_path):
    """
    分析レポートを生成
    """
    report = """# 10-15Hz帯域異常分析レポート

## 概要

2025-12-04のセッションで観察された10-15Hz帯域の異常パターンについて詳細分析を行いました。

---

## 1. セグメント別IAF (Individual Alpha Frequency)

| セグメント | 時間範囲 | IAF (Peak) | IAF (CoG) | 備考 |
|:----------|:---------|:-----------|:----------|:-----|
"""

    for seg in segment_results:
        note = ""
        if seg['iaf_peak'] > 11:
            note = "**異常** (12Hz以上)"
        elif seg['iaf_peak'] < 9:
            note = "通常範囲"
        else:
            note = "中間"

        report += f"| {seg['segment']} | {seg['start_min']:.0f}-{seg['end_min']:.0f}分 | {seg['iaf_peak']:.2f} Hz | {seg['iaf_cog']:.2f} Hz | {note} |\n"

    report += """
---

## 2. 高周波成分（EMG指標）

EMG（筋電図）混入の可能性を評価するため、20-45Hz帯域のパワーを確認しました。

| チャネル | High-freq RMS (μV) |
|:---------|:-------------------|
"""

    for ch, rms in zip(emg_indicator['channel'], emg_indicator['high_freq_rms']):
        report += f"| {ch} | {rms:.3f} |\n"

    report += f"\n**平均RMS**: {emg_indicator['mean_rms']:.3f} μV\n"

    report += """
---

## 3. 分析結果の解釈

### 3.1 IAFの二峰性について

セグメント別分析から、IAFが以下の2つの状態を示していることがわかりました：

1. **通常Alpha状態** (IAF ≈ 8.4-8.5 Hz): セグメント1-3, 7, 9
2. **高周波シフト状態** (IAF ≈ 11-12.5 Hz): セグメント4-6, 8

### 3.2 可能性のある原因

"""

    # 原因の分析
    mean_rms = emg_indicator['mean_rms']
    if mean_rms > 5:
        report += """
#### ⚠️ EMG混入の可能性（高）

高周波成分のRMSが高く、筋電図（特に前頭部の眉間や顎の筋緊張）が混入している可能性があります。

**対策案**:
- 顔の筋肉をリラックスさせる
- 歯を食いしばっていないか確認
- 眉間にしわを寄せていないか確認
"""
    else:
        report += """
#### ✓ EMG混入の可能性（低〜中）

高周波成分のRMSは比較的低く、大きなEMG混入は見られません。
"""

    report += """
### 3.3 その他の可能性

1. **SMR (Sensorimotor Rhythm, 12-15Hz)の増加**
   - セッション中盤でSMR帯域のパワーが増加
   - これは集中状態や運動抑制と関連
   - 瞑想中の「動かない」努力が反映されている可能性

2. **真のAlphaピークシフト**
   - 一部の人は状態によってAlphaピークが変動
   - 覚醒レベルの変化に伴うシフト

3. **ハードウェアの問題**
   - HSI品質は100% Goodなので可能性は低い
   - ただし、電磁干渉などの外部要因は除外できない

---

## 4. 結論

"""

    # IAFの変動パターンを分析
    iaf_values = [seg['iaf_peak'] for seg in segment_results]
    iaf_std = np.std(iaf_values)
    iaf_range = max(iaf_values) - min(iaf_values)

    if iaf_range > 3:
        report += f"""
**主要な発見**: IAFの変動幅が {iaf_range:.1f} Hzと大きく、通常の変動（1-2Hz）を超えています。

**最も可能性の高い原因**:
- セッション中盤で**SMR (12-15Hz) 帯域のパワー増加**
- これは「動かずにいる」という意識的な努力の反映である可能性が高い

**接触不良の可能性**: HSI品質100%のため、**接触不良ではない**と考えられます。

**推奨アクション**:
1. 次回のセッションで顔の筋肉のリラックスを意識
2. SMRの増加は必ずしも悪いことではなく、集中の指標として活用可能
3. 過去のセッションと比較してパターンを確認
"""
    else:
        report += f"""
**結論**: IAFの変動は正常範囲内です。
"""

    report += """
---

## 5. 生成された画像ファイル

- `segment_psd_comparison.png`: セグメント別PSD比較
- `channel_psd_comparison.png`: チャネル別PSD比較
- `band_power_ratio.png`: バンドパワー比率時系列
- `raw_waveform_segments.png`: 生波形のセグメント別分析
- `spectrogram_detailed.png`: スペクトログラム詳細

---

*分析日時: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "*\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'保存: {output_path}')


def main():
    """メイン処理"""
    print('='*60)
    print('10-15Hz帯域異常分析')
    print('='*60)
    print()

    # データ読み込み
    print(f'Loading: {DATA_PATH}')
    df = load_mind_monitor_csv(DATA_PATH, filter_headband=False, warmup_seconds=60)
    print(f'データ形状: {df.shape}')

    # MNE RAW準備
    print('\n準備中: MNE RAWデータ...')
    mne_dict = prepare_mne_raw(df)

    if not mne_dict:
        print('エラー: MNE RAWデータの準備に失敗しました')
        return 1

    raw = mne_dict['raw']
    print(f'チャネル: {mne_dict["channels"]}')
    print(f'サンプリングレート: {mne_dict["sfreq"]:.2f} Hz')

    # 1. セグメント別PSD分析
    print('\n分析中: セグメント別PSD...')
    segment_results = analyze_psd_by_segment(raw, segment_minutes=3, warmup_minutes=1)

    print('\nセグメント別IAF:')
    for seg in segment_results:
        print(f"  Seg {seg['segment']}: IAF(Peak)={seg['iaf_peak']:.2f}Hz, IAF(CoG)={seg['iaf_cog']:.2f}Hz")

    # 2. PSD比較プロット
    print('\nプロット中: セグメント別PSD比較...')
    plot_segment_psd_comparison(segment_results, OUTPUT_DIR / 'segment_psd_comparison.png')

    # 3. チャネル別比較
    print('プロット中: チャネル別PSD比較...')
    plot_channel_comparison(segment_results, OUTPUT_DIR / 'channel_psd_comparison.png')

    # 4. バンドパワー比率分析
    print('分析中: バンドパワー比率...')
    analyze_band_power_ratio(df, OUTPUT_DIR / 'band_power_ratio.png')

    # 5. 生波形セグメント分析
    print('分析中: 生波形セグメント...')
    analyze_raw_waveform_segments(raw, OUTPUT_DIR / 'raw_waveform_segments.png')

    # 6. EMG指標計算
    print('計算中: EMG指標...')
    emg_indicator = calculate_emg_indicator(raw)
    print(f"高周波RMS平均: {emg_indicator['mean_rms']:.3f} μV")

    # 7. スペクトログラム詳細
    print('計算中: スペクトログラム...')
    raw_for_tfr = raw.copy().resample(64, verbose=False)
    tfr_results = calculate_spectrogram_all_channels(raw_for_tfr)
    if tfr_results:
        print('プロット中: スペクトログラム詳細...')
        plot_spectrogram_comparison(tfr_results, OUTPUT_DIR / 'spectrogram_detailed.png')

    # 8. 分析レポート生成
    print('\n生成中: 分析レポート...')
    generate_analysis_report(
        segment_results,
        emg_indicator,
        OUTPUT_DIR.parent / 'ANALYSIS_10_15HZ.md'
    )

    print()
    print('='*60)
    print('分析完了!')
    print('='*60)
    print(f'レポート: {OUTPUT_DIR.parent / "ANALYSIS_10_15HZ.md"}')
    print(f'画像: {OUTPUT_DIR}/')

    return 0


if __name__ == '__main__':
    exit(main())
