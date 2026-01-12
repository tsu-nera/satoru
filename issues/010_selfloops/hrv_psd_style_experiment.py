#!/usr/bin/env python3
"""
HRV周波数解析をEEG PSDスタイルで表示する実験スクリプト

既存のEEG PSD可視化スタイル（lib/sensors/eeg/visualization/eeg_plots.py:plot_psd）
と同じビジュアルスタイルでHRVの周波数スペクトルを表示。
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
from lib.sensors.ecg.visualization.hrv_plot import HRV_FREQ_BANDS
from lib.visualization.utils import power_to_db, apply_frequency_band_shading, style_frequency_plot


def plot_hrv_frequency_eeg_style(
    hrv_data,
    img_path=None,
    freq_max=0.5
):
    """
    EEG PSDスタイルでHRV周波数解析をプロット

    Parameters
    ----------
    hrv_data : dict
        HRVデータ（rr_intervals, sampling_rate）
    img_path : str or Path, optional
        保存先パス
    freq_max : float
        表示する最大周波数（Hz）

    Returns
    -------
    fig : matplotlib.figure.Figure
        生成された図
    """
    # NeuroKit2でピーク検出
    peaks = nk.intervals_to_peaks(
        hrv_data['rr_intervals'],
        sampling_rate=hrv_data['sampling_rate']
    )

    # 周波数解析（パワースペクトル密度）
    hrv_freq = nk.hrv_frequency(
        peaks,
        sampling_rate=hrv_data['sampling_rate'],
        show=False
    )

    # PSDデータを取得（NeuroKit2内部実装を模倣）
    # Welch法でPSDを計算
    from scipy import signal

    rr_ms = hrv_data['rr_intervals']
    # RR間隔の時系列を等間隔に補間
    rr_times = np.cumsum(rr_ms) / 1000.0  # msからsに変換
    rr_times = np.insert(rr_times, 0, 0)
    rr_values = np.append(rr_ms, rr_ms[-1])

    # 4Hzで補間（NeuroKit2デフォルト）
    sampling_rate = 4.0
    interpolated_times = np.arange(0, rr_times[-1], 1.0 / sampling_rate)
    interpolated_rr = np.interp(interpolated_times, rr_times, rr_values)

    # Welch法でPSD計算
    freqs, psd = signal.welch(
        interpolated_rr,
        fs=sampling_rate,
        nperseg=min(len(interpolated_rr), 256),
        window='hann'
    )

    # dBスケールに変換（共通関数を使用）
    psd_db = power_to_db(psd)

    # プロット（EEG PSDスタイル）
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # 周波数範囲をマスク
    mask = (freqs >= 0.0) & (freqs <= freq_max)

    # メインのPSDライン
    ax.plot(freqs[mask], psd_db[mask], 'b-', linewidth=2.0, alpha=0.9, label='HRV PSD')

    # 周波数帯域を色分け（共通関数を使用）
    apply_frequency_band_shading(ax, HRV_FREQ_BANDS, freq_max, alpha=0.12)

    # HRV指標から取得した値をマーク
    lf_power = hrv_freq['HRV_LF'].values[0]
    hf_power = hrv_freq['HRV_HF'].values[0]
    lf_hf_ratio = hrv_freq['HRV_LFHF'].values[0]

    # ピーク周波数を推定
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.4)

    if np.any(lf_mask):
        lf_peak_idx = np.argmax(psd[lf_mask])
        lf_peak_freq = freqs[lf_mask][lf_peak_idx]
        lf_peak_power = psd_db[lf_mask][lf_peak_idx]
        ax.scatter([lf_peak_freq], [lf_peak_power], color='blue', s=150,
                  marker='o', zorder=5, edgecolors='white', linewidths=2, label='LF Peak')

    if np.any(hf_mask):
        hf_peak_idx = np.argmax(psd[hf_mask])
        hf_peak_freq = freqs[hf_mask][hf_peak_idx]
        hf_peak_power = psd_db[hf_mask][hf_peak_idx]
        ax.scatter([hf_peak_freq], [hf_peak_power], color='green', s=150,
                  marker='o', zorder=5, edgecolors='white', linewidths=2, label='HF Peak')

    # 軸設定（共通関数を使用）
    style_frequency_plot(ax, freq_max, title='HRV Frequency Analysis (EEG PSD Style)')

    # テキスト注釈（HRV指標）
    text_str = f'LF Power: {lf_power:.2f} ms²\nHF Power: {hf_power:.2f} ms²\nLF/HF: {lf_hf_ratio:.2f}'
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if img_path:
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='HRV周波数解析のEEG PSDスタイル表示実験'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='SelfLoops HRV dataファイルパス (.csv)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='出力画像パス（デフォルト: hrv_psd_eeg_style.png）'
    )
    parser.add_argument(
        '--warmup',
        type=float,
        default=0.0,
        help='測定開始からの除外期間（秒）'
    )

    args = parser.parse_args()

    # 出力パスの設定
    if args.output is None:
        args.output = Path('hrv_psd_eeg_style.png')

    # パスの検証
    if not args.data.exists():
        print(f'エラー: データファイルが見つかりません: {args.data}')
        return 1

    print('='*60)
    print('HRV周波数解析 - EEG PSDスタイル実験')
    print('='*60)
    print()

    # データ読み込み
    print(f'📁 Loading: {args.data.name}')
    df = load_selfloops_csv(str(args.data), warmup_seconds=args.warmup)
    print(f'   データ: {len(df)}点, {df["Time_sec"].iloc[-1] / 60:.1f}分')
    if args.warmup > 0:
        print(f'   ウォームアップ除外: {args.warmup}秒')
    print()

    # HRVデータ取得
    print('🔬 HRVデータ準備中...')
    hrv_data = get_hrv_data(df)
    print(f'   RR間隔数: {len(hrv_data["rr_intervals"])}')
    print(f'   サンプリングレート: {hrv_data["sampling_rate"]} Hz')
    print()

    # EEGスタイルのプロット生成
    print('📊 プロット生成中（EEG PSDスタイル）...')
    plot_hrv_frequency_eeg_style(hrv_data, img_path=args.output)
    print(f'✓ 保存完了: {args.output}')
    print()

    print('='*60)
    print('✅ 実験完了!')
    print('='*60)

    return 0


if __name__ == '__main__':
    exit(main())
