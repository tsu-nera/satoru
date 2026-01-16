#!/usr/bin/env python3
"""
HF Peak周波数のデバッグスクリプト

プロット用とピーク検出用の2つの方法でパワースペクトルを計算し、
ピーク周波数を比較する。
"""

import sys
from pathlib import Path
import numpy as np
from scipy import signal

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
from lib.sensors.ecg.hrv import calculate_power_spectrum, find_peak_frequency


def main():
    # データ読み込み
    data_path = project_root / 'data/selfloops/selfloops_2026-01-16--07-18-39.csv'
    print(f'Loading: {data_path}')
    sl_df = load_selfloops_csv(str(data_path), warmup_seconds=60.0)

    # HRVデータ取得
    hrv_data = get_hrv_data(sl_df, clean_artifacts=True)
    rr_intervals = hrv_data['rr_intervals_clean']

    print(f'\nR-R間隔数: {len(rr_intervals)}')
    print(f'平均R-R間隔: {np.mean(rr_intervals):.1f} ms')

    # ========================================
    # 方法1: calculate_power_spectrum() (ピーク検出で使用)
    # ========================================
    print('\n' + '='*60)
    print('方法1: calculate_power_spectrum() (デトレンド+ハニング窓)')
    print('='*60)

    freqs1, power1 = calculate_power_spectrum(rr_intervals, fs=4.0)

    # HF帯域のピークを探す
    hf_mask1 = (freqs1 >= 0.15) & (freqs1 <= 0.4)
    if np.any(hf_mask1):
        hf_peak_idx1 = np.argmax(power1[hf_mask1])
        hf_peak_freq1 = freqs1[hf_mask1][hf_peak_idx1]
        hf_peak_power1 = power1[hf_mask1][hf_peak_idx1]
        print(f'HF Peak: {hf_peak_freq1:.4f} Hz ({hf_peak_freq1 * 60:.1f} bpm)')
        print(f'HF Peak Power: {hf_peak_power1:.6f}')

    # LF帯域のピークも確認
    lf_mask1 = (freqs1 >= 0.04) & (freqs1 <= 0.15)
    if np.any(lf_mask1):
        lf_peak_idx1 = np.argmax(power1[lf_mask1])
        lf_peak_freq1 = freqs1[lf_mask1][lf_peak_idx1]
        lf_peak_power1 = power1[lf_mask1][lf_peak_idx1]
        print(f'LF Peak: {lf_peak_freq1:.4f} Hz ({lf_peak_freq1 * 60:.1f} bpm)')
        print(f'LF Peak Power: {lf_peak_power1:.6f}')

    # HF帯域の全ピークを表示（上位5つ）
    print(f'\nHF帯域(0.15-0.4 Hz)の上位5ピーク:')
    hf_freqs = freqs1[hf_mask1]
    hf_powers = power1[hf_mask1]
    sorted_indices = np.argsort(hf_powers)[::-1][:5]
    for i, idx in enumerate(sorted_indices, 1):
        freq = hf_freqs[idx]
        power = hf_powers[idx]
        print(f'  {i}. {freq:.4f} Hz ({freq*60:.1f} bpm) - Power: {power:.6f}')

    # ========================================
    # 方法2: signal.welch() 直接使用 (プロット関数で使用)
    # ========================================
    print('\n' + '='*60)
    print('方法2: signal.welch() 直接使用 (プロット関数)')
    print('='*60)

    # RR間隔を等間隔に補間
    rr_ms = rr_intervals
    rr_times = np.cumsum(rr_ms) / 1000.0
    rr_times = np.insert(rr_times, 0, 0)
    rr_values = np.append(rr_ms, rr_ms[-1])

    sampling_rate = 4.0
    interpolated_times = np.arange(0, rr_times[-1], 1.0 / sampling_rate)
    interpolated_rr = np.interp(interpolated_times, rr_times, rr_values)

    # Welch法でPSD計算
    freqs2, power2 = signal.welch(
        interpolated_rr,
        fs=sampling_rate,
        nperseg=min(len(interpolated_rr), 256),
        window='hann'
    )

    # HF帯域のピークを探す
    hf_mask2 = (freqs2 >= 0.15) & (freqs2 <= 0.4)
    if np.any(hf_mask2):
        hf_peak_idx2 = np.argmax(power2[hf_mask2])
        hf_peak_freq2 = freqs2[hf_mask2][hf_peak_idx2]
        hf_peak_power2 = power2[hf_mask2][hf_peak_idx2]
        print(f'HF Peak: {hf_peak_freq2:.4f} Hz ({hf_peak_freq2 * 60:.1f} bpm)')
        print(f'HF Peak Power: {hf_peak_power2:.6f}')

    # LF帯域のピークも確認
    lf_mask2 = (freqs2 >= 0.04) & (freqs2 <= 0.15)
    if np.any(lf_mask2):
        lf_peak_idx2 = np.argmax(power2[lf_mask2])
        lf_peak_freq2 = freqs2[lf_mask2][lf_peak_idx2]
        lf_peak_power2 = power2[lf_mask2][lf_peak_idx2]
        print(f'LF Peak: {lf_peak_freq2:.4f} Hz ({lf_peak_freq2 * 60:.1f} bpm)')
        print(f'LF Peak Power: {lf_peak_power2:.6f}')

    # HF帯域の全ピークを表示（上位5つ）
    print(f'\nHF帯域(0.15-0.4 Hz)の上位5ピーク:')
    hf_freqs = freqs2[hf_mask2]
    hf_powers = power2[hf_mask2]
    sorted_indices = np.argsort(hf_powers)[::-1][:5]
    for i, idx in enumerate(sorted_indices, 1):
        freq = hf_freqs[idx]
        power = hf_powers[idx]
        print(f'  {i}. {freq:.4f} Hz ({freq*60:.1f} bpm) - Power: {power:.6f}')

    # 0.25 Hz付近の詳細を確認
    print(f'\n0.25 Hz付近(0.23-0.27 Hz)のパワー値:')
    range_mask = (freqs2 >= 0.23) & (freqs2 <= 0.27)
    if np.any(range_mask):
        for freq, power in zip(freqs2[range_mask], power2[range_mask]):
            rel_power = (power / hf_peak_power2) * 100
            print(f'  {freq:.4f} Hz ({freq*60:.1f} bpm) - Power: {power:.2f} ({rel_power:.1f}% of peak)')

    # ========================================
    # 比較
    # ========================================
    print('\n' + '='*60)
    print('比較')
    print('='*60)
    print(f'HF Peak差: {abs(hf_peak_freq1 - hf_peak_freq2):.4f} Hz')
    print(f'LF Peak差: {abs(lf_peak_freq1 - lf_peak_freq2):.4f} Hz')

    # 呼吸数との比較
    print('\n' + '='*60)
    print('呼吸数との整合性確認')
    print('='*60)
    print('レポートの呼吸数:')
    print('  - Mean Breathing Rate: 3.8 bpm (0.0633 Hz)')
    print('  - Breathing Rate (Spectral): 9.4 bpm (0.1567 Hz)')
    print(f'\n計算されたHF Peak: {hf_peak_freq1:.4f} Hz ({hf_peak_freq1*60:.1f} bpm)')
    print(f'  → Spectral BRとの差: {abs(hf_peak_freq1*60 - 9.4):.1f} bpm')


if __name__ == '__main__':
    main()
