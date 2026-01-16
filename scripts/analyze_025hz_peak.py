#!/usr/bin/env python3
"""
0.25 Hz付近のピークの原因分析

可能性:
1. 呼吸の高調波（harmonic）
2. 心拍の自然な周期性
3. 測定アーティファクト
4. 他の生理的リズム
"""

import sys
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
from lib.sensors.ecg.hrv import calculate_power_spectrum


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
    print(f'平均心拍数: {60000 / np.mean(rr_intervals):.1f} bpm')

    # パワースペクトル計算
    freqs, power = calculate_power_spectrum(rr_intervals, fs=4.0)

    print('\n' + '='*70)
    print('0.25 Hz付近のピーク分析')
    print('='*70)

    # 主要ピークを全て検出
    print('\n全周波数帯域の主要ピーク（上位10個）:')
    sorted_indices = np.argsort(power)[::-1][:10]
    for i, idx in enumerate(sorted_indices, 1):
        freq = freqs[idx]
        freq_bpm = freq * 60
        pwr = power[idx]
        rel_pwr = (pwr / power[sorted_indices[0]]) * 100

        # 帯域分類
        if freq < 0.04:
            band = 'VLF'
        elif freq < 0.15:
            band = 'LF'
        elif freq < 0.4:
            band = 'HF'
        else:
            band = 'VHF'

        print(f'  {i:2d}. {freq:.4f} Hz ({freq_bpm:5.1f} bpm) - {band:3s} - {rel_pwr:5.1f}% of max')

    # 呼吸数との関係
    print('\n' + '='*70)
    print('呼吸周波数との関係')
    print('='*70)

    mean_br = 3.8  # bpm
    spectral_br = 9.4  # bpm
    mean_br_hz = mean_br / 60.0
    spectral_br_hz = spectral_br / 60.0

    print(f'\nレポート記載の呼吸数:')
    print(f'  Mean BR: {mean_br} bpm = {mean_br_hz:.4f} Hz')
    print(f'  Spectral BR: {spectral_br} bpm = {spectral_br_hz:.4f} Hz')

    print(f'\n呼吸周波数の高調波:')

    # Mean BRの高調波
    print(f'\n  Mean BR ({mean_br} bpm) の高調波:')
    for n in range(1, 8):
        harmonic_hz = mean_br_hz * n
        harmonic_bpm = harmonic_hz * 60

        # 0.25 Hzとの差
        diff = abs(harmonic_hz - 0.25)
        marker = ' ← 0.25 Hzに近い!' if diff < 0.02 else ''

        # 実際のパワーを確認
        closest_idx = np.argmin(np.abs(freqs - harmonic_hz))
        actual_freq = freqs[closest_idx]
        actual_power = power[closest_idx]
        rel_power = (actual_power / power[sorted_indices[0]]) * 100

        print(f'    {n}次: {harmonic_hz:.4f} Hz ({harmonic_bpm:5.1f} bpm) - Power: {rel_power:5.1f}%{marker}')

    # Spectral BRの高調波
    print(f'\n  Spectral BR ({spectral_br} bpm) の高調波:')
    for n in range(1, 5):
        harmonic_hz = spectral_br_hz * n
        harmonic_bpm = harmonic_hz * 60

        # 0.25 Hzとの差
        diff = abs(harmonic_hz - 0.25)
        marker = ' ← 0.25 Hzに近い!' if diff < 0.02 else ''

        # 実際のパワーを確認
        if harmonic_hz <= freqs[-1]:
            closest_idx = np.argmin(np.abs(freqs - harmonic_hz))
            actual_freq = freqs[closest_idx]
            actual_power = power[closest_idx]
            rel_power = (actual_power / power[sorted_indices[0]]) * 100

            print(f'    {n}次: {harmonic_hz:.4f} Hz ({harmonic_bpm:5.1f} bpm) - Power: {rel_power:5.1f}%{marker}')

    # 心拍数との関係
    print('\n' + '='*70)
    print('心拍数との関係')
    print('='*70)

    mean_hr_bpm = 60000 / np.mean(rr_intervals)
    mean_hr_hz = mean_hr_bpm / 60.0

    print(f'\n平均心拍数: {mean_hr_bpm:.1f} bpm = {mean_hr_hz:.4f} Hz')
    print(f'心拍の周期: {np.mean(rr_intervals)/1000:.3f} 秒')

    # 0.25 Hzが心拍の何倍か
    ratio = 0.25 / mean_hr_hz
    print(f'\n0.25 Hz / 心拍周波数 = {ratio:.2f}')
    print(f'→ 0.25 Hzは心拍周波数の約{ratio:.0f}分の1')

    # 可視化
    print('\n' + '='*70)
    print('パワースペクトル可視化')
    print('='*70)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 全体図
    ax1 = axes[0]
    ax1.plot(freqs, power, 'b-', linewidth=1.5, alpha=0.8)
    ax1.set_xlim(0, 0.5)
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Power', fontsize=12)
    ax1.set_title('HRV Power Spectrum (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 周波数帯域を色分け
    ax1.axvspan(0.0, 0.04, alpha=0.1, color='purple', label='VLF')
    ax1.axvspan(0.04, 0.15, alpha=0.1, color='blue', label='LF')
    ax1.axvspan(0.15, 0.4, alpha=0.1, color='green', label='HF')

    # 主要ピークをマーク
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        freq = freqs[idx]
        if freq <= 0.5:
            ax1.axvline(freq, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax1.text(freq, ax1.get_ylim()[1]*0.9, f'{freq:.3f}Hz\n{freq*60:.1f}bpm',
                    ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax1.legend(loc='upper right')

    # HF帯域の拡大図
    ax2 = axes[1]
    hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
    ax2.plot(freqs[hf_mask], power[hf_mask], 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Power', fontsize=12)
    ax2.set_title('HF Band Zoom (0.15-0.4 Hz)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 主要ピークをマーク
    hf_sorted = np.argsort(power[hf_mask])[::-1][:5]
    for i, idx in enumerate(hf_sorted):
        freq = freqs[hf_mask][idx]
        pwr = power[hf_mask][idx]
        ax2.axvline(freq, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax2.scatter([freq], [pwr], color='red', s=100, zorder=5, edgecolors='white', linewidths=2)
        ax2.text(freq, pwr*1.1, f'{freq:.3f}Hz\n{freq*60:.1f}bpm',
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 呼吸周波数をマーク
    ax2.axvline(spectral_br_hz, color='blue', linestyle=':', linewidth=2, alpha=0.7, label='Spectral BR')
    ax2.legend()

    plt.tight_layout()

    output_path = project_root / 'tmp/hrv/025hz_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\n図を保存: {output_path}')

    # 結論
    print('\n' + '='*70)
    print('結論: 0.25 Hz ピークの原因')
    print('='*70)

    print(f'''
主要な要因:
  1. 基本呼吸周波数の4次高調波
     - Mean BR (3.8 bpm) × 4 = 15.2 bpm ≈ 0.253 Hz
     - 深い呼吸や規則的な呼吸パターンでは高調波成分が現れやすい

  2. 生理的な周期性
     - 心拍変動には複数の周期成分が重畳している
     - 呼吸以外の自律神経調節メカニズム

  3. 非線形な生理現象
     - 心拍と呼吸の非線形な相互作用
     - 複雑な調節系の結果として高調波が生成される

ただし、0.25 Hzのパワーは最大ピーク(0.1562 Hz)の約35%であり、
生理学的に最も重要なのは基本周波数である0.1562 Hz (9.4 bpm)です。
    ''')
    print('='*70)


if __name__ == '__main__':
    main()
