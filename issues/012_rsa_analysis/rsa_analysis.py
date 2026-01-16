#!/usr/bin/env python3
"""
呼吸性洞性不整脈（RSA）分析スクリプト

RSA（Respiratory Sinus Arrhythmia）は呼吸に同期した心拍変動で、
副交感神経活動の指標として重要です。

主要分析項目:
1. RSA振幅の時系列変化（HF Power基準）
2. 呼吸-心拍の同期性（Coherence分析）
3. DFA分析（ゆらぎ解析）
4. 周波数領域解析（HF/LF比較）

参考文献:
- Heart rate dynamics in different levels of Zen meditation
  (International Journal of Cardiology, 2010)
- 経験豊富な瞑想者: DFA ≈ 0.5
- 初心者: DFA ≈ 0.78
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal, interpolate

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
from lib.sensors.ecg.respiration import (
    calculate_breathing_rate,
    analyze_breathing_hrv_correlation
)


def calculate_rsa_amplitude(rr_intervals, sampling_rate=1000, method='hf_power'):
    """
    RSA振幅を計算

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R間隔（ms）
    sampling_rate : int
        サンプリングレート（Hz）
    method : str
        'hf_power': HF成分のパワー（ms²）
        'peak_valley': ピーク-バレー差分（bpm）

    Returns
    -------
    float
        RSA振幅
    """
    if len(rr_intervals) < 10:
        return np.nan

    try:
        if method == 'hf_power':
            # HF成分（0.15-0.4 Hz）のパワーでRSA振幅を評価
            peaks = nk.intervals_to_peaks(rr_intervals, sampling_rate=sampling_rate)
            hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sampling_rate,
                                       show=False, normalize=False)
            return hrv_freq['HRV_HF'].values[0]
        elif method == 'peak_valley':
            # 心拍数の最大-最小差でRSA振幅を評価
            hr = 60000.0 / rr_intervals
            return np.max(hr) - np.min(hr)
    except Exception:
        return np.nan


def calculate_dfa(rr_intervals, sampling_rate=1000):
    """
    DFA (Detrended Fluctuation Analysis) を計算

    ゆらぎの長期相関を評価する指標
    - DFA < 0.5: 反相関（平均への回帰傾向）
    - DFA ≈ 0.5: ランダムウォーク
    - 0.5 < DFA < 1.0: 長期相関
    - DFA ≈ 1.0: 1/fノイズ（ピンクノイズ）

    瞑想の文脈:
    - 経験豊富な瞑想者: DFA ≈ 0.5
    - 初心者: DFA ≈ 0.78

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R間隔（ms）
    sampling_rate : int
        サンプリングレート（Hz）

    Returns
    -------
    dict
        - dfa_alpha1: 短期スケールのDFA指数
        - dfa_alpha2: 長期スケールのDFA指数
    """
    if len(rr_intervals) < 50:
        return {'dfa_alpha1': np.nan, 'dfa_alpha2': np.nan}

    try:
        peaks = nk.intervals_to_peaks(rr_intervals, sampling_rate=sampling_rate)
        hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=sampling_rate, show=False)

        return {
            'dfa_alpha1': hrv_nonlinear['HRV_DFA_alpha1'].values[0],
            'dfa_alpha2': hrv_nonlinear['HRV_DFA_alpha2'].values[0]
        }
    except Exception:
        return {'dfa_alpha1': np.nan, 'dfa_alpha2': np.nan}


def calculate_respiratory_heart_coherence(hr_signal, rsp_signal, fs=8.0):
    """
    呼吸-心拍間のコヒーレンスを計算

    Parameters
    ----------
    hr_signal : np.ndarray
        心拍数時系列
    rsp_signal : np.ndarray
        呼吸信号時系列
    fs : float
        サンプリング周波数（Hz）

    Returns
    -------
    dict
        - coherence_freq: コヒーレンス周波数軸
        - coherence: コヒーレンス値
        - peak_coherence: 最大コヒーレンス値
        - peak_freq: 最大コヒーレンスの周波数
    """
    # コヒーレンス計算（welch法）
    f, Cxy = signal.coherence(hr_signal, rsp_signal, fs=fs, nperseg=min(256, len(hr_signal)))

    # 呼吸帯域（0.05-0.5 Hz）での最大コヒーレンス
    mask = (f >= 0.05) & (f <= 0.5)
    f_br = f[mask]
    Cxy_br = Cxy[mask]

    if len(Cxy_br) > 0:
        peak_idx = np.argmax(Cxy_br)
        peak_coherence = Cxy_br[peak_idx]
        peak_freq = f_br[peak_idx]
    else:
        peak_coherence = np.nan
        peak_freq = np.nan

    return {
        'coherence_freq': f,
        'coherence': Cxy,
        'peak_coherence': peak_coherence,
        'peak_freq': peak_freq
    }


def analyze_rsa_time_series(hrv_data, window_minutes=3):
    """
    RSA指標の時系列変化を分析

    Parameters
    ----------
    hrv_data : dict
        HRVデータ
    window_minutes : float
        ウィンドウサイズ（分）

    Returns
    -------
    pd.DataFrame
        時系列RSA指標
    """
    window_sec = window_minutes * 60
    rr_intervals = hrv_data['rr_intervals_clean']
    rr_time = hrv_data['time']
    total_duration = rr_time[-1]

    results = []

    for start_time in np.arange(0, total_duration, window_sec):
        end_time = min(start_time + window_sec, total_duration)
        timestamp_min = (start_time + end_time) / 2 / 60

        # ウィンドウ内のR-R間隔
        mask = (rr_time >= start_time) & (rr_time < end_time)
        rr_window = rr_intervals[mask]

        if len(rr_window) > 50:
            # RSA振幅（HF Power）
            rsa_amplitude = calculate_rsa_amplitude(rr_window, method='hf_power')

            # DFA指標
            dfa = calculate_dfa(rr_window)

            # 基本HRV指標
            try:
                peaks = nk.intervals_to_peaks(rr_window, sampling_rate=1000)
                hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False, normalize=False)

                rmssd = hrv_time['HRV_RMSSD'].values[0]
                sdnn = hrv_time['HRV_SDNN'].values[0]
                lf_power = hrv_freq['HRV_LF'].values[0]
                hf_power = hrv_freq['HRV_HF'].values[0]
                lf_hf = hrv_freq['HRV_LFHF'].values[0]
            except Exception:
                rmssd = np.nan
                sdnn = np.nan
                lf_power = np.nan
                hf_power = np.nan
                lf_hf = np.nan
        else:
            rsa_amplitude = np.nan
            dfa = {'dfa_alpha1': np.nan, 'dfa_alpha2': np.nan}
            rmssd = np.nan
            sdnn = np.nan
            lf_power = np.nan
            hf_power = np.nan
            lf_hf = np.nan

        results.append({
            'Time (min)': timestamp_min,
            'RSA Amplitude (HF Power, ms^2)': rsa_amplitude,
            'DFA α1': dfa['dfa_alpha1'],
            'DFA α2': dfa['dfa_alpha2'],
            'RMSSD (ms)': rmssd,
            'SDNN (ms)': sdnn,
            'LF Power (ms^2)': lf_power,
            'HF Power (ms^2)': hf_power,
            'LF/HF': lf_hf
        })

    return pd.DataFrame(results)


def plot_rsa_analysis(hrv_data, rsa_metrics_df, breathing_result, coherence_result, output_dir):
    """
    RSA分析結果を可視化

    Parameters
    ----------
    hrv_data : dict
        HRVデータ
    rsa_metrics_df : pd.DataFrame
        RSA時系列指標
    breathing_result : RespirationResult
        呼吸分析結果
    coherence_result : dict
        コヒーレンス分析結果
    output_dir : Path
        出力ディレクトリ
    """
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)

    # (1) RSA振幅の時系列変化
    ax1 = fig.add_subplot(gs[0, :])
    time = rsa_metrics_df['Time (min)']
    rsa_amp = rsa_metrics_df['RSA Amplitude (HF Power, ms^2)']
    ax1.plot(time, rsa_amp, 'o-', linewidth=2, markersize=6, color='blue', label='HF Power')
    ax1.axhline(np.nanmean(rsa_amp), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.nanmean(rsa_amp):.1f} ms²')
    ax1.set_xlabel('Time (min)', fontsize=12)
    ax1.set_ylabel('RSA Amplitude (HF Power, ms²)', fontsize=12)
    ax1.set_title('Respiratory Sinus Arrhythmia (RSA) Amplitude Over Time',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # (2) DFA指標の時系列変化
    ax2 = fig.add_subplot(gs[1, :])
    dfa1 = rsa_metrics_df['DFA α1']
    dfa2 = rsa_metrics_df['DFA α2']
    ax2.plot(time, dfa1, 's-', linewidth=2, markersize=6, color='green', label='DFA α1 (short-term)')
    ax2.plot(time, dfa2, '^-', linewidth=2, markersize=6, color='orange', label='DFA α2 (long-term)')

    # 参照線: 経験豊富な瞑想者 vs 初心者
    ax2.axhline(0.5, color='blue', linestyle=':', linewidth=2,
               label='Experienced meditator (~0.5)')
    ax2.axhline(0.78, color='red', linestyle=':', linewidth=2,
               label='Beginner (~0.78)')

    ax2.set_xlabel('Time (min)', fontsize=12)
    ax2.set_ylabel('DFA Exponent', fontsize=12)
    ax2.set_title('Detrended Fluctuation Analysis (DFA) - Heart Rate Variability Dynamics',
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.5])

    # (3) 呼吸数 vs RSA振幅
    ax3 = fig.add_subplot(gs[2, 0])
    br_data = breathing_result.time_series
    merged = pd.merge(br_data, rsa_metrics_df, on='Time (min)', how='inner')

    ax3.scatter(merged['BR (bpm)'], merged['RSA Amplitude (HF Power, ms^2)'],
               alpha=0.7, s=80, c=merged['Time (min)'], cmap='viridis',
               edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Breathing Rate (bpm)', fontsize=12)
    ax3.set_ylabel('RSA Amplitude (HF Power, ms²)', fontsize=12)
    ax3.set_title('Breathing Rate vs RSA Amplitude', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Time (min)', fontsize=10)

    # (4) LF vs HF Power
    ax4 = fig.add_subplot(gs[2, 1])
    lf = rsa_metrics_df['LF Power (ms^2)']
    hf = rsa_metrics_df['HF Power (ms^2)']
    ax4.scatter(lf, hf, alpha=0.7, s=80, c=time, cmap='plasma',
               edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('LF Power (ms²)', fontsize=12)
    ax4.set_ylabel('HF Power (ms²)', fontsize=12)
    ax4.set_title('LF Power vs HF Power (RSA)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Time (min)', fontsize=10)

    # (5) LF/HF比の時系列変化
    ax5 = fig.add_subplot(gs[3, :])
    lf_hf = rsa_metrics_df['LF/HF']
    ax5.plot(time, lf_hf, 'o-', linewidth=2, markersize=6, color='purple')
    ax5.axhline(np.nanmean(lf_hf), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.nanmean(lf_hf):.2f}')
    ax5.set_xlabel('Time (min)', fontsize=12)
    ax5.set_ylabel('LF/HF Ratio', fontsize=12)
    ax5.set_title('Autonomic Balance (LF/HF Ratio) Over Time', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # (6) 呼吸-心拍コヒーレンス
    ax6 = fig.add_subplot(gs[4, :])
    if coherence_result is not None:
        freqs = coherence_result['coherence_freq']
        coherence = coherence_result['coherence']
        peak_freq = coherence_result['peak_freq']
        peak_coh = coherence_result['peak_coherence']

        ax6.plot(freqs, coherence, linewidth=2, color='navy')
        ax6.axvline(peak_freq, color='red', linestyle='--', linewidth=2,
                   label=f'Peak: {peak_freq:.3f} Hz ({peak_freq*60:.1f} bpm), Coh={peak_coh:.3f}')
        ax6.axvspan(0.15, 0.4, alpha=0.2, color='cyan', label='HF Band (RSA)')
        ax6.set_xlabel('Frequency (Hz)', fontsize=12)
        ax6.set_ylabel('Coherence', fontsize=12)
        ax6.set_title('Respiratory-Heart Rate Coherence', fontsize=14, fontweight='bold')
        ax6.set_xlim([0, 0.5])
        ax6.set_ylim([0, 1])
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Coherence data not available',
                ha='center', va='center', fontsize=14, color='red')
        ax6.set_xlim([0, 1])
        ax6.set_ylim([0, 1])

    output_path = output_dir / 'rsa_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'✓ RSAプロット保存: {output_path}')


def generate_rsa_report(hrv_data, rsa_metrics_df, breathing_result, coherence_result, output_dir):
    """
    RSA分析レポートを生成

    Parameters
    ----------
    hrv_data : dict
        HRVデータ
    rsa_metrics_df : pd.DataFrame
        RSA時系列指標
    breathing_result : RespirationResult
        呼吸分析結果
    coherence_result : dict
        コヒーレンス分析結果
    output_dir : Path
        出力ディレクトリ
    """
    # 統計サマリー
    mean_rsa = np.nanmean(rsa_metrics_df['RSA Amplitude (HF Power, ms^2)'])
    std_rsa = np.nanstd(rsa_metrics_df['RSA Amplitude (HF Power, ms^2)'])
    mean_dfa1 = np.nanmean(rsa_metrics_df['DFA α1'])
    std_dfa1 = np.nanstd(rsa_metrics_df['DFA α1'])
    mean_lf_hf = np.nanmean(rsa_metrics_df['LF/HF'])
    std_lf_hf = np.nanstd(rsa_metrics_df['LF/HF'])

    # 呼吸数情報
    mean_br = breathing_result.breathing_rate
    std_br = breathing_result.breathing_rate_std

    # コヒーレンス情報
    if coherence_result is not None:
        peak_coh = coherence_result['peak_coherence']
        peak_freq = coherence_result['peak_freq']
        coherence_section = f"""
### 呼吸-心拍コヒーレンス

- **最大コヒーレンス値**: {peak_coh:.3f}
- **最大コヒーレンス周波数**: {peak_freq:.3f} Hz ({peak_freq*60:.1f} bpm)

コヒーレンスが高い（>0.7）場合、呼吸と心拍が高度に同期しています。
"""
    else:
        coherence_section = """
### 呼吸-心拍コヒーレンス

データ不足のため計算できませんでした。
"""

    # DFAの評価
    if mean_dfa1 < 0.6:
        dfa_interpretation = "**経験豊富な瞑想者レベル** (DFA < 0.6)"
        dfa_note = "心拍のゆらぎがランダムウォークに近く、睡眠の境界に近い深い瞑想状態を示唆"
    elif mean_dfa1 < 0.9:
        dfa_interpretation = "**中級者レベル** (0.6 ≤ DFA < 0.9)"
        dfa_note = "適度な長期相関を持つゆらぎパターン"
    else:
        dfa_interpretation = "**初心者レベル** (DFA ≥ 0.9)"
        dfa_note = "より規則的な心拍パターン"

    # マークダウンテーブル生成
    def df_to_markdown(df):
        lines = []
        header = '| ' + ' | '.join(df.columns) + ' |'
        lines.append(header)
        separator = '|' + '|'.join([':-----:' for _ in df.columns]) + '|'
        lines.append(separator)

        for _, row in df.iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append('-')
                elif isinstance(val, float):
                    values.append(f'{val:.2f}')
                else:
                    values.append(str(val))
            line = '| ' + ' | '.join(values) + ' |'
            lines.append(line)

        return '\n'.join(lines)

    metrics_table = df_to_markdown(rsa_metrics_df)

    report = f"""# 呼吸性洞性不整脈（RSA）分析レポート

## 概要

呼吸性洞性不整脈（Respiratory Sinus Arrhythmia: RSA）は、呼吸に同期して心拍数が変動する正常な生理現象で、
副交感神経（迷走神経）活動の指標として重要です。

このレポートでは、ECGのR-R間隔データからRSAを定量化し、瞑想の習熟度を評価しました。

## 主要指標サマリー

| 指標 | 値 |
|:-----|---:|
| **測定時間** | {hrv_data['time'][-1]/60:.2f} 分 |
| **平均呼吸数** | {mean_br:.1f} ± {std_br:.1f} bpm |
| **平均RSA振幅（HF Power）** | {mean_rsa:.1f} ± {std_rsa:.1f} ms² |
| **平均DFA α1** | {mean_dfa1:.3f} ± {std_dfa1:.3f} |
| **平均LF/HF比** | {mean_lf_hf:.2f} ± {std_lf_hf:.2f} |

## RSA分析結果

### RSA振幅（HF Power）

- **平均**: {mean_rsa:.1f} ms²
- **標準偏差**: {std_rsa:.1f} ms²

RSA振幅はHF成分（0.15-0.4 Hz）のパワーで評価しています。
高い値ほど、呼吸による心拍調節が効果的であることを示します。

### DFA（ゆらぎ解析）

- **平均DFA α1**: {mean_dfa1:.3f}
- **評価**: {dfa_interpretation}
- **解釈**: {dfa_note}

**参考値（文献より）:**
- 経験豊富な禅瞑想者: DFA ≈ 0.5
- 初心者: DFA ≈ 0.78

**出典**: Peressutti et al., "Heart rate dynamics in different levels of Zen meditation",
International Journal of Cardiology, 2010

### 自律神経バランス（LF/HF比）

- **平均LF/HF比**: {mean_lf_hf:.2f}

LF/HF比が低いほど副交感神経優位（リラックス状態）を示します。
- < 1.0: 副交感神経優位
- 1.0-2.0: バランス
- > 2.0: 交感神経優位

{coherence_section}

## 時系列データ（3分ごと）

{metrics_table}

**指標の説明:**
- **RSA Amplitude (HF Power)**: 呼吸性洞性不整脈の振幅（HF成分パワー、ms²）
- **DFA α1**: 短期スケールのゆらぎ指数（瞑想習熟度の指標）
- **DFA α2**: 長期スケールのゆらぎ指数
- **RMSSD**: 連続R-R間隔差の二乗平均平方根（副交感神経活動）
- **SDNN**: R-R間隔の標準偏差（全体的なHRV）
- **LF Power**: 低周波パワー（0.04-0.15 Hz）
- **HF Power**: 高周波パワー（0.15-0.4 Hz、RSAを反映）
- **LF/HF**: 自律神経バランス指標

## 瞑想習熟度の評価

### 総合評価

1. **DFA指数**: {mean_dfa1:.3f} → {dfa_interpretation}
2. **RSA振幅**: {mean_rsa:.1f} ms² → {'高い' if mean_rsa > 500 else '中程度' if mean_rsa > 200 else '低い'}副交感神経活動
3. **LF/HF比**: {mean_lf_hf:.2f} → {'副交感神経優位' if mean_lf_hf < 1.0 else 'バランス' if mean_lf_hf < 2.0 else '交感神経優位'}

### RSAと瞑想の関係

論文「Heart rate dynamics in different levels of Zen meditation」によると、
経験豊富な坐禅実践者は以下の特徴を示します：

1. **低いDFA指数（~0.5）**: 睡眠の境界に近い深い瞑想状態
2. **高いHF成分**: 強化された副交感神経活動
3. **強化されたRSA**: 呼吸と心拍の高度な同期

あなたのデータはこれらの基準と比較して評価できます。

## 可視化

![RSA分析結果](rsa_analysis.png)

## 参考文献

1. Peressutti, C., Martín-González, J. M., García-Manso, J. M., & Mesa, D. (2010).
   Heart rate dynamics in different levels of Zen meditation.
   International Journal of Cardiology, 145(1), 142-146.

2. Task Force of the European Society of Cardiology and the North American Society
   of Pacing and Electrophysiology (1996). Heart rate variability: standards of
   measurement, physiological interpretation and clinical use. Circulation, 93(5), 1043-1065.

---

生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    report_path = output_dir / 'RSA_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'✓ レポート保存: {report_path}')


def main():
    """メイン実行関数"""
    print('='*70)
    print('呼吸性洞性不整脈（RSA）分析')
    print('='*70)
    print()

    # 最新のSelfloopsデータを使用
    csv_path = project_root / 'data/selfloops/selfloops_2026-01-12--06-21-05.csv'
    output_dir = Path(__file__).parent

    # 1. データ読み込み
    print(f'データ読み込み: {csv_path.name}')
    df = load_selfloops_csv(str(csv_path), warmup_seconds=0)
    hrv_data = get_hrv_data(df, use_device_hr=False, clean_artifacts=True)
    print(f'測定時間: {hrv_data["time"][-1]/60:.2f} 分')
    print()

    # 2. 呼吸分析（既存ライブラリ活用）
    print('呼吸数を計算中...')
    breathing_result = calculate_breathing_rate(
        hrv_data,
        target_fs=8.0,
        peak_distance=8.0,
        window_minutes=3.0
    )
    print(f'平均呼吸数: {breathing_result.breathing_rate:.1f} ± {breathing_result.breathing_rate_std:.1f} bpm')
    print()

    # 3. RSA時系列分析
    print('RSA指標の時系列変化を計算中...')
    rsa_metrics_df = analyze_rsa_time_series(hrv_data, window_minutes=3)
    print(f'ウィンドウ数: {len(rsa_metrics_df)}')
    print()

    # 4. 呼吸-心拍コヒーレンス分析
    print('呼吸-心拍コヒーレンスを計算中...')
    try:
        # リサンプリング済み信号を使用
        rr_intervals = hrv_data['rr_intervals_clean']
        hr_signal = 60000.0 / rr_intervals
        time_original = np.cumsum(rr_intervals) / 1000

        target_fs = 8.0
        time_resampled = np.arange(0, time_original[-1], 1.0/target_fs)

        f = interpolate.interp1d(time_original, hr_signal, kind='cubic', fill_value='extrapolate')
        hr_resampled = f(time_resampled)

        # EDR信号
        edr_signal = nk.ecg_rsp(hr_resampled, sampling_rate=int(target_fs), method='vangent2019')

        coherence_result = calculate_respiratory_heart_coherence(hr_resampled, edr_signal, fs=target_fs)
        print(f'最大コヒーレンス: {coherence_result["peak_coherence"]:.3f} @ {coherence_result["peak_freq"]:.3f} Hz')
    except Exception as e:
        print(f'コヒーレンス計算エラー: {e}')
        coherence_result = None
    print()

    # 5. 可視化
    print('プロット生成中...')
    plot_rsa_analysis(hrv_data, rsa_metrics_df, breathing_result, coherence_result, output_dir)
    print()

    # 6. レポート生成
    print('レポート生成中...')
    generate_rsa_report(hrv_data, rsa_metrics_df, breathing_result, coherence_result, output_dir)
    print()

    print('='*70)
    print('RSA分析完了!')
    print('='*70)


if __name__ == '__main__':
    main()
