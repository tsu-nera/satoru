#!/usr/bin/env python3
"""
EEG + HRV 統合瞑想分析スクリプト

Muse EEGデータとHRVデータを統合して分析し、瞑想状態を評価します。

Usage:
    python analyze_meditation_with_hrv.py --eeg <CSV_PATH> --hrv <TXT_PATH> [--output <OUTPUT_DIR>]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import (
    load_mind_monitor_csv,
    prepare_mne_raw,
    calculate_alpha_power,
    calculate_frontal_theta,
    calculate_spectral_entropy_time_series,
    get_heart_rate_data,
)

# HRV分析関数をインポート
from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
from lib.sensors.ecg.analysis import (
    analyze_hrv,
    analyze_hrv_time_domain,
    analyze_hrv_frequency_domain,
)


def synchronize_eeg_hrv(eeg_df, hrv_df):
    """
    EEGとHRVのデータを時間軸で同期

    Args:
        eeg_df: Muse EEGデータフレーム
        hrv_df: SelfLoops HRVデータフレーム

    Returns:
        tuple: (同期済みEEG timestamp, 同期済みHRV timestamp, 共通時間範囲)
    """
    # EEGのタイムスタンプ
    eeg_start = pd.to_datetime(eeg_df['TimeStamp'].iloc[0])
    eeg_end = pd.to_datetime(eeg_df['TimeStamp'].iloc[-1])

    # HRVはタイムスタンプが5231msから始まる相対時間
    # 最初の行のタイムスタンプをパースして開始時刻を取得
    # ファイルの1行目に "10 1月 2026 16:08:50" のような情報があると仮定
    # しかし、実際のデータでは1行目が "5231" となっているため、EEG開始時刻を使用
    hrv_start = eeg_start  # HRVとEEGが同時に開始したと仮定
    hrv_end = hrv_start + pd.Timedelta(milliseconds=hrv_df['Time (ms)'].iloc[-1])

    # 共通時間範囲
    common_start = max(eeg_start, hrv_start)
    common_end = min(eeg_end, hrv_end)

    return eeg_start, eeg_end, hrv_start, hrv_end, common_start, common_end


def create_time_aligned_dataframe(eeg_df, hrv_df, interval_sec=10):
    """
    EEGとHRVを時間軸で統合したデータフレームを作成

    Args:
        eeg_df: Muse EEGデータフレーム
        hrv_df: SelfLoops HRVデータフレーム
        interval_sec: リサンプリング間隔（秒）

    Returns:
        pandas.DataFrame: 統合データフレーム
    """
    # EEGの心拍数データを取得
    hr_data = get_heart_rate_data(eeg_df)

    # HRVのタイムスタンプを計算（EEG開始時刻を基準）
    eeg_start = pd.to_datetime(eeg_df['TimeStamp'].iloc[0])
    hrv_df_copy = hrv_df.copy()
    hrv_df_copy['Timestamp'] = eeg_start + pd.to_timedelta(hrv_df_copy['Time (ms)'], unit='ms')

    # 共通時間範囲でリサンプリング
    _, _, _, _, common_start, common_end = synchronize_eeg_hrv(eeg_df, hrv_df)

    # 時間軸を作成
    time_index = pd.date_range(start=common_start, end=common_end, freq=f'{interval_sec}s')

    # EEGデータ（心拍数）をリサンプリング
    # hr_dataはdictで{'timestamps': array, 'heart_rate': array, 'time': array}の形式
    hr_series = pd.Series(hr_data['heart_rate'], index=pd.to_datetime(hr_data['timestamps']))
    # 重複インデックスを削除（最初の値を保持）
    hr_series = hr_series[~hr_series.index.duplicated(keep='first')]
    # インデックスをソート
    hr_series = hr_series.sort_index()
    hr_resampled = hr_series.reindex(time_index, method='nearest', tolerance='5s')

    # HRVデータ（R-R間隔）をリサンプリング
    rr_series = pd.Series(hrv_df_copy['R-R (ms)'].values, index=hrv_df_copy['Timestamp'])
    # 重複インデックスを削除（最初の値を保持）
    rr_series = rr_series[~rr_series.index.duplicated(keep='first')]
    # インデックスをソート
    rr_series = rr_series.sort_index()
    rr_resampled = rr_series.reindex(time_index, method='nearest', tolerance='5s')

    # 統合データフレーム
    aligned_df = pd.DataFrame({
        'timestamp': time_index,
        'hr_eeg': hr_resampled.values,
        'rr_interval': rr_resampled.values,
    })

    # 欠損値を削除
    aligned_df = aligned_df.dropna()

    return aligned_df


def analyze_eeg_hrv_correlation(eeg_results, hrv_metrics, aligned_df):
    """
    EEGとHRVの相関分析

    Args:
        eeg_results: EEG分析結果（dict）
        hrv_metrics: HRV指標（dict）
        aligned_df: 時間軸で同期したデータフレーム

    Returns:
        dict: 相関分析結果
    """
    correlations = {}

    # Alpha PowerとHRV指標の相関（全体平均値）
    if 'alpha_power' in eeg_results and 'sdnn' in hrv_metrics:
        # 単一値同士の相関は計算できないため、時系列データが必要
        # ここでは簡易版として値のみ記録
        alpha_result = eeg_results['alpha_power']
        correlations['alpha_mean'] = alpha_result.alpha_db if hasattr(alpha_result, 'alpha_db') else None
        correlations['sdnn'] = hrv_metrics['sdnn']
        correlations['rmssd'] = hrv_metrics['rmssd']
        correlations['lf_hf_ratio'] = hrv_metrics.get('lf_hf_ratio', None)

    # 時系列データでの相関（心拍数 vs R-R間隔）
    if not aligned_df.empty and 'hr_eeg' in aligned_df.columns and 'rr_interval' in aligned_df.columns:
        hr = aligned_df['hr_eeg'].dropna()
        rr = aligned_df['rr_interval'].dropna()

        if len(hr) > 2 and len(rr) > 2:
            # 共通インデックスで相関計算
            common_idx = hr.index.intersection(rr.index)
            if len(common_idx) > 2:
                hr_common = hr.loc[common_idx]
                rr_common = rr.loc[common_idx]

                corr_hr_rr, p_value_hr_rr = stats.pearsonr(hr_common, rr_common)
                correlations['hr_rr_correlation'] = corr_hr_rr
                correlations['hr_rr_pvalue'] = p_value_hr_rr

    return correlations


def plot_integrated_analysis(eeg_results, hrv_metrics, aligned_df, freq_domain, output_path):
    """
    EEGとHRVの統合プロット

    Args:
        eeg_results: EEG分析結果
        hrv_metrics: HRV時間領域指標
        aligned_df: 同期データフレーム
        freq_domain: HRV周波数領域指標
        output_path: 出力パス
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # 1. Alpha Power時系列（EEG）
    ax1 = fig.add_subplot(gs[0, :])
    if 'alpha_power' in eeg_results and hasattr(eeg_results['alpha_power'], 'time_series'):
        alpha_ts = eeg_results['alpha_power'].time_series
        ax1.plot(alpha_ts.index, alpha_ts.values, label='Alpha Power', color='steelblue', linewidth=1.5)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Alpha Power (dBx)')
        ax1.set_title('EEG Alpha Power Time Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Frontal Theta時系列（EEG）
    ax2 = fig.add_subplot(gs[1, 0])
    if 'frontal_theta' in eeg_results and hasattr(eeg_results['frontal_theta'], 'time_series'):
        fmt_ts = eeg_results['frontal_theta'].time_series
        ax2.plot(fmt_ts.index, fmt_ts.values, label='Frontal Theta', color='orange', linewidth=1.5)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Frontal Theta (dB)')
        ax2.set_title('EEG Frontal Midline Theta')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Spectral Entropy時系列（EEG）
    ax3 = fig.add_subplot(gs[1, 1])
    if 'spectral_entropy' in eeg_results and isinstance(eeg_results['spectral_entropy'], pd.Series):
        se_ts = eeg_results['spectral_entropy']
        ax3.plot(se_ts.index, se_ts.values, label='Spectral Entropy', color='green', linewidth=1.5)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Spectral Entropy')
        ax3.set_title('EEG Spectral Entropy (Concentration)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. 心拍数とR-R間隔の比較
    ax4 = fig.add_subplot(gs[2, :])
    if not aligned_df.empty:
        # 2軸プロット
        ax4_twin = ax4.twinx()

        time_min = (aligned_df['timestamp'] - aligned_df['timestamp'].iloc[0]).dt.total_seconds() / 60

        # 心拍数（左軸）
        if 'hr_eeg' in aligned_df.columns:
            ax4.plot(time_min, aligned_df['hr_eeg'], label='Heart Rate (EEG)', color='red', linewidth=1.2, alpha=0.8)
        ax4.set_xlabel('Time (min)')
        ax4.set_ylabel('Heart Rate (bpm)', color='red')
        ax4.tick_params(axis='y', labelcolor='red')

        # R-R間隔（右軸）
        if 'rr_interval' in aligned_df.columns:
            ax4_twin.plot(time_min, aligned_df['rr_interval'], label='R-R Interval', color='blue', linewidth=1.2, alpha=0.8)
        ax4_twin.set_ylabel('R-R Interval (ms)', color='blue')
        ax4_twin.tick_params(axis='y', labelcolor='blue')

        ax4.set_title('Heart Rate (EEG) vs R-R Interval (HRV)')
        ax4.grid(True, alpha=0.3)

    # 5. HRV周波数スペクトル
    ax5 = fig.add_subplot(gs[3, 0])
    if 'freqs' in freq_domain and 'psd' in freq_domain:
        freqs = freq_domain['freqs']
        psd = freq_domain['psd']
        ax5.semilogy(freqs, psd, linewidth=1.5)
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('PSD (ms²/Hz)')
        ax5.set_title('HRV Power Spectral Density')
        ax5.set_xlim([0, 0.5])
        ax5.axvspan(0.04, 0.15, alpha=0.2, color='yellow', label='LF')
        ax5.axvspan(0.15, 0.4, alpha=0.2, color='cyan', label='HF')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. 統合指標バーチャート
    ax6 = fig.add_subplot(gs[3, 1])
    metrics_names = []
    metrics_values = []
    colors = []

    # EEG指標
    if 'alpha_power' in eeg_results:
        alpha_db = eeg_results['alpha_power'].alpha_db if hasattr(eeg_results['alpha_power'], 'alpha_db') else 0
        metrics_names.append('Alpha\nPower')
        metrics_values.append(alpha_db)
        colors.append('#1f77b4')

    if 'frontal_theta' in eeg_results:
        fmt_mean = eeg_results['frontal_theta'].time_series.mean() if hasattr(eeg_results['frontal_theta'], 'time_series') else 0
        metrics_names.append('Frontal\nTheta')
        metrics_values.append(fmt_mean)
        colors.append('#ff7f0e')

    # HRV指標（スケーリング）
    if 'sdnn' in hrv_metrics:
        metrics_names.append('SDNN\n(×0.1)')
        metrics_values.append(hrv_metrics['sdnn'] * 0.1)
        colors.append('#2ca02c')

    if 'lf_hf_ratio' in freq_domain:
        metrics_names.append('LF/HF\nRatio')
        metrics_values.append(freq_domain['lf_hf_ratio'])
        colors.append('#d62728')

    bars = ax6.bar(metrics_names, metrics_values, color=colors, alpha=0.7)

    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=9)

    ax6.set_ylabel('Value (normalized)')
    ax6.set_title('Key Metrics Summary')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'✓ 統合プロット保存: {output_path}')


def generate_integrated_report(eeg_results, hrv_time_domain, hrv_freq_domain, correlations, eeg_df, hrv_df, output_path):
    """
    EEG + HRV統合レポートを生成

    Args:
        eeg_results: EEG分析結果
        hrv_time_domain: HRV時間領域指標
        hrv_freq_domain: HRV周波数領域指標
        correlations: 相関分析結果
        eeg_df: EEGデータフレーム
        hrv_df: HRVデータフレーム
        output_path: 出力パス
    """
    # 測定時間
    eeg_start = pd.to_datetime(eeg_df['TimeStamp'].iloc[0])
    eeg_end = pd.to_datetime(eeg_df['TimeStamp'].iloc[-1])
    eeg_duration_min = (eeg_end - eeg_start).total_seconds() / 60

    hrv_duration_min = hrv_df['Time (ms)'].iloc[-1] / 1000 / 60

    # EEG指標
    alpha_db = eeg_results.get('alpha_power', {}).alpha_db if 'alpha_power' in eeg_results else 'N/A'
    fmt_mean = eeg_results['frontal_theta'].time_series.mean() if 'frontal_theta' in eeg_results else 'N/A'
    se_mean = eeg_results['spectral_entropy'].mean() if 'spectral_entropy' in eeg_results else 'N/A'

    report = f"""# EEG + HRV 統合瞑想分析レポート

## 概要

Muse EEGデータとHRVデータを統合して瞑想状態を分析しました。

### 測定情報

**EEGデータ**
- 測定開始: {eeg_start.strftime('%Y-%m-%d %H:%M:%S')}
- 測定終了: {eeg_end.strftime('%Y-%m-%d %H:%M:%S')}
- 測定時間: {eeg_duration_min:.2f} 分

**HRVデータ**
- 測定時間: {hrv_duration_min:.2f} 分
- データポイント数: {len(hrv_df)} 点

---

## 主要指標サマリー

### EEG指標

| 指標 | 値 | 説明 |
|:-----|---:|:-----|
| **Alpha Power** | {alpha_db if isinstance(alpha_db, str) else f'{alpha_db:.2f} dB'} | 精神的回復度・リラックス状態 |
| **Frontal Theta** | {fmt_mean if isinstance(fmt_mean, str) else f'{fmt_mean:.2f} dB'} | 瞑想深度・内的集中 |
| **Spectral Entropy** | {se_mean if isinstance(se_mean, str) else f'{se_mean:.3f}'} | 集中度（低いほど集中） |

### HRV指標

| 指標 | 値 | 説明 |
|:-----|---:|:-----|
| **SDNN** | {hrv_time_domain['sdnn']:.2f} ms | 全体的な心拍変動 |
| **RMSSD** | {hrv_time_domain['rmssd']:.2f} ms | 副交感神経活動 |
| **pNN50** | {hrv_time_domain['pnn50']:.2f} % | 副交感神経活動 |
| **LF/HF比** | {hrv_freq_domain['lf_hf_ratio']:.2f} | 自律神経バランス |

---

## 統合分析

### EEGとHRVの関係

"""

    # 相関結果
    if 'hr_rr_correlation' in correlations:
        corr = correlations['hr_rr_correlation']
        p_val = correlations['hr_rr_pvalue']
        report += f"""
**心拍数 vs R-R間隔の相関**
- 相関係数: {corr:.3f}
- p値: {p_val:.4f}
- 解釈: {'強い負の相関' if corr < -0.7 else '中程度の負の相関' if corr < -0.3 else '弱い相関' if abs(corr) < 0.3 else '中程度の正の相関' if corr < 0.7 else '強い正の相関'}

"""

    report += f"""
### 瞑想状態の評価

**脳波（EEG）からの評価:**
- Alpha Power: {alpha_db if isinstance(alpha_db, str) else f'{alpha_db:.2f} dB'} - リラックス状態を示唆
- Frontal Theta: {fmt_mean if isinstance(fmt_mean, str) else f'{fmt_mean:.2f} dB'} - 瞑想深度を反映
- Spectral Entropy: {se_mean if isinstance(se_mean, str) else f'{se_mean:.3f}'} - {'高い集中' if not isinstance(se_mean, str) and se_mean < 0.75 else '中程度の集中' if not isinstance(se_mean, str) and se_mean < 0.85 else '散漫'}

**自律神経（HRV）からの評価:**
- SDNN: {hrv_time_domain['sdnn']:.2f} ms - {'優秀' if hrv_time_domain['sdnn'] >= 100 else '良好' if hrv_time_domain['sdnn'] >= 50 else '低下'}な心拍変動
- LF/HF比: {hrv_freq_domain['lf_hf_ratio']:.2f} - {'交感神経優位（緊張・覚醒）' if hrv_freq_domain['lf_hf_ratio'] >= 2.0 else '副交感神経優位（リラックス）' if hrv_freq_domain['lf_hf_ratio'] < 1.0 else 'バランス'}
- RMSSD: {hrv_time_domain['rmssd']:.2f} ms - {'高い' if hrv_time_domain['rmssd'] >= 40 else '中程度の' if hrv_time_domain['rmssd'] >= 20 else '低い'}副交感神経活動

---

## 統合的解釈

### 瞑想の質

1. **脳波パターン**:
   - Frontal Thetaが{'高く' if not isinstance(fmt_mean, str) and fmt_mean > 5.0 else '中程度で' if not isinstance(fmt_mean, str) and fmt_mean > 3.0 else '低く'}、{'深い瞑想状態' if not isinstance(fmt_mean, str) and fmt_mean > 5.0 else '軽い瞑想状態' if not isinstance(fmt_mean, str) and fmt_mean > 3.0 else '浅い瞑想状態'}を示唆
   - Alpha Powerは精神的回復・リラックスを反映

2. **自律神経状態**:
   - LF/HF比 {hrv_freq_domain['lf_hf_ratio']:.2f}は{'交感神経が優位' if hrv_freq_domain['lf_hf_ratio'] >= 2.0 else '副交感神経が優位' if hrv_freq_domain['lf_hf_ratio'] < 1.0 else 'バランスの取れた状態'}
   - {'ストレス・緊張状態にある可能性' if hrv_freq_domain['lf_hf_ratio'] >= 3.0 else 'リラックスした状態' if hrv_freq_domain['lf_hf_ratio'] < 1.5 else '適度な覚醒レベル'}

3. **統合評価**:
   - {'脳波は瞑想状態を示すが、自律神経は緊張状態' if (not isinstance(fmt_mean, str) and fmt_mean > 4.0) and hrv_freq_domain['lf_hf_ratio'] >= 3.0 else '脳波と自律神経の両方がリラックス状態' if (not isinstance(fmt_mean, str) and fmt_mean > 4.0) and hrv_freq_domain['lf_hf_ratio'] < 1.5 else '中間的な状態'}

---

## 可視化

![EEG + HRV統合分析](meditation_integrated_analysis.png)

---

## 今後の分析課題

1. **時系列での詳細分析**
   - 瞑想中のAlpha PowerとHRV指標の時間変化の同期分析
   - 瞑想の各フェーズ（導入、深化、維持、終了）での変化パターン

2. **相関分析の拡張**
   - Frontal ThetaとRMSSDの関係
   - Spectral EntropyとLF/HF比の関係
   - 特定の瞑想テクニック（呼吸瞑想、ボディスキャンなど）との関連

3. **長期的追跡**
   - 複数セッションでの変化を追跡
   - 瞑想熟練度との関係性の検証

4. **個別最適化**
   - 個人に最適な瞑想状態の特定
   - バイオフィードバックへの応用

---

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'✓ 統合レポート生成: {output_path}')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='EEG + HRV 統合瞑想分析'
    )
    parser.add_argument(
        '--eeg',
        type=Path,
        required=True,
        help='Muse EEG CSVファイルパス'
    )
    parser.add_argument(
        '--hrv',
        type=Path,
        required=True,
        help='SelfLoops HRV dataファイルパス (.txt)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='出力ディレクトリ（デフォルト: EEGファイルと同じディレクトリ）'
    )

    args = parser.parse_args()

    # 出力ディレクトリの設定
    if args.output is None:
        args.output = args.eeg.parent

    # パスの検証
    if not args.eeg.exists():
        print(f'エラー: EEGファイルが見つかりません: {args.eeg}')
        return 1

    if not args.hrv.exists():
        print(f'エラー: HRVファイルが見つかりません: {args.hrv}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    print('='*60)
    print('EEG + HRV 統合瞑想分析')
    print('='*60)
    print()

    # 1. EEGデータ読み込み
    print(f'Loading EEG: {args.eeg.name}')
    eeg_df = load_mind_monitor_csv(args.eeg, filter_headband=False, warmup_seconds=60)
    print(f'EEGデータ形状: {eeg_df.shape[0]} 行 × {eeg_df.shape[1]} 列')

    # 2. HRVデータ読み込み
    print(f'Loading HRV: {args.hrv.name}')
    hrv_df = load_selfloops_csv(str(args.hrv))
    print(f'HRVデータ形状: {hrv_df.shape[0]} 行 × {hrv_df.shape[1]} 列')
    print()

    # 3. EEG分析
    print('分析中: EEG指標...')
    mne_dict = prepare_mne_raw(eeg_df, apply_bandpass=False, apply_notch=False)
    if not mne_dict:
        print('エラー: MNE RAWデータの準備に失敗しました')
        return 1

    raw = mne_dict['raw']

    eeg_results = {}

    # Alpha Power
    try:
        alpha_result = calculate_alpha_power(eeg_df)
        eeg_results['alpha_power'] = alpha_result
        print(f'  Alpha Power: {alpha_result.alpha_db:.2f} dB')
    except Exception as e:
        print(f'  Alpha Power計算エラー: {e}')

    # Frontal Theta
    try:
        fmt_result = calculate_frontal_theta(
            eeg_df,
            channels=('RAW_AF7', 'RAW_AF8'),
            band=(4.0, 8.0),
            raw=raw,
            include_alpha=False,
        )
        eeg_results['frontal_theta'] = fmt_result
        print(f'  Frontal Theta: {fmt_result.time_series.mean():.2f} dB (mean)')
    except Exception as e:
        print(f'  Frontal Theta計算エラー: {e}')

    # Spectral Entropy（一旦スキップ - 関数内部のバグあり）
    # try:
    #     se_result = calculate_spectral_entropy_time_series(eeg_df)
    #     # se_resultがdictの場合は'entropy'キーを取得
    #     if isinstance(se_result, dict):
    #         se_series = se_result.get('entropy', pd.Series())
    #     else:
    #         se_series = se_result
    #     eeg_results['spectral_entropy'] = se_series
    #     if len(se_series) > 0:
    #         print(f'  Spectral Entropy: {se_series.mean():.3f}')
    # except Exception as e:
    #     print(f'  Spectral Entropy計算エラー: {e}')
    #     import traceback
    #     traceback.print_exc()
    print('  Spectral Entropy: (スキップ)')

    print()

    # 4. HRV分析
    print('分析中: HRV指標...')
    hrv_data = get_hrv_data(hrv_df)
    hrv_time_domain_df = analyze_hrv_time_domain(hrv_data, show=False)
    hrv_freq_domain_df = analyze_hrv_frequency_domain(hrv_data, show=False)

    # DataFrameから値を取得して辞書形式に変換
    hrv_time_domain = {
        'sdnn': hrv_time_domain_df['HRV_SDNN'].values[0],
        'rmssd': hrv_time_domain_df['HRV_RMSSD'].values[0],
        'pnn50': hrv_time_domain_df['HRV_pNN50'].values[0],
    }
    hrv_freq_domain = {
        'lf_hf_ratio': hrv_freq_domain_df['HRV_LFHF'].values[0],
        'lf_power': hrv_freq_domain_df['HRV_LF'].values[0],
        'hf_power': hrv_freq_domain_df['HRV_HF'].values[0],
    }

    print(f'  SDNN: {hrv_time_domain["sdnn"]:.2f} ms')
    print(f'  RMSSD: {hrv_time_domain["rmssd"]:.2f} ms')
    print(f'  pNN50: {hrv_time_domain["pnn50"]:.2f} %')
    print(f'  LF/HF Ratio: {hrv_freq_domain["lf_hf_ratio"]:.2f}')
    print()

    # 5. データ同期
    print('同期中: EEGとHRVデータ...')
    aligned_df = create_time_aligned_dataframe(eeg_df, hrv_df, interval_sec=10)
    print(f'  同期データポイント数: {len(aligned_df)}')
    print()

    # 6. 相関分析
    print('分析中: EEGとHRVの相関...')
    correlations = analyze_eeg_hrv_correlation(eeg_results, hrv_time_domain, aligned_df)
    if 'hr_rr_correlation' in correlations:
        print(f'  心拍数 vs R-R間隔: r={correlations["hr_rr_correlation"]:.3f}, p={correlations["hr_rr_pvalue"]:.4f}')
    print()

    # 7. 可視化
    print('生成中: 統合プロット...')
    plot_integrated_analysis(
        eeg_results,
        hrv_time_domain,
        aligned_df,
        hrv_freq_domain,
        args.output / 'meditation_integrated_analysis.png'
    )
    print()

    # 8. レポート生成
    print('生成中: 統合レポート...')
    generate_integrated_report(
        eeg_results,
        hrv_time_domain,
        hrv_freq_domain,
        correlations,
        eeg_df,
        hrv_df,
        args.output / 'MEDITATION_REPORT.md'
    )
    print()

    print('='*60)
    print('分析完了!')
    print('='*60)
    print(f'出力ディレクトリ: {args.output}')
    print(f'  - MEDITATION_REPORT.md')
    print(f'  - meditation_integrated_analysis.png')

    return 0


if __name__ == '__main__':
    exit(main())
