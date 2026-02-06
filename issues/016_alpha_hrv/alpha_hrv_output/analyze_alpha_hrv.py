#!/usr/bin/env python3
"""
Alpha波とHRVの関係分析スクリプト

Muse EEGのAlpha波データとSelfLoops HRVデータを統合して、
リラックス状態の相関関係を分析します。

Usage:
    python analyze_alpha_hrv.py --eeg <CSV_PATH> --hrv <CSV_PATH> [--output <OUTPUT_DIR>]
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
from scipy import stats

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import load_mind_monitor_csv
from lib.sensors.eeg.alpha_power import calculate_alpha_power
from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
from lib.sensors.ecg.hrv import calculate_hrv_standard_set


def create_aligned_timeseries(alpha_result, hrv_result, eeg_start, hrv_start, resample_interval='30s'):
    """
    Alpha波とHRVの時系列データを同期

    Parameters
    ----------
    alpha_result : AlphaPowerResult
        Alpha波解析結果
    hrv_result : HRVResult
        HRV解析結果
    eeg_start : datetime
        EEG開始時刻
    hrv_start : datetime
        HRV開始時刻
    resample_interval : str
        リサンプリング間隔

    Returns
    -------
    pd.DataFrame
        同期された時系列データ
    """
    # Alpha波時系列
    alpha_ts = alpha_result.time_series.copy()

    # HRV時系列（RMSSD, LF/HF）
    rmssd_ts = hrv_result.time_series['rmssd'].copy()
    lfhf_ts = hrv_result.time_series['lfhf_ratio'].copy()

    # 時間オフセット調整（HRVの開始時刻をEEGに合わせる）
    time_offset = eeg_start - hrv_start
    rmssd_ts.index = rmssd_ts.index + time_offset
    lfhf_ts.index = lfhf_ts.index + time_offset

    # リサンプリング
    alpha_resampled = alpha_ts.resample(resample_interval).mean()
    rmssd_resampled = rmssd_ts.resample(resample_interval).mean()
    lfhf_resampled = lfhf_ts.resample(resample_interval).mean()

    # 共通時間範囲
    common_start = max(alpha_resampled.index.min(), rmssd_resampled.index.min())
    common_end = min(alpha_resampled.index.max(), rmssd_resampled.index.max())

    # 時間範囲でフィルタ
    alpha_filtered = alpha_resampled[(alpha_resampled.index >= common_start) & (alpha_resampled.index <= common_end)]
    rmssd_filtered = rmssd_resampled[(rmssd_resampled.index >= common_start) & (rmssd_resampled.index <= common_end)]
    lfhf_filtered = lfhf_resampled[(lfhf_resampled.index >= common_start) & (lfhf_resampled.index <= common_end)]

    # 統合
    aligned_df = pd.DataFrame({
        'timestamp': alpha_filtered.index,
        'alpha_power': alpha_filtered.values,
    }).set_index('timestamp')

    # インデックスを統一してマージ
    aligned_df['rmssd'] = rmssd_filtered.reindex(aligned_df.index, method='nearest', tolerance='30s')
    aligned_df['lfhf_ratio'] = lfhf_filtered.reindex(aligned_df.index, method='nearest', tolerance='30s')

    # 欠損値を含む行を削除
    aligned_df = aligned_df.dropna()

    return aligned_df


def calculate_correlations(aligned_df):
    """
    Alpha波とHRV指標の相関を計算

    Parameters
    ----------
    aligned_df : pd.DataFrame
        同期された時系列データ

    Returns
    -------
    dict
        相関分析結果
    """
    correlations = {}

    if len(aligned_df) < 5:
        return correlations

    # Alpha vs RMSSD
    if 'alpha_power' in aligned_df.columns and 'rmssd' in aligned_df.columns:
        alpha = aligned_df['alpha_power'].dropna()
        rmssd = aligned_df['rmssd'].dropna()
        common_idx = alpha.index.intersection(rmssd.index)

        if len(common_idx) >= 5:
            r, p = stats.pearsonr(alpha.loc[common_idx], rmssd.loc[common_idx])
            correlations['alpha_rmssd'] = {'r': r, 'p': p, 'n': len(common_idx)}

    # Alpha vs LF/HF
    if 'alpha_power' in aligned_df.columns and 'lfhf_ratio' in aligned_df.columns:
        alpha = aligned_df['alpha_power'].dropna()
        lfhf = aligned_df['lfhf_ratio'].dropna()
        common_idx = alpha.index.intersection(lfhf.index)

        if len(common_idx) >= 5:
            r, p = stats.pearsonr(alpha.loc[common_idx], lfhf.loc[common_idx])
            correlations['alpha_lfhf'] = {'r': r, 'p': p, 'n': len(common_idx)}

    return correlations


def plot_alpha_hrv_analysis(alpha_result, hrv_result, aligned_df, correlations, output_path):
    """
    Alpha波とHRVの統合プロット

    Parameters
    ----------
    alpha_result : AlphaPowerResult
        Alpha波解析結果
    hrv_result : HRVResult
        HRV解析結果
    aligned_df : pd.DataFrame
        同期された時系列データ
    correlations : dict
        相関分析結果
    output_path : Path
        出力パス
    """
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # 1. Alpha Power時系列
    ax1 = fig.add_subplot(gs[0, :])
    alpha_ts = alpha_result.time_series
    time_min = (alpha_ts.index - alpha_ts.index[0]).total_seconds() / 60
    ax1.plot(time_min, alpha_ts.values, label='Alpha Power', color='steelblue', linewidth=1.5)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Alpha Power (dBx)')
    ax1.set_title('EEG Alpha Power Time Series (Relaxation Indicator)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=alpha_ts.mean(), color='steelblue', linestyle='--', alpha=0.5, label=f'Mean: {alpha_ts.mean():.1f}')

    # 2. HRV RMSSD時系列
    ax2 = fig.add_subplot(gs[1, 0])
    rmssd_ts = hrv_result.time_series['rmssd']
    time_min_hrv = (rmssd_ts.index - rmssd_ts.index[0]).total_seconds() / 60
    ax2.plot(time_min_hrv, rmssd_ts.values, label='RMSSD', color='green', linewidth=1.5)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('RMSSD (ms)')
    ax2.set_title('HRV RMSSD Time Series (Parasympathetic Activity)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=rmssd_ts.mean(), color='green', linestyle='--', alpha=0.5)

    # 3. HRV LF/HF時系列
    ax3 = fig.add_subplot(gs[1, 1])
    lfhf_ts = hrv_result.time_series['lfhf_ratio']
    time_min_lfhf = (lfhf_ts.index - lfhf_ts.index[0]).total_seconds() / 60
    ax3.plot(time_min_lfhf, lfhf_ts.values, label='LF/HF Ratio', color='orange', linewidth=1.5)
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('LF/HF Ratio')
    ax3.set_title('HRV LF/HF Ratio Time Series (Autonomic Balance)')
    ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Balance Line (1.0)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Alpha vs RMSSD散布図
    ax4 = fig.add_subplot(gs[2, 0])
    if not aligned_df.empty and 'alpha_power' in aligned_df.columns and 'rmssd' in aligned_df.columns:
        valid_data = aligned_df[['alpha_power', 'rmssd']].dropna()
        if len(valid_data) > 0:
            ax4.scatter(valid_data['alpha_power'], valid_data['rmssd'],
                       alpha=0.6, edgecolors='k', linewidths=0.5, c='steelblue')

            # 回帰直線
            if len(valid_data) >= 5:
                z = np.polyfit(valid_data['alpha_power'], valid_data['rmssd'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_data['alpha_power'].min(), valid_data['alpha_power'].max(), 100)
                ax4.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)

                if 'alpha_rmssd' in correlations:
                    corr = correlations['alpha_rmssd']
                    ax4.annotate(f"r = {corr['r']:.3f}\np = {corr['p']:.4f}\nn = {corr['n']}",
                                xy=(0.05, 0.95), xycoords='axes fraction',
                                fontsize=10, va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax4.set_xlabel('Alpha Power (dBx)')
    ax4.set_ylabel('RMSSD (ms)')
    ax4.set_title('Alpha Power vs RMSSD Correlation')
    ax4.grid(True, alpha=0.3)

    # 5. Alpha vs LF/HF散布図
    ax5 = fig.add_subplot(gs[2, 1])
    if not aligned_df.empty and 'alpha_power' in aligned_df.columns and 'lfhf_ratio' in aligned_df.columns:
        valid_data = aligned_df[['alpha_power', 'lfhf_ratio']].dropna()
        if len(valid_data) > 0:
            ax5.scatter(valid_data['alpha_power'], valid_data['lfhf_ratio'],
                       alpha=0.6, edgecolors='k', linewidths=0.5, c='orange')

            # 回帰直線
            if len(valid_data) >= 5:
                z = np.polyfit(valid_data['alpha_power'], valid_data['lfhf_ratio'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_data['alpha_power'].min(), valid_data['alpha_power'].max(), 100)
                ax5.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)

                if 'alpha_lfhf' in correlations:
                    corr = correlations['alpha_lfhf']
                    ax5.annotate(f"r = {corr['r']:.3f}\np = {corr['p']:.4f}\nn = {corr['n']}",
                                xy=(0.05, 0.95), xycoords='axes fraction',
                                fontsize=10, va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax5.set_xlabel('Alpha Power (dBx)')
    ax5.set_ylabel('LF/HF Ratio')
    ax5.set_title('Alpha Power vs LF/HF Ratio Correlation')
    ax5.grid(True, alpha=0.3)

    # 6. 統合指標サマリー
    ax6 = fig.add_subplot(gs[3, :])

    # テーブルデータ作成
    table_data = [
        ['Alpha Power (Mean)', f'{alpha_result.alpha_db:.2f} dB', 'Brain relaxation indicator'],
        ['Alpha Power Score', f'{alpha_result.score:.1f} dBx', 'Muse-equivalent score'],
        ['RMSSD (Mean)', f'{hrv_result.metadata["mean_rmssd"]:.2f} ms', 'Parasympathetic activity'],
        ['LF/HF Ratio (Mean)', f'{hrv_result.metadata["mean_lfhf"]:.2f}', 'Autonomic balance'],
    ]

    # 相関結果を追加
    if 'alpha_rmssd' in correlations:
        corr = correlations['alpha_rmssd']
        sig = '***' if corr['p'] < 0.001 else '**' if corr['p'] < 0.01 else '*' if corr['p'] < 0.05 else ''
        table_data.append(['Alpha-RMSSD Correlation', f"r = {corr['r']:.3f}{sig}",
                          'Positive = relaxation coherence'])

    if 'alpha_lfhf' in correlations:
        corr = correlations['alpha_lfhf']
        sig = '***' if corr['p'] < 0.001 else '**' if corr['p'] < 0.01 else '*' if corr['p'] < 0.05 else ''
        table_data.append(['Alpha-LF/HF Correlation', f"r = {corr['r']:.3f}{sig}",
                          'Negative = more relaxed'])

    ax6.axis('off')
    table = ax6.table(cellText=table_data,
                     colLabels=['Metric', 'Value', 'Interpretation'],
                     loc='center',
                     cellLoc='left',
                     colWidths=[0.3, 0.25, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # ヘッダースタイル
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax6.set_title('Summary: Alpha Wave and HRV Relationship', fontsize=12, fontweight='bold', pad=20)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Analysis plot saved: {output_path}')


def generate_report(alpha_result, hrv_result, aligned_df, correlations, eeg_duration, hrv_duration, output_path):
    """
    Alpha波とHRVの関係分析レポートを生成

    Parameters
    ----------
    alpha_result : AlphaPowerResult
        Alpha波解析結果
    hrv_result : HRVResult
        HRV解析結果
    aligned_df : pd.DataFrame
        同期された時系列データ
    correlations : dict
        相関分析結果
    eeg_duration : float
        EEG測定時間（分）
    hrv_duration : float
        HRV測定時間（分）
    output_path : Path
        出力パス
    """
    # 相関の解釈
    alpha_rmssd_interp = ""
    alpha_lfhf_interp = ""

    if 'alpha_rmssd' in correlations:
        r = correlations['alpha_rmssd']['r']
        p = correlations['alpha_rmssd']['p']
        if p < 0.05:
            if r > 0.5:
                alpha_rmssd_interp = "強い正の相関: Alpha波とRMSSDが同期して増加。脳と心臓の両方がリラックス状態を示している。"
            elif r > 0.3:
                alpha_rmssd_interp = "中程度の正の相関: Alpha波の増加とともに副交感神経活動も増加する傾向。"
            elif r > 0:
                alpha_rmssd_interp = "弱い正の相関: Alpha波とRMSSDに若干の正の関係が見られる。"
            elif r > -0.3:
                alpha_rmssd_interp = "弱い負の相関: Alpha波とRMSSDの関係は明確ではない。"
            else:
                alpha_rmssd_interp = "負の相関: Alpha波とRMSSDが逆方向に変動。脳のリラックスと心臓の状態が一致していない可能性。"
        else:
            alpha_rmssd_interp = "統計的に有意な相関は見られなかった。"

    if 'alpha_lfhf' in correlations:
        r = correlations['alpha_lfhf']['r']
        p = correlations['alpha_lfhf']['p']
        if p < 0.05:
            if r < -0.5:
                alpha_lfhf_interp = "強い負の相関: Alpha波が高いときLF/HFが低い。脳のリラックスと副交感神経優位が連動している理想的な状態。"
            elif r < -0.3:
                alpha_lfhf_interp = "中程度の負の相関: Alpha波の増加とともに副交感神経がやや優位になる傾向。"
            elif r < 0:
                alpha_lfhf_interp = "弱い負の相関: Alpha波とLF/HFに若干の負の関係が見られる。"
            elif r < 0.3:
                alpha_lfhf_interp = "弱い正の相関: Alpha波とLF/HFの関係は期待と異なる可能性。"
            else:
                alpha_lfhf_interp = "正の相関: Alpha波が高いときにも交感神経が活性化。精神的には落ち着いているが身体は緊張状態の可能性。"
        else:
            alpha_lfhf_interp = "統計的に有意な相関は見られなかった。"

    # RMSSD評価
    mean_rmssd = hrv_result.metadata["mean_rmssd"]
    if mean_rmssd >= 50:
        rmssd_eval = "高い（リラックス状態）"
    elif mean_rmssd >= 30:
        rmssd_eval = "標準"
    else:
        rmssd_eval = "低い（ストレス状態の可能性）"

    # LF/HF評価
    mean_lfhf = hrv_result.metadata["mean_lfhf"]
    if mean_lfhf < 1.0:
        lfhf_eval = "副交感神経優位（リラックス）"
    elif mean_lfhf <= 2.0:
        lfhf_eval = "バランス状態"
    else:
        lfhf_eval = "交感神経優位（緊張・覚醒）"

    report = f"""# Alpha波とHRVの関係分析レポート

## 概要

このレポートでは、脳波（EEG）のAlpha波と心拍変動（HRV）の関係を分析し、
リラックス状態の生理学的指標の関連性を検証します。

**分析日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 測定データ

| 項目 | 値 |
|:-----|---:|
| EEG測定時間 | {eeg_duration:.1f} 分 |
| HRV測定時間 | {hrv_duration:.1f} 分 |
| 同期データポイント数 | {len(aligned_df)} 点 |

---

## Alpha波（脳のリラックス指標）

Alpha波（8-13Hz）は、リラックスした覚醒状態で増加する脳波成分です。

| 指標 | 値 | 評価 |
|:-----|---:|:-----|
| Alpha Power (平均) | {alpha_result.alpha_db:.2f} dB | 精神的リラックス度 |
| Alpha Score | {alpha_result.score:.1f} dBx | Muse相当スコア |
| 最小値 | {alpha_result.time_series.min():.1f} dBx | - |
| 最大値 | {alpha_result.time_series.max():.1f} dBx | - |
| 標準偏差 | {alpha_result.time_series.std():.1f} dBx | 変動の大きさ |

---

## HRV（心臓のリラックス指標）

HRVは心拍間隔の変動を測定し、自律神経系の状態を反映します。

### 時間領域指標

| 指標 | 値 | 評価 |
|:-----|---:|:-----|
| RMSSD (平均) | {mean_rmssd:.2f} ms | {rmssd_eval} |

### 周波数領域指標

| 指標 | 値 | 評価 |
|:-----|---:|:-----|
| LF/HF Ratio (平均) | {mean_lfhf:.2f} | {lfhf_eval} |

---

## Alpha波とHRVの相関分析

### Alpha Power vs RMSSD

"""
    if 'alpha_rmssd' in correlations:
        corr = correlations['alpha_rmssd']
        sig = '***' if corr['p'] < 0.001 else '**' if corr['p'] < 0.01 else '*' if corr['p'] < 0.05 else 'ns'
        report += f"""| 指標 | 値 |
|:-----|---:|
| 相関係数 (r) | {corr['r']:.3f} |
| p値 | {corr['p']:.4f} ({sig}) |
| サンプル数 | {corr['n']} |

**解釈**: {alpha_rmssd_interp}

"""
    else:
        report += "相関分析に十分なデータがありませんでした。\n\n"

    report += """### Alpha Power vs LF/HF Ratio

"""
    if 'alpha_lfhf' in correlations:
        corr = correlations['alpha_lfhf']
        sig = '***' if corr['p'] < 0.001 else '**' if corr['p'] < 0.01 else '*' if corr['p'] < 0.05 else 'ns'
        report += f"""| 指標 | 値 |
|:-----|---:|
| 相関係数 (r) | {corr['r']:.3f} |
| p値 | {corr['p']:.4f} ({sig}) |
| サンプル数 | {corr['n']} |

**解釈**: {alpha_lfhf_interp}

"""
    else:
        report += "相関分析に十分なデータがありませんでした。\n\n"

    report += f"""---

## 総合評価

### 脳と心臓のリラックス状態

1. **脳波（Alpha波）**: {alpha_result.score:.1f} dBx
   - Alpha波は精神的なリラックス・落ち着きを反映

2. **心拍変動（HRV）**:
   - RMSSD {mean_rmssd:.2f} ms → {rmssd_eval}
   - LF/HF {mean_lfhf:.2f} → {lfhf_eval}

3. **脳-心臓の連動**:
"""

    if 'alpha_rmssd' in correlations and correlations['alpha_rmssd']['p'] < 0.05:
        r = correlations['alpha_rmssd']['r']
        if r > 0.3:
            report += "   - 脳と心臓のリラックス状態が連動している（正の相関）\n"
        elif r < -0.3:
            report += "   - 脳と心臓のリラックス状態が逆方向（負の相関）\n"
        else:
            report += "   - 脳と心臓の連動は弱い\n"
    else:
        report += "   - 統計的に有意な連動は検出されず\n"

    report += f"""
---

## 可視化

![Alpha-HRV分析](alpha_hrv_analysis.png)

---

## 理論的背景

### Alpha波とリラックス

- Alpha波（8-13Hz）は、閉眼時やリラックス時に後頭部で優位になる
- 瞑想やマインドフルネス実践時に増加することが知られている
- Alpha波の増加は精神的な落ち着きと関連

### HRVとリラックス

- **RMSSD**: 副交感神経（リラックス系）活動を反映
  - 高い値 = リラックス状態
  - 低い値 = ストレス状態

- **LF/HF比**: 自律神経バランスを反映
  - < 1.0: 副交感神経優位（リラックス）
  - 1.0-2.0: バランス状態
  - > 2.0: 交感神経優位（緊張・覚醒）

### 期待される関係

理論的には、Alpha波とリラックス関連HRV指標の間には正の相関が期待される：
- Alpha波 ↑ と RMSSD ↑ が同期
- Alpha波 ↑ と LF/HF比 ↓ が同期

ただし、個人差や測定条件により結果は異なることがある。

---

**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'Report saved: {output_path}')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='Alpha波とHRVの関係分析'
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
        help='SelfLoops HRV CSVファイルパス'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='出力ディレクトリ（デフォルト: カレントディレクトリ）'
    )

    args = parser.parse_args()

    # 出力ディレクトリの設定
    if args.output is None:
        args.output = Path(__file__).parent / 'alpha_hrv_output'

    # パスの検証
    if not args.eeg.exists():
        print(f'Error: EEGファイルが見つかりません: {args.eeg}')
        return 1

    if not args.hrv.exists():
        print(f'Error: HRVファイルが見つかりません: {args.hrv}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    print('='*60)
    print('Alpha波とHRVの関係分析')
    print('='*60)
    print()

    # 1. EEGデータ読み込み
    print(f'Loading EEG: {args.eeg.name}')
    eeg_df = load_mind_monitor_csv(args.eeg, filter_headband=False, warmup_seconds=60)
    print(f'  データ形状: {eeg_df.shape[0]} 行 × {eeg_df.shape[1]} 列')

    eeg_start = pd.to_datetime(eeg_df['TimeStamp'].iloc[0])
    eeg_end = pd.to_datetime(eeg_df['TimeStamp'].iloc[-1])
    eeg_duration = (eeg_end - eeg_start).total_seconds() / 60
    print(f'  測定時間: {eeg_duration:.1f} 分')
    print()

    # 2. HRVデータ読み込み
    print(f'Loading HRV: {args.hrv.name}')
    hrv_df = load_selfloops_csv(str(args.hrv))
    print(f'  データ形状: {hrv_df.shape[0]} 行 × {hrv_df.shape[1]} 列')

    hrv_duration = hrv_df['Time (ms)'].iloc[-1] / 1000 / 60
    print(f'  測定時間: {hrv_duration:.1f} 分')
    print()

    # 3. Alpha波解析
    print('Analyzing: Alpha Power...')
    try:
        alpha_result = calculate_alpha_power(eeg_df, resample_interval='10s')
        print(f'  Alpha Power: {alpha_result.alpha_db:.2f} dB')
        print(f'  Alpha Score: {alpha_result.score:.1f} dBx')
    except Exception as e:
        print(f'  Error: {e}')
        return 1
    print()

    # 4. HRV解析
    print('Analyzing: HRV...')
    try:
        hrv_data = get_hrv_data(hrv_df)
        hrv_result = calculate_hrv_standard_set(hrv_data, window_seconds=180, step_seconds=30)
        print(f'  RMSSD (mean): {hrv_result.metadata["mean_rmssd"]:.2f} ms')
        print(f'  LF/HF (mean): {hrv_result.metadata["mean_lfhf"]:.2f}')
    except Exception as e:
        print(f'  Error: {e}')
        return 1
    print()

    # 5. 時系列同期
    print('Synchronizing: EEG and HRV timeseries...')
    # HRVの開始時刻を推定（EEGと同時開始と仮定）
    hrv_start = eeg_start
    aligned_df = create_aligned_timeseries(alpha_result, hrv_result, eeg_start, hrv_start)
    print(f'  同期データポイント数: {len(aligned_df)}')
    print()

    # 6. 相関分析
    print('Calculating: Correlations...')
    correlations = calculate_correlations(aligned_df)
    if 'alpha_rmssd' in correlations:
        corr = correlations['alpha_rmssd']
        print(f'  Alpha vs RMSSD: r={corr["r"]:.3f}, p={corr["p"]:.4f}')
    if 'alpha_lfhf' in correlations:
        corr = correlations['alpha_lfhf']
        print(f'  Alpha vs LF/HF: r={corr["r"]:.3f}, p={corr["p"]:.4f}')
    print()

    # 7. 可視化
    print('Generating: Analysis plot...')
    plot_alpha_hrv_analysis(
        alpha_result,
        hrv_result,
        aligned_df,
        correlations,
        args.output / 'alpha_hrv_analysis.png'
    )
    print()

    # 8. レポート生成
    print('Generating: Report...')
    generate_report(
        alpha_result,
        hrv_result,
        aligned_df,
        correlations,
        eeg_duration,
        hrv_duration,
        args.output / 'ALPHA_HRV_REPORT.md'
    )
    print()

    print('='*60)
    print('Analysis complete!')
    print('='*60)
    print(f'Output directory: {args.output}')
    print(f'  - ALPHA_HRV_REPORT.md')
    print(f'  - alpha_hrv_analysis.png')

    return 0


if __name__ == '__main__':
    exit(main())
