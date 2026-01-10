#!/usr/bin/env python3
"""
Muse心拍数 vs SelfLoops心拍数 比較分析スクリプト

Muse EEGのPPG心拍数とSelfLoops HRVのECG心拍数を比較し、
どちらがより信頼できるかを分析します。

Usage:
    python compare_heart_rate.py --eeg <CSV_PATH> --hrv <TXT_PATH> [--output <OUTPUT_DIR>]
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

from lib import load_mind_monitor_csv, get_heart_rate_data

# HRV分析関数をインポート
from analyze_hrv import load_selfloops_data


def load_selfloops_with_timestamp(file_path):
    """
    SelfLoops HRV dataを読み込み、開始タイムスタンプも取得

    Args:
        file_path: データファイルのパス

    Returns:
        tuple: (DataFrame, start_timestamp)
    """
    # 1行目のタイムスタンプを読み取る
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    # "10 1月 2026 16:08:50" 形式をパース
    # 日本語の月名を数字に変換
    month_map = {
        '1月': '01', '2月': '02', '3月': '03', '4月': '04',
        '5月': '05', '6月': '06', '7月': '07', '8月': '08',
        '9月': '09', '10月': '10', '11月': '11', '12月': '12'
    }

    parts = first_line.split()
    if len(parts) == 4:
        day = parts[0]
        month_str = parts[1]
        year = parts[2]
        time_str = parts[3]

        # 月を数字に変換
        month = month_map.get(month_str, '01')

        # ISO形式に変換
        timestamp_str = f'{year}-{month}-{day.zfill(2)} {time_str}'
        start_timestamp = pd.to_datetime(timestamp_str)
    else:
        # パースに失敗した場合はNoneを返す
        start_timestamp = None

    # データを読み込む
    df = load_selfloops_data(file_path)

    return df, start_timestamp


def calculate_hr_from_rr(rr_intervals_ms):
    """
    R-R間隔から心拍数を計算

    Args:
        rr_intervals_ms: R-R間隔の配列 (ms)

    Returns:
        numpy.ndarray: 心拍数の配列 (bpm)
    """
    # HR (bpm) = 60000 ms/min / RR (ms)
    hr = 60000.0 / rr_intervals_ms
    return hr


def synchronize_heart_rates(muse_hr_data, selfloops_df, selfloops_start_time, interval_sec=1):
    """
    MuseとSelfLoopsの心拍数を時間軸で同期

    Args:
        muse_hr_data: Muse心拍数データ（dict）
        selfloops_df: SelfLoops HRVデータフレーム
        selfloops_start_time: SelfLoopsの測定開始時刻
        interval_sec: リサンプリング間隔（秒）

    Returns:
        pandas.DataFrame: 同期済みデータフレーム
    """
    # Muse心拍数の時系列
    muse_start = pd.to_datetime(muse_hr_data['timestamps'][0])
    muse_timestamps = pd.to_datetime(muse_hr_data['timestamps'])
    muse_hr = muse_hr_data['heart_rate']

    # SelfLoops心拍数の計算とタイムスタンプ
    selfloops_hr = calculate_hr_from_rr(selfloops_df['R-R (ms)'].values)
    # 正しい開始時刻を使用
    if selfloops_start_time is not None:
        selfloops_timestamps = selfloops_start_time + pd.to_timedelta(selfloops_df['Time (ms)'], unit='ms')
    else:
        # フォールバック: Museの開始時刻を使用
        print('  警告: SelfLoopsの開始時刻が取得できませんでした。Museの開始時刻を使用します。')
        selfloops_timestamps = muse_start + pd.to_timedelta(selfloops_df['Time (ms)'], unit='ms')

    # 共通時間範囲
    common_start = max(muse_timestamps.min(), selfloops_timestamps.min())
    common_end = min(muse_timestamps.max(), selfloops_timestamps.max())

    # 時間軸を作成
    time_index = pd.date_range(start=common_start, end=common_end, freq=f'{interval_sec}s')

    # Muse心拍数をリサンプリング
    muse_series = pd.Series(muse_hr, index=muse_timestamps)
    muse_series = muse_series[~muse_series.index.duplicated(keep='first')]
    muse_series = muse_series.sort_index()
    muse_resampled = muse_series.reindex(time_index, method='nearest', tolerance='5s')

    # SelfLoops心拍数をリサンプリング
    selfloops_series = pd.Series(selfloops_hr, index=selfloops_timestamps)
    selfloops_series = selfloops_series[~selfloops_series.index.duplicated(keep='first')]
    selfloops_series = selfloops_series.sort_index()
    selfloops_resampled = selfloops_series.reindex(time_index, method='nearest', tolerance='5s')

    # 統合データフレーム
    aligned_df = pd.DataFrame({
        'timestamp': time_index,
        'muse_hr': muse_resampled.values,
        'selfloops_hr': selfloops_resampled.values,
    })

    # 欠損値を削除
    aligned_df = aligned_df.dropna()

    return aligned_df


def create_minute_by_minute_comparison(aligned_df):
    """
    1分ごとの比較表を作成

    Args:
        aligned_df: 同期済みデータフレーム

    Returns:
        pandas.DataFrame: 1分ごとの統計
    """
    # 経過時間（分）を計算
    aligned_df_copy = aligned_df.copy()
    aligned_df_copy['elapsed_min'] = (
        (aligned_df_copy['timestamp'] - aligned_df_copy['timestamp'].iloc[0]).dt.total_seconds() / 60
    ).astype(int)

    # 1分ごとにグループ化して統計計算
    minute_stats = []
    for minute in sorted(aligned_df_copy['elapsed_min'].unique()):
        minute_data = aligned_df_copy[aligned_df_copy['elapsed_min'] == minute]

        if len(minute_data) > 0:
            muse_mean = minute_data['muse_hr'].mean()
            selfloops_mean = minute_data['selfloops_hr'].mean()
            diff = muse_mean - selfloops_mean
            abs_diff = abs(diff)

            minute_stats.append({
                'minute': minute,
                'muse_hr_mean': muse_mean,
                'muse_hr_std': minute_data['muse_hr'].std(),
                'selfloops_hr_mean': selfloops_mean,
                'selfloops_hr_std': minute_data['selfloops_hr'].std(),
                'diff': diff,
                'abs_diff': abs_diff,
                'n_samples': len(minute_data),
            })

    return pd.DataFrame(minute_stats)


def calculate_agreement_statistics(muse_hr, selfloops_hr):
    """
    2つの心拍数測定の一致度統計を計算

    Args:
        muse_hr: Muse心拍数
        selfloops_hr: SelfLoops心拍数

    Returns:
        dict: 統計指標
    """
    # 基本統計
    mean_diff = np.mean(muse_hr - selfloops_hr)
    std_diff = np.std(muse_hr - selfloops_hr, ddof=1)

    # Bland-Altman統計
    # Limits of Agreement (LoA): mean ± 1.96*SD
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # 相関
    correlation, p_value = stats.pearsonr(muse_hr, selfloops_hr)

    # 絶対誤差
    abs_diff = np.abs(muse_hr - selfloops_hr)
    mean_abs_diff = np.mean(abs_diff)
    median_abs_diff = np.median(abs_diff)

    # 相対誤差（％）
    rel_error = 100 * abs_diff / selfloops_hr
    mean_rel_error = np.mean(rel_error)

    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((muse_hr - selfloops_hr) ** 2))

    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'loa_upper': loa_upper,
        'loa_lower': loa_lower,
        'correlation': correlation,
        'p_value': p_value,
        'mean_abs_diff': mean_abs_diff,
        'median_abs_diff': median_abs_diff,
        'mean_rel_error': mean_rel_error,
        'rmse': rmse,
    }


def plot_comparison(aligned_df, stats_dict, minute_df, output_path):
    """
    心拍数比較の可視化

    Args:
        aligned_df: 同期済みデータフレーム
        stats_dict: 統計指標
        minute_df: 1分ごとの比較データ
        output_path: 出力パス
    """
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    muse_hr = aligned_df['muse_hr'].values
    selfloops_hr = aligned_df['selfloops_hr'].values
    time_min = (aligned_df['timestamp'] - aligned_df['timestamp'].iloc[0]).dt.total_seconds() / 60

    # 1. 時系列比較
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_min, muse_hr, label='Muse (PPG)', color='blue', linewidth=1.2, alpha=0.8)
    ax1.plot(time_min, selfloops_hr, label='SelfLoops (ECG)', color='red', linewidth=1.2, alpha=0.8)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Heart Rate (bpm)')
    ax1.set_title('Heart Rate Time Series Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 統計情報をテキストで追加
    textstr = f'Correlation: {stats_dict["correlation"]:.3f}\nMean Diff: {stats_dict["mean_diff"]:.2f} bpm\nRMSE: {stats_dict["rmse"]:.2f} bpm'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 散布図 + 回帰線
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(selfloops_hr, muse_hr, alpha=0.5, s=20, edgecolors='none')

    # 回帰線
    slope, intercept = np.polyfit(selfloops_hr, muse_hr, 1)
    x_line = np.array([selfloops_hr.min(), selfloops_hr.max()])
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, 'r-', linewidth=2, label=f'y={slope:.3f}x+{intercept:.2f}')

    # 理想線（y=x）
    ax2.plot(x_line, x_line, 'k--', linewidth=1.5, label='y=x (ideal)', alpha=0.7)

    ax2.set_xlabel('SelfLoops HR (bpm)')
    ax2.set_ylabel('Muse HR (bpm)')
    ax2.set_title('Scatter Plot: Muse vs SelfLoops')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 3. Bland-Altman Plot
    ax3 = fig.add_subplot(gs[1, 1])
    mean_hr = (muse_hr + selfloops_hr) / 2
    diff_hr = muse_hr - selfloops_hr

    ax3.scatter(mean_hr, diff_hr, alpha=0.5, s=20, edgecolors='none')

    # 平均線
    ax3.axhline(stats_dict['mean_diff'], color='blue', linestyle='-', linewidth=2, label=f'Mean: {stats_dict["mean_diff"]:.2f}')

    # Limits of Agreement
    ax3.axhline(stats_dict['loa_upper'], color='red', linestyle='--', linewidth=1.5, label=f'LoA: ±{1.96 * stats_dict["std_diff"]:.2f}')
    ax3.axhline(stats_dict['loa_lower'], color='red', linestyle='--', linewidth=1.5)

    # ゼロ線
    ax3.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    ax3.set_xlabel('Mean HR (bpm)')
    ax3.set_ylabel('Difference (Muse - SelfLoops) (bpm)')
    ax3.set_title('Bland-Altman Plot')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 誤差分布ヒストグラム
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(diff_hr, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(stats_dict['mean_diff'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_dict["mean_diff"]:.2f}')
    ax4.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax4.set_xlabel('Difference (Muse - SelfLoops) (bpm)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Differences')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 絶対誤差の時系列
    ax5 = fig.add_subplot(gs[2, 1])
    abs_diff = np.abs(diff_hr)
    ax5.plot(time_min, abs_diff, color='purple', linewidth=1.2, alpha=0.8)
    ax5.axhline(stats_dict['mean_abs_diff'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_dict["mean_abs_diff"]:.2f}')
    ax5.set_xlabel('Time (min)')
    ax5.set_ylabel('Absolute Difference (bpm)')
    ax5.set_title('Absolute Error Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 1分ごとの平均比較
    ax6 = fig.add_subplot(gs[3, :])
    if minute_df is not None and len(minute_df) > 0:
        x = minute_df['minute'].values

        # エラーバー付きプロット
        ax6.errorbar(x, minute_df['muse_hr_mean'], yerr=minute_df['muse_hr_std'],
                     label='Muse (PPG)', fmt='o-', color='blue', capsize=3, alpha=0.8)
        ax6.errorbar(x, minute_df['selfloops_hr_mean'], yerr=minute_df['selfloops_hr_std'],
                     label='SelfLoops (ECG)', fmt='s-', color='red', capsize=3, alpha=0.8)

        ax6.set_xlabel('Elapsed Time (min)')
        ax6.set_ylabel('Heart Rate (bpm)')
        ax6.set_title('Minute-by-Minute Heart Rate Comparison (Mean ± SD)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'✓ 比較プロット保存: {output_path}')


def generate_comparison_report(aligned_df, stats_dict, minute_df, output_path):
    """
    心拍数比較レポートを生成

    Args:
        aligned_df: 同期済みデータフレーム
        stats_dict: 統計指標
        minute_df: 1分ごとの比較データ
        output_path: 出力パス
    """
    muse_hr = aligned_df['muse_hr'].values
    selfloops_hr = aligned_df['selfloops_hr'].values

    # 基本統計
    muse_mean = np.mean(muse_hr)
    muse_std = np.std(muse_hr, ddof=1)
    selfloops_mean = np.mean(selfloops_hr)
    selfloops_std = np.std(selfloops_hr, ddof=1)

    report = f"""# Muse vs SelfLoops 心拍数比較分析レポート

## 概要

Muse EEGのPPG（光学式）心拍数とSelfLoops HRVのECG心拍数を比較し、
測定精度と信頼性を評価しました。

### 測定方式の違い

**Muse（PPG - Photoplethysmography）**
- 測定方式: 光学式（緑色LED）
- 測定部位: 耳後部（TP9/TP10）
- 原理: 血液量変化による光の吸収変化を検出
- 特徴: 非侵襲的、装着が容易

**SelfLoops（ECG - Electrocardiogram）**
- 測定方式: 電気信号（R-R間隔から計算）
- 測定部位: 胸部ストラップ（推定）
- 原理: 心臓の電気活動を直接測定
- 特徴: 医療グレードの精度、ゴールドスタンダード

---

## 比較データサマリー

### 基本統計

| 指標 | Muse (PPG) | SelfLoops (ECG) | 単位 |
|:-----|----------:|----------------:|:-----|
| 平均心拍数 | {muse_mean:.2f} | {selfloops_mean:.2f} | bpm |
| 標準偏差 | {muse_std:.2f} | {selfloops_std:.2f} | bpm |
| 最小値 | {np.min(muse_hr):.2f} | {np.min(selfloops_hr):.2f} | bpm |
| 最大値 | {np.max(muse_hr):.2f} | {np.max(selfloops_hr):.2f} | bpm |
| データポイント数 | {len(muse_hr)} | {len(selfloops_hr)} | - |

---

## 一致度分析

### 相関分析

| 指標 | 値 |
|:-----|---:|
| **Pearson相関係数** | {stats_dict['correlation']:.4f} |
| **p値** | {stats_dict['p_value']:.6f} |
| **解釈** | {'強い正の相関' if stats_dict['correlation'] > 0.9 else '中程度の正の相関' if stats_dict['correlation'] > 0.7 else '弱い正の相関' if stats_dict['correlation'] > 0.5 else '相関が弱い'} |

### Bland-Altman分析

| 指標 | 値 | 説明 |
|:-----|---:|:-----|
| **平均差** | {stats_dict['mean_diff']:.2f} bpm | Muse - SelfLoopsの平均値 |
| **標準偏差** | {stats_dict['std_diff']:.2f} bpm | 差の標準偏差 |
| **LoA上限** | {stats_dict['loa_upper']:.2f} bpm | 平均 + 1.96SD |
| **LoA下限** | {stats_dict['loa_lower']:.2f} bpm | 平均 - 1.96SD |

> **Limits of Agreement (LoA)**: 95%のデータ点がこの範囲内に入る

### 誤差指標

| 指標 | 値 | 説明 |
|:-----|---:|:-----|
| **平均絶対誤差** | {stats_dict['mean_abs_diff']:.2f} bpm | |Muse - SelfLoops|の平均 |
| **中央絶対誤差** | {stats_dict['median_abs_diff']:.2f} bpm | |Muse - SelfLoops|の中央値 |
| **平均相対誤差** | {stats_dict['mean_rel_error']:.2f} % | 誤差の割合 |
| **RMSE** | {stats_dict['rmse']:.2f} bpm | 二乗平均平方根誤差 |

---

## 可視化

![心拍数比較](hr_comparison.png)

---

## 1分ごとの詳細比較

以下は1分ごとの平均心拍数と誤差の詳細です。

| 経過時間 | Muse平均 | Muse SD | SelfLoops平均 | SelfLoops SD | 差分 | 絶対誤差 | サンプル数 |
|:---------|--------:|---------:|-------------:|------------:|-----:|---------:|----------:|
"""

    # 1分ごとの比較表を追加
    if minute_df is not None and len(minute_df) > 0:
        for _, row in minute_df.iterrows():
            minute_int = int(row['minute'])
            report += f"| {minute_int:2d}分 | {row['muse_hr_mean']:6.2f} | {row['muse_hr_std']:5.2f} | {row['selfloops_hr_mean']:6.2f} | {row['selfloops_hr_std']:5.2f} | {row['diff']:5.2f} | {row['abs_diff']:5.2f} | {int(row['n_samples']):3d} |\n"
    else:
        report += "| (データなし) | - | - | - | - | - | - | - |\n"

    report += """
> **注**:
> - 差分 = Muse - SelfLoops（正の値はMuseが高い）
> - 絶対誤差 = |差分|
> - SD = 標準偏差（その分の変動幅）

---

## 評価と結論

### 1. 測定精度

**相関係数: {stats_dict['correlation']:.3f}**
- {'非常に高い相関を示しており、MuseとSelfLoopsは類似した心拍数トレンドを捉えている' if stats_dict['correlation'] > 0.9 else '高い相関を示しており、両者は同様の傾向を示す' if stats_dict['correlation'] > 0.8 else '中程度の相関。一部の測定で乖離がある可能性'}

**平均絶対誤差: {stats_dict['mean_abs_diff']:.2f} bpm**
- {'非常に良好な一致' if stats_dict['mean_abs_diff'] < 2 else '良好な一致' if stats_dict['mean_abs_diff'] < 5 else '中程度の一致' if stats_dict['mean_abs_diff'] < 10 else '一致度が低い'}
- {'臨床的に許容可能な範囲内' if stats_dict['mean_abs_diff'] < 5 else '一部のアプリケーションでは注意が必要'}

### 2. バイアス（系統誤差）

**平均差: {stats_dict['mean_diff']:.2f} bpm**
- {'Museは若干高めに測定' if stats_dict['mean_diff'] > 1 else 'Museは若干低めに測定' if stats_dict['mean_diff'] < -1 else '系統誤差はほぼゼロ'}
- {'一貫したバイアスがあるため、補正可能' if abs(stats_dict['mean_diff']) > 2 else 'バイアスは小さく、臨床的に無視できるレベル'}

### 3. どちらがより信頼できるか？

**SelfLoops (ECG)が医療的ゴールドスタンダード**

理由：
1. **測定原理**: ECGは心臓の電気活動を直接測定するため、最も正確
2. **医療グレード**: HRV分析のゴールドスタンダード
3. **R-R間隔**: 心拍ごとの正確な間隔を測定可能

**Muse (PPG)の利点と限界**

利点：
- 非侵襲的で装着が容易
- EEGと同時測定が可能
- {'平均心拍数の推定には十分な精度' if stats_dict['mean_abs_diff'] < 5 else '概算としては使用可能'}

限界：
- 動作アーティファクトに敏感
- {'瞬間的な心拍数変動（HRV）の測定には不向き' if stats_dict['correlation'] < 0.95 else 'HRV分析にも使用可能だが、ECGより精度は劣る'}
- 測定部位（耳後部）の血流により影響を受ける

### 4. 推奨される使用シーン

**SelfLoops (ECG)を推奨:**
- HRV分析（SDNN、RMSSD、LF/HFなど）
- 正確な心拍数変動の測定が必要な場合
- 医療・研究用途

**Muse (PPG)で十分:**
- 平均心拍数のモニタリング
- トレンドの把握
- EEG分析と組み合わせた瞑想分析

---

## 改善の提案

### 1. Museの精度向上

- 測定前にセンサー部位の清掃
- ヘッドバンドの適切な装着（締め付けすぎない）
- 動作を最小限に抑える

### 2. データ統合の最適化

- {'バイアス補正: Muse HR = 測定値 - {stats_dict["mean_diff"]:.2f} bpm' if abs(stats_dict['mean_diff']) > 2 else 'バイアス補正は不要'}
- 移動平均フィルタの適用でノイズ除去

### 3. 用途に応じた選択

- **瞑想の質的評価**: Muse PPGで十分（トレンドが重要）
- **HRV詳細分析**: SelfLoops ECGを使用（精度が重要）

---

## まとめ

- **相関**: {stats_dict['correlation']:.3f}（{'優秀' if stats_dict['correlation'] > 0.9 else '良好' if stats_dict['correlation'] > 0.8 else '中程度'}）
- **平均誤差**: {stats_dict['mean_abs_diff']:.2f} bpm（{'許容範囲' if stats_dict['mean_abs_diff'] < 5 else '要注意'}）
- **推奨**: **精密測定にはSelfLoops ECG、EEG統合分析にはMuse PPG**

---

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'✓ 比較レポート生成: {output_path}')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='Muse vs SelfLoops 心拍数比較分析'
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
    print('Muse vs SelfLoops 心拍数比較分析')
    print('='*60)
    print()

    # 1. データ読み込み
    print(f'Loading EEG: {args.eeg.name}')
    eeg_df = load_mind_monitor_csv(args.eeg, filter_headband=False, warmup_seconds=60)
    print(f'EEGデータ形状: {eeg_df.shape[0]} 行 × {eeg_df.shape[1]} 列')

    print(f'Loading HRV: {args.hrv.name}')
    hrv_df, hrv_start_time = load_selfloops_with_timestamp(args.hrv)
    print(f'HRVデータ形状: {hrv_df.shape[0]} 行 × {hrv_df.shape[1]} 列')
    if hrv_start_time:
        print(f'SelfLoops測定開始時刻: {hrv_start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    # 2. Muse心拍数データ取得
    print('取得中: Muse心拍数データ...')
    muse_hr_data = get_heart_rate_data(eeg_df)
    print(f'  データポイント数: {len(muse_hr_data["heart_rate"])}')
    print(f'  平均心拍数: {np.mean(muse_hr_data["heart_rate"]):.2f} bpm')
    print()

    # 3. データ同期
    print('同期中: 心拍数データ...')
    aligned_df = synchronize_heart_rates(muse_hr_data, hrv_df, hrv_start_time, interval_sec=1)
    print(f'  同期データポイント数: {len(aligned_df)}')
    if len(aligned_df) > 0:
        print(f'  同期範囲: {aligned_df["timestamp"].iloc[0]} ~ {aligned_df["timestamp"].iloc[-1]}')
    print()

    # 4. 1分ごとの比較データ作成
    print('作成中: 1分ごとの比較表...')
    minute_df = create_minute_by_minute_comparison(aligned_df)
    print(f'  1分ごとのデータポイント数: {len(minute_df)}')
    print()

    # 5. 統計分析
    print('分析中: 一致度統計...')
    stats_dict = calculate_agreement_statistics(
        aligned_df['muse_hr'].values,
        aligned_df['selfloops_hr'].values
    )
    print(f'  相関係数: {stats_dict["correlation"]:.4f}')
    print(f'  平均差: {stats_dict["mean_diff"]:.2f} bpm')
    print(f'  平均絶対誤差: {stats_dict["mean_abs_diff"]:.2f} bpm')
    print(f'  RMSE: {stats_dict["rmse"]:.2f} bpm')
    print()

    # 6. 可視化
    print('生成中: 比較プロット...')
    plot_comparison(aligned_df, stats_dict, minute_df, args.output / 'hr_comparison.png')
    print()

    # 7. レポート生成
    print('生成中: 比較レポート...')
    generate_comparison_report(aligned_df, stats_dict, minute_df, args.output / 'HR_COMPARISON_REPORT.md')
    print()

    print('='*60)
    print('分析完了!')
    print('='*60)
    print(f'出力ディレクトリ: {args.output}')
    print(f'  - HR_COMPARISON_REPORT.md')
    print(f'  - hr_comparison.png')

    return 0


if __name__ == '__main__':
    exit(main())
