"""
Spectral Entropy (SE) 解析モジュール

パワースペクトル密度（PSD）から周波数成分の多様性を測定し、
脳波の「複雑性」や「集中度」を定量化する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SpectralEntropyResult:
    """Spectral Entropy解析結果を保持するデータクラス。"""

    entropy: float
    statistics: pd.DataFrame
    metadata: dict


def _calculate_shannon_entropy(psd_values: np.ndarray, normalize: bool = True) -> float:
    """
    Shannon Entropyを計算

    Parameters
    ----------
    psd_values : np.ndarray
        パワースペクトル密度の値
    normalize : bool
        0-1に正規化するか（最大エントロピーで割る）

    Returns
    -------
    float
        Spectral Entropy値
    """
    # ゼロとNaNを除去、微小値でクリップ
    psd = np.maximum(psd_values, np.finfo(float).eps)

    # 確率分布に正規化
    p = psd / np.sum(psd)

    # Shannon Entropy計算: H = -∑ p(f) * log2(p(f))
    entropy = -np.sum(p * np.log2(p))

    # 正規化（最大エントロピーで割る）
    if normalize:
        max_entropy = np.log2(len(psd))
        if max_entropy > 0:
            entropy = entropy / max_entropy

    return entropy


def calculate_spectral_entropy(
    psd_dict: dict,
    freq_range: Optional[Tuple[float, float]] = (1.0, 40.0),
    normalize: bool = True,
) -> SpectralEntropyResult:
    """
    Spectral Entropyを計算

    Parameters
    ----------
    psd_dict : dict
        calculate_psd()の戻り値
        - 'freqs': 周波数配列
        - 'psds': パワースペクトル密度配列（チャネル数 x 周波数数）
        - 'channels': チャネル名リスト
    freq_range : tuple, optional
        計算対象周波数範囲（Hz）。デフォルトは1-40Hz
    normalize : bool
        0-1に正規化するか

    Returns
    -------
    SpectralEntropyResult
        エントロピー値、統計情報、メタデータ
    """
    freqs = psd_dict['freqs']
    psds = psd_dict['psds']
    channels = psd_dict['channels']

    # 周波数範囲でマスク
    freq_low, freq_high = freq_range
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    freqs_selected = freqs[freq_mask]

    # 各チャネルのエントロピーを計算
    entropy_values = []
    for i, ch_name in enumerate(channels):
        psd_ch = psds[i][freq_mask]
        entropy_ch = _calculate_shannon_entropy(psd_ch, normalize=normalize)
        entropy_values.append(entropy_ch)

    # 全チャネルの平均エントロピー
    mean_entropy = np.mean(entropy_values)
    median_entropy = np.median(entropy_values)
    std_entropy = np.std(entropy_values)

    # 統計DataFrameを作成
    stats_df = pd.DataFrame(
        [
            {'Metric': 'Mean', 'Value': mean_entropy, 'Unit': 'normalized' if normalize else 'bits'},
            {'Metric': 'Median', 'Value': median_entropy, 'Unit': 'normalized' if normalize else 'bits'},
            {'Metric': 'Std Dev', 'Value': std_entropy, 'Unit': 'normalized' if normalize else 'bits'},
        ]
    )

    metadata = {
        'channels': [ch.replace('RAW_', '') for ch in channels],
        'freq_range': freq_range,
        'normalize': normalize,
        'method': 'shannon',
        'entropy_by_channel': {
            ch.replace('RAW_', ''): ent
            for ch, ent in zip(channels, entropy_values)
        },
    }

    return SpectralEntropyResult(
        entropy=mean_entropy,
        statistics=stats_df,
        metadata=metadata,
    )


def calculate_spectral_entropy_time_series(
    tfr_dict: dict,
    freq_range: Optional[Tuple[float, float]] = (1.0, 40.0),
    normalize: bool = True,
    start_time: Optional[pd.Timestamp] = None,
) -> SpectralEntropyResult:
    """
    時間変化するSpectral Entropyを計算

    Parameters
    ----------
    tfr_dict : dict
        calculate_spectrogram()の戻り値
        - 'power': パワー配列（周波数 x 時間）
        - 'freqs': 周波数配列
        - 'times': 時間配列（秒）
    freq_range : tuple, optional
        計算対象周波数範囲（Hz）
    normalize : bool
        0-1に正規化するか
    start_time : pd.Timestamp, optional
        セッション開始時刻

    Returns
    -------
    SpectralEntropyResult
        時系列エントロピー、統計情報、メタデータ
    """
    power = tfr_dict['power']
    freqs = tfr_dict['freqs']
    times = tfr_dict['times']

    # 周波数範囲でマスク
    freq_low, freq_high = freq_range
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    freqs_selected = freqs[freq_mask]
    power_selected = power[freq_mask, :]

    # 各時間点でのエントロピーを計算
    entropy_over_time = []
    for t_idx in range(power_selected.shape[1]):
        psd_at_t = power_selected[:, t_idx]
        entropy_at_t = _calculate_shannon_entropy(psd_at_t, normalize=normalize)
        entropy_over_time.append(entropy_at_t)

    entropy_array = np.array(entropy_over_time)

    # タイムインデックス作成
    if start_time is not None:
        time_index = start_time + pd.to_timedelta(times, unit='s')
        entropy_series = pd.Series(entropy_array, index=time_index)
    else:
        entropy_series = pd.Series(entropy_array, index=times)

    # セッション前半・後半の比較
    midpoint = entropy_series.index[len(entropy_series) // 2]
    first_half = entropy_series[entropy_series.index <= midpoint]
    second_half = entropy_series[entropy_series.index > midpoint]

    first_mean = first_half.mean() if not first_half.empty else np.nan
    second_mean = second_half.mean() if not second_half.empty else np.nan

    if first_mean and not np.isnan(first_mean) and first_mean != 0:
        change_percent = ((second_mean - first_mean) / first_mean) * 100.0
    else:
        change_percent = np.nan

    # 統計DataFrameを作成
    stats_df = pd.DataFrame(
        [
            {'Metric': 'Mean', 'Value': entropy_series.mean(), 'Unit': 'normalized' if normalize else 'bits'},
            {'Metric': 'Median', 'Value': entropy_series.median(), 'Unit': 'normalized' if normalize else 'bits'},
            {'Metric': 'Std Dev', 'Value': entropy_series.std(), 'Unit': 'normalized' if normalize else 'bits'},
            {'Metric': 'First Half Mean', 'Value': first_mean, 'Unit': 'normalized' if normalize else 'bits'},
            {'Metric': 'Second Half Mean', 'Value': second_mean, 'Unit': 'normalized' if normalize else 'bits'},
            {'Metric': 'Change Rate (2nd/1st)', 'Value': change_percent, 'Unit': '%'},
        ]
    )

    metadata = {
        'freq_range': freq_range,
        'normalize': normalize,
        'method': 'shannon',
        'first_half_mean': first_mean,
        'second_half_mean': second_mean,
        'change_percent': change_percent,
        'time_points': len(entropy_array),
    }

    return SpectralEntropyResult(
        entropy=entropy_series.mean(),
        statistics=stats_df,
        metadata=metadata,
    )


def plot_spectral_entropy(
    result: SpectralEntropyResult,
    img_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[pd.DataFrame, object]:
    """
    Spectral Entropyの統計情報をプロット

    Parameters
    ----------
    result : SpectralEntropyResult
        calculate_spectral_entropy または calculate_spectral_entropy_time_series の戻り値
    img_path : str, optional
        画像保存パス
    title : str, optional
        グラフタイトル

    Returns
    -------
    (stats_df, fig)
        統計情報とFigureオブジェクト
    """
    import matplotlib.pyplot as plt

    stats_df = result.statistics
    metadata = result.metadata

    # プロットデータの準備
    labels = stats_df['Metric'].tolist()
    values = stats_df['Value'].tolist()

    # 変化率以外の値でプロット
    plot_labels = [l for l in labels if 'Change Rate' not in l]
    plot_values = [v for l, v in zip(labels, values) if 'Change Rate' not in l]

    default_title = 'Spectral Entropy (SE)'
    plot_title = title or default_title

    fig, ax = plt.subplots(figsize=(10, 6))

    # 棒グラフ
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(
        range(len(plot_values)),
        plot_values,
        color=colors[:len(plot_values)],
        alpha=0.7,
        edgecolor='black',
    )

    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels, rotation=15, ha='right')
    ax.set_ylabel('Spectral Entropy', fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 変化率を表示
    change_percent = metadata.get('change_percent')
    if change_percent is not None and not np.isnan(change_percent):
        interpretation = "低下（注意集中）" if change_percent < 0 else "上昇（注意散漫）"
        ax.text(
            0.98,
            0.95,
            f'変化率: {change_percent:+.1f}%\n{interpretation}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
        )

    plt.tight_layout()

    if img_path:
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return stats_df, fig
