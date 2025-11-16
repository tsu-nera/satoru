"""
Frontal Midline Theta (Fmθ) 解析モジュール

AF7/AF8チャネルをMNE-Pythonの処理パイプラインでバンドパス→ヒルベルト包絡へ変換し、
Fmθパワーの時系列と統計指標を算出する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import mne

from .preprocessing import prepare_mne_raw

# 代表的なFmθ帯域プリセット（必要に応じて切り替え可能）
FMTHETA_BAND_OPTIONS: Dict[str, Tuple[float, float]] = {
    'narrow': (6.0, 7.0),
    'medium': (5.0, 7.0),
    'wide': (4.0, 8.0),
}


@dataclass
class FrontalThetaResult:
    """Fmθ解析結果を保持するデータクラス。"""

    time_series: pd.Series
    statistics: pd.DataFrame
    metadata: dict


def _prepare_raw_for_fmtheta(
    df: pd.DataFrame,
    channels: Sequence[str],
    sfreq: Optional[float] = None,
) -> mne.io.BaseRaw:
    """AF7/AF8のみ抽出したRawオブジェクトを取得。"""
    mne_dict = prepare_mne_raw(df, sfreq=sfreq)
    if not mne_dict:
        raise ValueError('Failed to construct RAW data.')

    raw = mne_dict['raw'].copy()
    available = set(raw.ch_names)
    missing = [ch for ch in channels if ch not in available]
    if missing:
        raise ValueError(f'Specified channels not found: {missing}')

    raw.pick_channels(list(channels))
    return raw


def calculate_frontal_theta(
    df: pd.DataFrame,
    channels: Optional[Iterable[str]] = None,
    band: Optional[Tuple[float, float]] = None,
    band_key: Optional[str] = None,
    resample_interval: str = '2S',
    smoothing_seconds: float = 6.0,
    rolling_window_seconds: float = 8.0,
    raw: Optional[mne.io.BaseRaw] = None,
) -> FrontalThetaResult:
    """
    Frontal Midline Theta (Fmθ) の指標を計算する。

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorの生データ（TimeStamp, RAW_AF7, RAW_AF8などを含む）。
    channels : iterable, optional
        Fmθ解析に使用するRAWチャネル。デフォルトはAF7/AF8。
    band : tuple, optional
        抽出する周波数帯域 (Hz)。指定しない場合はband_keyかnarrowを使用。
    band_key : str, optional
        `FMTHETA_BAND_OPTIONS` に定義された帯域キー。例: 'narrow', 'medium', 'wide'
    resample_interval : str
        可視化用にリサンプルする間隔。
    smoothing_seconds : float
        ローリング平均による平滑化時定数（秒）。

    Returns
    -------
    FrontalThetaResult
        時系列・統計情報・メタデータを含む解析結果。
    """
    if channels is None:
        channels = ('RAW_AF7', 'RAW_AF8')

    channel_list = tuple(channels)

    if band is not None:
        band_tuple = band
        band_label = 'custom'
    else:
        key = band_key or 'narrow'
        if key not in FMTHETA_BAND_OPTIONS:
            raise ValueError(f'未定義のFmθ帯域キーです: {key}')
        band_tuple = FMTHETA_BAND_OPTIONS[key]
        band_label = key

    if raw is None:
        raw = _prepare_raw_for_fmtheta(df, channel_list)
    else:
        raw = raw.copy().pick_channels(list(channel_list))

    raw_filtered = raw.copy().filter(
        l_freq=band_tuple[0],
        h_freq=band_tuple[1],
        fir_design='firwin',
        phase='zero',
        verbose=False,
    )
    raw_envelope = raw_filtered.copy().apply_hilbert(envelope=True, verbose=False)

    env_data_uv = raw_envelope.get_data(units='uV')
    times = raw_envelope.times
    start_time = pd.to_datetime(df['TimeStamp'].min())
    time_index = start_time + pd.to_timedelta(times, unit='s')

    power_df = pd.DataFrame(
        env_data_uv.T ** 2,
        index=time_index,
        columns=list(channel_list),
    )
    power_series = power_df.mean(axis=1)

    # 外れ値除去（90パーセンタイルでクリップ）
    # 256Hzの高サンプリングレートでは極端な値がより顕著に現れるため必須
    # Z-scoreではなくパーセンタイルベースの方が、装着直後の集中的なスパイクに効果的
    # 90パーセンタイル: 上位10%の極端な値を除去（装着直後の不安定な信号を積極的に除外）
    upper_bound = power_series.quantile(0.90)
    lower_bound = 0.0  # パワーは非負
    power_series = power_series.clip(lower=lower_bound, upper=upper_bound)

    if resample_interval:
        power_series = power_series.resample(resample_interval).median()

    if smoothing_seconds and smoothing_seconds > 0:
        window = f'{max(int(smoothing_seconds), 1)}S'
        power_series = power_series.rolling(window=window, min_periods=1).mean()

    if rolling_window_seconds and rolling_window_seconds > 0:
        window = f'{max(int(rolling_window_seconds), 1)}S'
        power_series = power_series.rolling(window=window, min_periods=1).median()

    series = power_series.dropna()
    if series.empty:
        raise ValueError('Fmθ time series is empty.')

    stats = {
        'Mean (μV²)': series.mean(),
        'Median (μV²)': series.median(),
        'Std Dev (μV²)': series.std(),
    }

    midpoint = series.index[0] + (series.index[-1] - series.index[0]) / 2
    first_half = series[series.index <= midpoint]
    second_half = series[series.index > midpoint]

    first_mean = first_half.mean() if not first_half.empty else np.nan
    second_mean = second_half.mean() if not second_half.empty else np.nan

    if first_mean and not np.isnan(first_mean) and first_mean != 0:
        increase_rate = ((second_mean - first_mean) / first_mean) * 100.0
    else:
        increase_rate = np.nan

    stats_df = pd.DataFrame(
        [
            {'Metric': 'Mean', 'Value': stats['Mean (μV²)'], 'Unit': 'μV²'},
            {'Metric': 'Median', 'Value': stats['Median (μV²)'], 'Unit': 'μV²'},
            {'Metric': 'Std Dev', 'Value': stats['Std Dev (μV²)'], 'Unit': 'μV²'},
            {'Metric': 'First Half Mean', 'Value': first_mean, 'Unit': 'μV²'},
            {'Metric': 'Second Half Mean', 'Value': second_mean, 'Unit': 'μV²'},
            {'Metric': 'Increase Rate (2nd/1st)', 'Value': increase_rate, 'Unit': '%'},
        ]
    )

    metadata = {
        'channels': list(channels),
        'band': band_tuple,
        'band_key': band_label,
        'sfreq': float(raw.info['sfreq']),
        'first_half_mean': first_mean,
        'second_half_mean': second_mean,
        'increase_rate_percent': increase_rate,
        'method': 'mne_hilbert',
        'filter_settings': {
            'l_freq': band_tuple[0],
            'h_freq': band_tuple[1],
            'fir_design': 'firwin',
            'phase': 'zero',
        },
        'processing': {
            'resample_interval': resample_interval,
            'smoothing_seconds': smoothing_seconds,
            'rolling_window_seconds': rolling_window_seconds,
        },
    }

    return FrontalThetaResult(
        time_series=series,
        statistics=stats_df,
        metadata=metadata,
    )


def plot_frontal_theta(
    result: FrontalThetaResult,
    img_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[pd.Series, object]:
    """
    Fmθの時間推移をプロットする。

    Parameters
    ----------
    result : FrontalThetaResult
        `calculate_frontal_theta` の戻り値。
    img_path : str, optional
        画像保存パス。
    title : str
        グラフタイトル。

    Returns
    -------
    (series, fig)
        表示に使用した時系列とFigureオブジェクト。
    """
    import matplotlib.pyplot as plt

    series = result.time_series
    if series.empty:
        raise ValueError('Fmθ time series for plotting is empty.')

    elapsed_minutes = (series.index - series.index[0]).total_seconds() / 60.0

    metadata = result.metadata
    band_key = metadata.get('band_key', '')
    band_range = metadata.get('band', (None, None))
    label = f'Fmθ ({band_range[0]}-{band_range[1]} Hz)' if None not in band_range else 'Fmθ'
    default_title = f'Frontal Midline Theta - {band_key}' if band_key else 'Frontal Midline Theta'
    plot_title = title or default_title

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        elapsed_minutes,
        series.values,
        color='#1f77b4',
        linewidth=2.2,
        label=label,
    )
    midpoint = elapsed_minutes[len(elapsed_minutes) // 2] if len(elapsed_minutes) else 0
    if midpoint:
        ax.axvline(midpoint, color='gray', linestyle='--', alpha=0.5, label='Session midpoint')

    increase = metadata.get('increase_rate_percent')
    if increase is not None and not np.isnan(increase):
        ax.text(
            0.02,
            0.95,
            f'Δ後半/前半: {increase:.1f}%',
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )
    ax.set_xlabel('Elapsed Time (min)', fontsize=12)
    ax.set_ylabel('Fmθ Power (μV²)', fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    plt.tight_layout()

    if img_path:
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return series, fig
