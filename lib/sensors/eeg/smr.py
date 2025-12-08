"""
SMR (Sensorimotor Rhythm) 解析モジュール

AF7/AF8チャネルのSMR帯域（12-15Hz）パワーを計算する。

注意:
- 本来のSMRはC3/C4（感覚運動野直上）で測定される
- MuseはC3/C4をカバーしていないため、AF領域での測定は代替的なもの
- 実測データでは、AF領域でSMR帯域が鮮明に観察される場合がある
- 本モジュールでは「SMR-band (AF)」または「High Alpha (AF)」として扱う

関連する状態:
- 身体の静止、運動抑制
- 集中、注意制御
- 穏やかな覚醒 (Calm Alertness)

参考:
- SMR帯域: 12-15Hz (Low Beta / High Alpha)
- 増加する条件: 身体を静止させている時、運動を抑制している時
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import mne

from .preprocessing import prepare_mne_raw
from .core.hilbert_power import calculate_channel_average_power
from .core.statistics import (
    calculate_half_comparison,
    create_statistics_dataframe,
    create_metadata,
)

# SMR帯域の定義
SMR_BAND: Tuple[float, float] = (12.0, 15.0)

# SMR帯域プリセット
SMR_BAND_OPTIONS: Dict[str, Tuple[float, float]] = {
    'narrow': (12.0, 15.0),  # 標準的なSMR帯域
    'wide': (12.0, 18.0),    # 広めのSMR/Low Beta
}


@dataclass
class SMRResult:
    """SMR解析結果を保持するデータクラス。"""

    time_series: pd.Series  # SMRパワー時系列 (dB)
    statistics: pd.DataFrame
    metadata: dict


def _prepare_raw_for_channels(
    df: pd.DataFrame,
    channels: list,
    sfreq: Optional[float] = None,
) -> mne.io.BaseRaw:
    """指定チャネルを含むRawオブジェクトを取得。"""
    mne_dict = prepare_mne_raw(df, sfreq=sfreq)
    if not mne_dict:
        raise ValueError('Failed to construct RAW data.')

    raw = mne_dict['raw'].copy()
    available = set(raw.ch_names)
    missing = [ch for ch in channels if ch not in available]
    if missing:
        raise ValueError(f'Specified channels not found: {missing}')

    return raw


def calculate_smr(
    df: pd.DataFrame,
    channels: Optional[Iterable[str]] = None,
    band: Optional[Tuple[float, float]] = None,
    band_key: Optional[str] = None,
    resample_interval: str = '10S',
    smoothing_seconds: float = 6.0,
    rolling_window_seconds: float = 8.0,
    raw: Optional[mne.io.BaseRaw] = None,
) -> SMRResult:
    """
    SMR帯域（12-15Hz）のパワーを計算する。

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorの生データ（TimeStamp, RAW_AF7, RAW_AF8などを含む）。
    channels : iterable, optional
        SMR解析に使用するRAWチャネル。デフォルトはAF7/AF8。
    band : tuple, optional
        抽出する周波数帯域 (Hz)。指定しない場合はband_keyかnarrowを使用。
    band_key : str, optional
        `SMR_BAND_OPTIONS` に定義された帯域キー。例: 'narrow', 'wide'
    resample_interval : str
        可視化用にリサンプルする間隔。デフォルトは10秒。
    smoothing_seconds : float
        ローリング平均による平滑化時定数（秒）。
    rolling_window_seconds : float
        メディアンフィルタのウィンドウ幅（秒）。
    raw : mne.io.BaseRaw, optional
        既存のRawオブジェクト。Noneの場合は新規作成。

    Returns
    -------
    SMRResult
        時系列・統計情報・メタデータを含む解析結果。
        時系列パワーはdB単位（10*log10(μV²)）で出力される。

    Notes
    -----
    AF領域での測定について:
    - 本来のSMRはC3/C4（感覚運動野）で測定される
    - Museではその位置をカバーしていないため、代替としてAF領域を使用
    - 実測データでは、AF領域でSMR帯域が鮮明に観察されることがある
    - これは前頭葉の注意制御・集中活動を反映している可能性がある
    """
    if channels is None:
        channels = ('RAW_AF7', 'RAW_AF8')

    channel_list = list(channels)

    if band is not None:
        band_tuple = band
        band_label = 'custom'
    else:
        key = band_key or 'narrow'
        if key not in SMR_BAND_OPTIONS:
            raise ValueError(f'未定義のSMR帯域キーです: {key}')
        band_tuple = SMR_BAND_OPTIONS[key]
        band_label = key

    # RAWデータ準備
    if raw is None:
        raw = _prepare_raw_for_channels(df, channel_list)
    else:
        raw = raw.copy()

    # セッション開始時刻
    start_time = pd.to_datetime(df['TimeStamp'].min())

    # 処理パラメータ
    processing_params = {
        'resample_interval': resample_interval,
        'smoothing_seconds': smoothing_seconds,
        'rolling_window_seconds': rolling_window_seconds,
    }

    # ヒルベルト変換でバンドパワー計算（チャネル平均）
    smr_series = calculate_channel_average_power(
        raw=raw,
        band=band_tuple,
        channels=channel_list,
        start_time=start_time,
        resample_interval=resample_interval,
        smoothing_seconds=smoothing_seconds,
        rolling_window_seconds=rolling_window_seconds,
        outlier_percentile=0.90,
    )

    if smr_series.empty:
        raise ValueError('SMR time series is empty.')

    # 統計DataFrame作成
    stats_df = create_statistics_dataframe(
        smr_series,
        name='Value',
        unit='dB',
        include_half_comparison=True,
    )
    # Unit列を削除（後方互換性のため）
    if 'Unit' in stats_df.columns:
        stats_df = stats_df.drop(columns=['Unit'])

    # メタデータ作成
    metadata = create_metadata(
        series=smr_series,
        band=band_tuple,
        channels=channel_list,
        sfreq=float(raw.info['sfreq']),
        processing_params=processing_params,
        extra={
            'band_key': band_label,
            'measurement_note': 'AF領域での測定（本来のSMRはC3/C4）',
        },
    )

    # 後方互換性のためのキー追加
    half_stats = calculate_half_comparison(smr_series)
    metadata['increase_db'] = half_stats['change_db']
    metadata['increase_rate_percent'] = half_stats['change_percent']

    return SMRResult(
        time_series=smr_series,
        statistics=stats_df,
        metadata=metadata,
    )
