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

import mne
import pandas as pd

from ._band_power_base import calculate_band_power

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


def calculate_smr(
    df: pd.DataFrame,
    channels: Optional[Iterable[str]] = None,
    band: Optional[Tuple[float, float]] = None,
    band_key: Optional[str] = None,
    resample_interval: str = '10s',
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
    computation = calculate_band_power(
        df,
        channels=channels,
        band=band,
        band_key=band_key,
        band_options=SMR_BAND_OPTIONS,
        default_band_key='narrow',
        band_label_for_error='SMR',
        empty_series_message='SMR time series is empty.',
        metadata_extra={'measurement_note': 'AF領域での測定（本来のSMRはC3/C4）'},
        resample_interval=resample_interval,
        smoothing_seconds=smoothing_seconds,
        rolling_window_seconds=rolling_window_seconds,
        raw=raw,
    )

    return SMRResult(
        time_series=computation.time_series,
        statistics=computation.statistics,
        metadata=computation.metadata,
    )
