"""
Jinja2カスタムフィルタ

EEG/瞑想分析レポートテンプレート用のフォーマット関数を提供
"""

from datetime import datetime
import pandas as pd
import numpy as np
from jinja2 import Undefined


def number_format(value, decimals=2):
    """
    数値をフォーマット（NaN/None/Undefined対応）

    Parameters
    ----------
    value : float or None
        数値
    decimals : int
        小数点以下の桁数

    Returns
    -------
    str
        フォーマットされた数値。NaN/None/Undefinedの場合は'N/A'

    Examples
    --------
    >>> number_format(12.345, 2)
    '12.35'
    >>> number_format(None, 2)
    'N/A'
    >>> number_format(float('nan'), 2)
    'N/A'
    """
    if isinstance(value, Undefined) or value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    return f"{value:.{decimals}f}"


def format_percent(value, decimals=1):
    """
    パーセント表示にフォーマット

    Parameters
    ----------
    value : float
        0-1の値（例: 0.753 → 75.3%）または0-100の値
    decimals : int
        小数点以下の桁数

    Returns
    -------
    str
        フォーマットされたパーセント表示

    Examples
    --------
    >>> format_percent(0.753, 1)
    '75.3%'
    >>> format_percent(75.3, 1)
    '75.3%'
    """
    if isinstance(value, Undefined) or value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'

    # 0-1の範囲の場合は100倍する
    if 0 <= value <= 1:
        value = value * 100

    return f"{value:.{decimals}f}%"


def format_db(value, decimals=2, with_sign=False):
    """
    dB単位でフォーマット

    Parameters
    ----------
    value : float
        dB値
    decimals : int
        小数点以下の桁数
    with_sign : bool
        常に符号を表示するか

    Returns
    -------
    str
        フォーマットされたdB値

    Examples
    --------
    >>> format_db(2.5, 2)
    '2.50 dB'
    >>> format_db(2.5, 2, with_sign=True)
    '+2.50 dB'
    """
    if isinstance(value, Undefined) or value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'

    sign = '+' if with_sign and value > 0 else ''
    return f"{sign}{value:.{decimals}f} dB"


def format_hz(value, decimals=2):
    """
    Hz単位でフォーマット

    Parameters
    ----------
    value : float
        周波数値
    decimals : int
        小数点以下の桁数

    Returns
    -------
    str
        フォーマットされた周波数値

    Examples
    --------
    >>> format_hz(10.5, 2)
    '10.50 Hz'
    """
    if isinstance(value, Undefined) or value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    return f"{value:.{decimals}f} Hz"


def format_timestamp(value):
    """
    タイムスタンプを秒精度に整形

    Parameters
    ----------
    value : datetime, pd.Timestamp, str, or None
        タイムスタンプ

    Returns
    -------
    str
        フォーマットされたタイムスタンプ（YYYY-MM-DD HH:MM:SS）

    Examples
    --------
    >>> format_timestamp(datetime(2025, 1, 13, 10, 30, 45))
    '2025-01-13 10:30:45'
    """
    if isinstance(value, Undefined) or value is None:
        return 'N/A'
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    if hasattr(value, 'strftime'):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    return str(value)


def format_duration(seconds):
    """
    秒を分表示に変換

    Parameters
    ----------
    seconds : float or None
        秒数

    Returns
    -------
    str
        フォーマットされた分数（小数点1桁）

    Examples
    --------
    >>> format_duration(1800)
    '30.0 分'
    >>> format_duration(None)
    'N/A'
    """
    if isinstance(seconds, Undefined) or seconds is None:
        return 'N/A'
    try:
        minutes = float(seconds) / 60.0
        return f"{minutes:.1f} 分"
    except (TypeError, ValueError):
        return 'N/A'


def format_change(value, unit='', decimals=1, positive_is_good=True):
    """
    変化量をフォーマット（良い変化は太字）

    Parameters
    ----------
    value : float
        変化量
    unit : str
        単位（'%', 'dB', 'Hz'など）
    decimals : int
        小数点以下の桁数
    positive_is_good : bool
        プラスが良い変化かどうか

    Returns
    -------
    str
        フォーマットされた変化量（良い変化は太字）

    Examples
    --------
    >>> format_change(2.5, '%', 1, True)
    '**+2.5%**'
    >>> format_change(-1.3, '%', 1, False)
    '**-1.3%**'
    """
    if isinstance(value, Undefined) or value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    if value == 0:
        return f"±0{unit}"

    sign = '+' if value > 0 else ''
    formatted = f"{sign}{value:.{decimals}f}{unit}"

    # 良い変化の判定
    is_good = (value > 0 and positive_is_good) or (value < 0 and not positive_is_good)

    if is_good:
        return f"**{formatted}**"
    else:
        return formatted


def df_to_markdown(df, floatfmt='.2f', index=True, standardize_columns=False):
    """
    DataFrameをMarkdown表形式に変換

    Parameters
    ----------
    df : pd.DataFrame or None
        DataFrame
    floatfmt : str
        浮動小数点のフォーマット
    index : bool
        インデックスを表示するか
    standardize_columns : bool
        カラム名を標準形式（Metric/Value/Unit）に変換するか

    Returns
    -------
    str
        Markdown形式の表

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1.234, 2.567]})
    >>> df_to_markdown(df, floatfmt='.2f', index=False)
    '|   A |\n|----:|\n| 1.23|\n| 2.57|'
    >>> df = pd.DataFrame({'指標': ['A'], '値': [1.23], '単位': ['ms']})
    >>> df_to_markdown(df, index=False, standardize_columns=True)
    '| Metric | Value | Unit |\n...'
    """
    if df is None:
        return ''
    # DataFrameでない場合は変換
    if not isinstance(df, pd.DataFrame):
        return ''
    if df.empty:
        return ''

    # カラム名の標準化
    if standardize_columns:
        # 標準カラム名マッピング（日本語 → 英語）
        column_mapping = {
            '指標': 'Metric',
            'Metric': 'Metric',
            '値': 'Value',
            'Value': 'Value',
            '単位': 'Unit',
            'Unit': 'Unit',
        }

        # カラム名をリネーム
        df = df.rename(columns=column_mapping)

    return df.to_markdown(floatfmt=floatfmt, index=index)


def format_score(score, max_score=100, decimals=1):
    """
    スコアをフォーマット

    Parameters
    ----------
    score : float
        スコア値
    max_score : float
        最大スコア（デフォルト100）
    decimals : int
        小数点以下の桁数

    Returns
    -------
    str
        フォーマットされたスコア

    Examples
    --------
    >>> format_score(75.234, 100, 1)
    '75.2/100'
    """
    if isinstance(score, Undefined) or score is None or (isinstance(score, float) and np.isnan(score)):
        return 'N/A'
    return f"{score:.{decimals}f}/{max_score}"
