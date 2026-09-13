"""
タップ打刻ログローダー

`lib/recorders/tap_log.py` が出力するタップ打刻CSV（マインドワンダリングの
自覚を記録したもの）の読み込みと、Museセッションとのファイルマッチングを行う。
"""

import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

#: タップログファイル名の正規表現（例: taps_2026-01-10--16-08-53.csv）
_TAP_LOG_FILENAME_RE = re.compile(r'^taps_(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})\.csv$')

#: ファイル名のタイムスタンプ部分をパースするフォーマット
_TAP_LOG_TIMESTAMP_FORMAT = '%Y-%m-%d--%H-%M-%S'

#: 必須列（欠けていたらValueError）
_REQUIRED_COLUMNS = ['seq', 'event', 'client_ts', 'server_ts', 'elapsed_s']


def load_tap_log_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    タップ打刻CSVを読み込み、naive localな `TimeStamp` 列を付与して返す。

    `server_ts` はtz付き（例: `+09:00`）だが、Muse側の `TimeStamp` はnaive
    localのため、記録時の壁時計を保ったままtzを落として揃える。tz-aware
    Seriesへの `tz_convert(None)` / `tz_localize(None)` はUTCへ変換してから
    tzを落とすため、そのまま使うとオフセット分（例: 9時間）ずれる。そのため
    先頭行のオフセットを一度取り出し、UTC変換後の時刻へ明示的に足し戻してから
    tzを落とす。

    Parameters
    ----------
    csv_path : str or Path
        タップ打刻CSVのパス

    Returns
    -------
    pd.DataFrame
        元の列（`seq`, `event`, `client_ts`, `server_ts`, `elapsed_s`）に加えて
        `TimeStamp`（naive local）を持つDataFrame。
        `df.attrs['session_start']` に `start` 行の `TimeStamp` を保存する。

    Raises
    ------
    ValueError
        必須列が欠けている場合
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'タップログに必須列が欠けています: {missing}')

    parsed = pd.to_datetime(df['server_ts'], format='ISO8601', utc=True)

    # 先頭行のserver_ts文字列からutcoffsetを取り出す（tzハードコード回避）
    first_ts = pd.Timestamp(df['server_ts'].iloc[0])
    offset = first_ts.utcoffset()

    df['TimeStamp'] = (parsed + offset).dt.tz_localize(None)

    start_rows = df.loc[df['event'] == 'start', 'TimeStamp']
    session_start = start_rows.iloc[0] if len(start_rows) else None
    df.attrs['session_start'] = session_start

    return df


def find_tap_log_for_session(
    tap_dir: Union[str, Path],
    session_start,
    tolerance_minutes: float = 5.0,
) -> Optional[Path]:
    """
    Museセッションの開始時刻に最も近いタップログファイルを探す。

    タップアプリはMuseとは別に手動で起動するため、開始時刻が数分ずれうる。
    SelfLoopsの「完全一致→同一分」方式より広く、`tolerance_minutes` 以内で
    最も近いファイルを1つ返す。

    Parameters
    ----------
    tap_dir : str or Path
        タップログファイルが置かれているディレクトリ
    session_start : datetime-like
        Museセッションの開始時刻
    tolerance_minutes : float, default 5.0
        許容する時刻差（分）

    Returns
    -------
    Path or None
        最も近いタップログファイルのパス。許容範囲内に無ければ `None`。
    """
    tap_dir = Path(tap_dir)
    if not tap_dir.exists():
        return None

    session_start_ts = pd.Timestamp(session_start)
    tolerance = pd.Timedelta(minutes=tolerance_minutes)

    best_path = None
    best_diff = None

    for path in tap_dir.iterdir():
        match = _TAP_LOG_FILENAME_RE.match(path.name)
        if not match:
            continue

        try:
            file_ts = pd.Timestamp(pd.to_datetime(match.group(1), format=_TAP_LOG_TIMESTAMP_FORMAT))
        except ValueError:
            continue

        diff = abs(file_ts - session_start_ts)
        if diff > tolerance:
            continue

        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_path = path

    return best_path
