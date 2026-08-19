"""
セッションログCSV管理

瞑想セッションの主要指標をCSVに記録・管理する機能を提供します。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build


def _get_column_headers() -> List[str]:
    """
    セッションログのカラムヘッダーを取得。

    Returns
    -------
    list of str
        20個のカラム名のリスト
    """
    return [
        'timestamp',
        'duration_min',
        'fm_theta_mean',
        'fm_theta_best',
        'iaf_mean',
        'iaf_best',
        'alpha_mean',
        'alpha_best',
        'beta_mean',
        'beta_best',
        'theta_alpha_mean',
        'theta_alpha_best',
        'hrv_mean',
        'hrv_best',
        'aperiodic_exponent',
        'aperiodic_offset',
        'alpha_osc_db',
        'theta_osc_db',
        'alpha_cf_hz',
        'theta_peak_detected',
    ]


def _column_letter(n: int) -> str:
    """
    1始まりの列番号をA1記法の列文字に変換する（26列を超えるAA, AB, ...にも対応）。

    Parameters
    ----------
    n : int
        1始まりの列番号（1=A, 2=B, ..., 27=AA）。

    Returns
    -------
    str
        A1記法の列文字。
    """
    letters = ''
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _header_last_column() -> str:
    """`_get_column_headers()` の列数からA1記法の最終列文字を求める。"""
    return _column_letter(len(_get_column_headers()))


def _extract_session_data(results: Dict) -> Dict:
    """
    分析結果からセッションログ用のデータを抽出する。

    Parameters
    ----------
    results : dict
        分析結果を格納した辞書

    Returns
    -------
    dict
        セッションデータの辞書。`_get_column_headers()` と同じキー・順序を持つ。

    Raises
    ------
    ValueError
        start_timeが見つからない場合
    """
    # データ抽出
    info = results.get('data_info', {})
    mean_metrics = results.get('mean_metrics', {})
    best_metrics = results.get('best_metrics', {})
    # 非周期成分（1/f）はmean_metrics/best_metricsに載らないため、
    # resultsから直接読む（追加し忘れるとサイレントに欠落するため注意）。
    aperiodic_info = results.get('aperiodic', {})

    # タイムスタンプ（記録開始時刻）
    start_time = info.get('start_time')
    if start_time is None:
        raise ValueError('results["data_info"]["start_time"]が見つかりません')

    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    # 計測時間（分）
    duration_sec = info.get('duration_sec')
    duration_min = duration_sec / 60.0 if duration_sec is not None else float('nan')

    alpha_peak = aperiodic_info.get('alpha_peak')
    theta_peak = aperiodic_info.get('theta_peak')

    # セッションデータ
    return {
        'timestamp': timestamp_str,
        'duration_min': duration_min,
        'fm_theta_mean': mean_metrics.get('fm_theta_mean', float('nan')),
        'fm_theta_best': best_metrics.get('fm_theta_best', float('nan')),
        'iaf_mean': mean_metrics.get('iaf_mean', float('nan')),
        'iaf_best': best_metrics.get('iaf_best', float('nan')),
        'alpha_mean': mean_metrics.get('alpha_mean', float('nan')),
        'alpha_best': best_metrics.get('alpha_best', float('nan')),
        'beta_mean': mean_metrics.get('beta_mean', float('nan')),
        'beta_best': best_metrics.get('beta_best', float('nan')),
        'theta_alpha_mean': mean_metrics.get('theta_alpha_mean', float('nan')),
        'theta_alpha_best': best_metrics.get('theta_alpha_best', float('nan')),
        'hrv_mean': mean_metrics.get('hrv_mean', float('nan')),
        'hrv_best': best_metrics.get('hrv_best', float('nan')),
        'aperiodic_exponent': aperiodic_info.get('exponent', float('nan')),
        'aperiodic_offset': aperiodic_info.get('offset', float('nan')),
        'alpha_osc_db': aperiodic_info.get('alpha_osc_db', float('nan')),
        'theta_osc_db': aperiodic_info.get('theta_osc_db', float('nan')),
        'alpha_cf_hz': alpha_peak['center_hz'] if alpha_peak is not None else float('nan'),
        'theta_peak_detected': theta_peak is not None,
    }


def write_to_csv(
    results: Dict,
    csv_path: Optional[Path] = None,
) -> Path:
    """
    セッションログCSVにセッションデータを追記する。

    Parameters
    ----------
    results : dict
        分析結果を格納した辞書。以下のキーを含む必要がある：
        - 'data_info': {'start_time': pd.Timestamp, 'duration_sec': float}
        - 'mean_metrics': {'fm_theta_mean': float, 'iaf_mean': float, ...}
        - 'best_metrics': {'fm_theta_best': float, 'iaf_best': float, ...}
        - 'aperiodic': {'exponent': float, 'offset': float, ...}（あれば）
    csv_path : Path, optional
        出力先CSVファイルパス。指定しない場合は
        'issues/007_daily_dashboard/session_log.csv' を使用。

    Returns
    -------
    Path
        書き込んだCSVファイルのパス

    Notes
    -----
    CSVスキーマ（20カラム、`_get_column_headers()` が唯一の真実の源）:
    - timestamp: セッション開始時刻 (YYYY-MM-DD HH:MM:SS)
    - duration_min: 計測時間（分）
    - fm_theta_mean: Fmθ平均 (dB)
    - fm_theta_best: Fmθ最良値 (dB)
    - iaf_mean: IAF平均 (Hz)
    - iaf_best: IAF最良値 (Hz)
    - alpha_mean: Alpha平均 (dB)
    - alpha_best: Alpha最良値 (dB)
    - beta_mean: Beta平均 (dB)
    - beta_best: Beta最小値 (dB)
    - theta_alpha_mean: θ/α比平均 (ratio)
    - theta_alpha_best: θ/α比最良値 (ratio)
    - hrv_mean: HRV (RMSSD) 平均 (ms)
    - hrv_best: HRV (RMSSD) 最良値 (ms)
    - aperiodic_exponent: 非周期成分exponent
    - aperiodic_offset: 非周期成分offset
    - alpha_osc_db: α振動性パワー (dB)
    - theta_osc_db: θ振動性パワー (dB)
    - alpha_cf_hz: specparam由来のαピーク中心周波数 (Hz)
    - theta_peak_detected: θピークが検出されたか (bool)
    """
    # デフォルトのCSVパス
    if csv_path is None:
        lib_dir = Path(__file__).parent
        project_root = lib_dir.parent
        log_dir = project_root / 'issues' / '007_daily_dashboard'
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / 'session_log.csv'

    # データ抽出
    new_record = _extract_session_data(results)

    # CSVの存在確認
    if csv_path.exists():
        # 既存CSVに追記
        df = pd.read_csv(csv_path)
        df_new = pd.DataFrame([new_record])
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        # 新規作成
        df = pd.DataFrame([new_record])

    # CSV保存
    df.to_csv(csv_path, index=False, float_format='%.3f')

    return csv_path


def write_to_google_sheets(
    results: Dict,
    spreadsheet_id: str,
    credentials_path: Optional[Path] = None,
    sheet_name: str = 'Muse',
) -> None:
    """
    セッションデータをGoogle Spreadsheetsに書き込む。

    Parameters
    ----------
    results : dict
        分析結果を格納した辞書。log_session_metrics()と同じ形式。
    spreadsheet_id : str
        書き込み先のGoogle SpreadsheetのID
    credentials_path : Path, optional
        サービスアカウントJSONファイルのパス。
        指定しない場合は 'private/gdrive-creds.json' を使用。
        環境変数 GDRIVE_CREDS_JSON が設定されている場合はそちらを優先。
    sheet_name : str, default='シート1'
        書き込み先のシート名

    Notes
    -----
    - スプレッドシートは事前にサービスアカウントと共有されている必要があります
    - データは既存データの末尾に追記されます
    - 最初の行がヘッダー行として扱われます
    - GitHub Actionsでは環境変数 GDRIVE_CREDS_JSON から認証情報を読み込みます
    - 読み取り・ヘッダー書き込み・追記のレンジは `_get_column_headers()` の列数から
      動的に算出する（列数が変わった際にA:Nのようなハードコードとズレるのを防ぐ）。
    """
    import json

    # Sheets APIスコープ
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    # 環境変数から認証情報を取得（GitHub Actions用）
    creds_json = os.environ.get('GDRIVE_CREDS_JSON')
    if creds_json:
        # JSON文字列から認証情報を作成
        credentials_info = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info, scopes=SCOPES
        )
    else:
        # ローカル実行用：ファイルから認証情報を取得
        if credentials_path is None:
            lib_dir = Path(__file__).parent
            project_root = lib_dir.parent
            credentials_path = project_root / 'private' / 'gdrive-creds.json'

        if not credentials_path.exists():
            raise FileNotFoundError(f'認証情報ファイルが見つかりません: {credentials_path}')

        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path),
            scopes=SCOPES,
        )

    # Sheets APIサービス構築
    service = build('sheets', 'v4', credentials=credentials)

    # データ抽出
    session_data = _extract_session_data(results)
    last_col = _header_last_column()

    # 新しい行のデータ（文字列にフォーマット）
    new_row = []
    for key in _get_column_headers():
        value = session_data[key]
        if key == 'timestamp':
            new_row.append(value)
        elif isinstance(value, bool):
            new_row.append('TRUE' if value else 'FALSE')
        elif pd.isna(value):
            new_row.append('')
        else:
            new_row.append(f'{value:.3f}')

    # スプレッドシートの既存データを取得
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f'{sheet_name}!A:{last_col}',
        ).execute()
        values = result.get('values', [])
    except Exception:
        # シートが存在しない場合はヘッダーを作成
        values = []

    # ヘッダーが存在しない場合は作成
    if not values:
        # ヘッダーを最初の行に書き込み
        header_body = {'values': [_get_column_headers()]}
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f'{sheet_name}!A1:{last_col}1',
            valueInputOption='USER_ENTERED',
            body=header_body,
        ).execute()
        values = [_get_column_headers()]

    # 新しい行を追加
    next_row = len(values) + 1
    range_name = f'{sheet_name}!A{next_row}:{last_col}{next_row}'

    # データを書き込み
    body = {'values': [new_row]}
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption='USER_ENTERED',
        body=body,
    ).execute()
