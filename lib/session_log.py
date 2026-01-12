"""
セッションログCSV管理

瞑想セッションの主要指標をCSVに記録・管理する機能を提供します。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build


def _get_column_headers() -> List[str]:
    """
    セッションログのカラムヘッダーを取得。

    Returns
    -------
    list of str
        14個のカラム名のリスト
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
    ]


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
        セッションデータの辞書。以下のキーを含む：
        - timestamp: セッション開始時刻文字列 (YYYY-MM-DD HH:MM:SS)
        - duration_min: 計測時間（分）
        - fm_theta_mean, fm_theta_best, ...（全12カラム分）

    Raises
    ------
    ValueError
        start_timeが見つからない場合
    """
    # データ抽出
    info = results.get('data_info', {})
    mean_metrics = results.get('mean_metrics', {})
    best_metrics = results.get('best_metrics', {})

    # タイムスタンプ（記録開始時刻）
    start_time = info.get('start_time')
    if start_time is None:
        raise ValueError('results["data_info"]["start_time"]が見つかりません')

    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    # 計測時間（分）
    duration_sec = info.get('duration_sec')
    duration_min = duration_sec / 60.0 if duration_sec is not None else np.nan

    # セッションデータ
    return {
        'timestamp': timestamp_str,
        'duration_min': duration_min,
        'fm_theta_mean': mean_metrics.get('fm_theta_mean', np.nan),
        'fm_theta_best': best_metrics.get('fm_theta_best', np.nan),
        'iaf_mean': mean_metrics.get('iaf_mean', np.nan),
        'iaf_best': best_metrics.get('iaf_best', np.nan),
        'alpha_mean': mean_metrics.get('alpha_mean', np.nan),
        'alpha_best': best_metrics.get('alpha_best', np.nan),
        'beta_mean': mean_metrics.get('beta_mean', np.nan),
        'beta_best': best_metrics.get('beta_best', np.nan),
        'theta_alpha_mean': mean_metrics.get('theta_alpha_mean', np.nan),
        'theta_alpha_best': best_metrics.get('theta_alpha_best', np.nan),
        'hrv_mean': mean_metrics.get('hrv_mean', np.nan),
        'hrv_best': best_metrics.get('hrv_best', np.nan),
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
    csv_path : Path, optional
        出力先CSVファイルパス。指定しない場合は
        'issues/007_daily_dashboard/session_log.csv' を使用。

    Returns
    -------
    Path
        書き込んだCSVファイルのパス

    Notes
    -----
    CSVスキーマ（14カラム）:
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

    # 新しい行のデータ（文字列にフォーマット）
    new_row = []
    for key in _get_column_headers():
        value = session_data[key]
        if key == 'timestamp':
            new_row.append(value)
        elif np.isnan(value):
            new_row.append('')
        else:
            new_row.append(f'{value:.3f}')

    # スプレッドシートの既存データを取得
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f'{sheet_name}!A:N',
        ).execute()
        values = result.get('values', [])
    except Exception as e:
        # シートが存在しない場合はヘッダーを作成
        values = []

    # ヘッダーが存在しない場合は作成
    if not values:
        # ヘッダーを最初の行に書き込み
        header_body = {'values': [_get_column_headers()]}
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f'{sheet_name}!A1:N1',
            valueInputOption='USER_ENTERED',
            body=header_body,
        ).execute()
        values = [_get_column_headers()]

    # 新しい行を追加
    next_row = len(values) + 1
    range_name = f'{sheet_name}!A{next_row}:N{next_row}'

    # データを書き込み
    body = {'values': [new_row]}
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption='USER_ENTERED',
        body=body,
    ).execute()
