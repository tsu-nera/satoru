"""
セッションログCSV管理

瞑想セッションの主要指標をCSVに記録・管理する機能を提供します。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def log_session_metrics(
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
    CSVスキーマ（12カラム）:
    - date: 日付 (YYYY-MM-DD)
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
    """
    # デフォルトのCSVパス
    if csv_path is None:
        # プロジェクトルートを取得
        lib_dir = Path(__file__).parent
        project_root = lib_dir.parent
        log_dir = project_root / 'issues' / '007_daily_dashboard'
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / 'session_log.csv'

    # データ抽出
    info = results.get('data_info', {})
    mean_metrics = results.get('mean_metrics', {})
    best_metrics = results.get('best_metrics', {})

    # 日付（記録開始時刻から）
    start_time = info.get('start_time')
    if start_time is None:
        raise ValueError('results["data_info"]["start_time"]が見つかりません')

    date_str = start_time.strftime('%Y-%m-%d')

    # 計測時間（分）
    duration_sec = info.get('duration_sec')
    duration_min = duration_sec / 60.0 if duration_sec is not None else np.nan

    # 新しいレコード
    new_record = {
        'date': date_str,
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
    }

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
