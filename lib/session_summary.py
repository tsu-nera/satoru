"""
セッションサマリー生成モジュール

脳波解析結果を1行のDataFrameにまとめ、日次集計用のCSV形式で出力する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


@dataclass
class SessionSummaryResult:
    """セッションサマリー結果を保持するデータクラス"""

    summary: pd.DataFrame  # 1行のDataFrame（CSV互換）
    metadata: dict


def extract_metric_value(
    stats_df: pd.DataFrame,
    metric_name: str,
    value_col: str = 'Value'
) -> Optional[float]:
    """
    統計DataFrameから指標値を抽出

    Parameters
    ----------
    stats_df : pd.DataFrame
        統計情報DataFrame（'Metric' または '指標' カラムを持つ）
    metric_name : str
        抽出する指標名（英語）
    value_col : str
        値のカラム名（デフォルト: 'Value'）

    Returns
    -------
    Optional[float]
        指標値（見つからない場合はNone）
    """
    if stats_df is None or stats_df.empty:
        return None

    # カラム名を正規化（後方互換性のため）
    metric_col = 'Metric' if 'Metric' in stats_df.columns else '指標'
    if value_col not in stats_df.columns:
        value_col = '値' if '値' in stats_df.columns else 'Value'

    # メトリック名の正規化マッピング（英語→日本語の後方互換）
    metric_aliases = {
        'Mean': ['Mean', '平均', '平均値', '平均SE', '平均FAA'],
        'Median': ['Median', '中央値'],
        'Std Dev': ['Std Dev', '標準偏差'],
        'First Half Mean': ['First Half Mean', '前半平均'],
        'Second Half Mean': ['Second Half Mean', '後半平均'],
        'Mean FAA': ['Mean FAA', '平均FAA'],
    }

    # エイリアスを使って検索
    search_names = metric_aliases.get(metric_name, [metric_name])
    for alias in search_names:
        matched = stats_df[stats_df[metric_col] == alias]
        if not matched.empty:
            return matched[value_col].iloc[0]

    return None


def format_timestamp_for_csv(value) -> Optional[str]:
    """Datetime表示をCSV用に整形"""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    if hasattr(value, 'strftime'):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    return str(value)


def seconds_to_minutes(value) -> Optional[float]:
    """秒を分表示用に変換"""
    try:
        return float(value) / 60.0
    except (TypeError, ValueError):
        return None


def generate_session_summary(
    data_path: Path,
    results: Dict[str, Any]
) -> SessionSummaryResult:
    """
    解析結果から日次集計用のサマリーを生成

    Parameters
    ----------
    data_path : Path
        入力CSVファイルパス
    results : dict
        分析結果を格納した辞書（run_full_analysis()の戻り値）

    Returns
    -------
    SessionSummaryResult
        - summary: CSV出力可能な1行DataFrame
        - metadata: 補足情報
    """
    info = results.get('data_info', {})

    # 基本情報
    summary = {
        'session_date': info.get('start_time').strftime('%Y-%m-%d') if info.get('start_time') else None,
        'session_start_time': format_timestamp_for_csv(info.get('start_time')),
        'session_end_time': format_timestamp_for_csv(info.get('end_time')),
        'session_duration_min': seconds_to_minutes(info.get('duration_sec')),
        'data_file': data_path.name,
    }

    # HSI品質
    if 'hsi_stats' in results:
        hsi_data = results['hsi_stats']
        summary['hsi_quality_score'] = hsi_data.get('overall_quality')
        summary['hsi_good_ratio'] = hsi_data.get('good_ratio', 0.0)

    # バンド比率（新しい縦長形式に対応）
    if 'band_ratios_stats' in results:
        ratios_df = results['band_ratios_stats']

        # Alpha/Beta（リラックス度）
        alpha_beta_mean = extract_metric_value(ratios_df, 'Alpha/Beta Mean')
        if alpha_beta_mean is None:
            # 旧形式（横長）への後方互換
            old_row = ratios_df[ratios_df.get('Metric', ratios_df.get('指標', pd.Series())) == 'Alpha/Beta']
            if not old_row.empty:
                alpha_beta_mean = old_row.get('Value', old_row.get('平均値', pd.Series())).iloc[0]

        summary['alpha_beta_ratio_mean'] = alpha_beta_mean

        # Beta/Theta（集中度）
        beta_theta_mean = extract_metric_value(ratios_df, 'Beta/Theta Mean')
        if beta_theta_mean is None:
            old_row = ratios_df[ratios_df.get('Metric', ratios_df.get('指標', pd.Series())) == 'Beta/Theta']
            if not old_row.empty:
                beta_theta_mean = old_row.get('Value', old_row.get('平均値', pd.Series())).iloc[0]

        summary['beta_theta_ratio_mean'] = beta_theta_mean

        # Theta/Alpha（瞑想深度）
        theta_alpha_mean = extract_metric_value(ratios_df, 'Theta/Alpha Mean')
        if theta_alpha_mean is None:
            old_row = ratios_df[ratios_df.get('Metric', ratios_df.get('指標', pd.Series())) == 'Theta/Alpha']
            if not old_row.empty:
                theta_alpha_mean = old_row.get('Value', old_row.get('平均値', pd.Series())).iloc[0]

        summary['theta_alpha_ratio_mean'] = theta_alpha_mean

    # Fmθ
    if 'frontal_theta_stats' in results:
        fmtheta_df = results['frontal_theta_stats']
        summary['fmtheta_mean_uv2'] = extract_metric_value(fmtheta_df, 'Mean')
        summary['fmtheta_std_uv2'] = extract_metric_value(fmtheta_df, 'Std Dev')

    # IAF
    if 'iaf' in results:
        iaf_data = results['iaf']
        summary['iaf_mean_hz'] = iaf_data['value']
        summary['iaf_std_hz'] = iaf_data['std']

    # Spectral Entropy
    if 'spectral_entropy_stats' in results:
        se_df = results['spectral_entropy_stats']
        summary['spectral_entropy_mean'] = extract_metric_value(se_df, 'Mean')

    # FAA
    if 'faa_stats' in results:
        faa_df = results['faa_stats']
        summary['faa_mean'] = extract_metric_value(faa_df, 'Mean FAA')

    # 総合スコア
    if 'session_score' in results:
        summary['session_score'] = results['session_score']
        summary['session_level'] = results['session_level']

    # ピークパフォーマンス
    if 'segment_peak_range' in results:
        summary['peak_time_range'] = results['segment_peak_range']
        summary['peak_score'] = results.get('segment_peak_score')

    # メタデータ
    metadata = {
        'generated_from': str(data_path),
        'n_metrics': len([v for v in summary.values() if v is not None]),
    }

    return SessionSummaryResult(
        summary=pd.DataFrame([summary]),
        metadata=metadata
    )
