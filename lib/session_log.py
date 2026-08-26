"""
セッションログCSV管理

瞑想セッションの主要指標をCSVに記録・管理する機能を提供します。

保存先は `logs/session_log.csv`（git管理）。カラムは固定ではなく、
`_extract_session_data()` が返すキーがそのまま列になります。
新しい指標を追加したいときは `_extract_session_data()` にキーを足すだけでよく、
既存行はその列がNaNのまま残ります。逆にキーを削っても既存列は保持されます。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

#: 既知カラムの推奨並び順。ここに無いキーは末尾に回る（列の増減を妨げない）。
CANONICAL_COLUMNS: List[str] = [
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
    'delta_rel_pct',
    'theta_rel_pct',
    'alpha_rel_pct',
    'artifact_reject_pct',
    'breathing_rate_bpm',
    'breathing_rate_std',
    'respiratory_period_s',
    'rsa_amplitude_ms',
    'rsa_band_power_ms2',
    'sd1',
    'note',
]


#: CSV往復でfloat/文字列に化けたbool値を戻すための対応表。
# True/1/1.0 はPythonでは同一キーなので True/False のみで数値も拾える。
_BOOL_MAP = {
    True: True, False: False,
    'True': True, 'False': False,
    'TRUE': True, 'FALSE': False,
}


def default_log_path() -> Path:
    """セッションログCSVの既定パス（`logs/session_log.csv`）。"""
    return Path(__file__).parent.parent / 'logs' / 'session_log.csv'


def _order_columns(columns) -> List[str]:
    """CANONICAL_COLUMNS の順を優先しつつ、未知の列を末尾に並べる。"""
    known = [c for c in CANONICAL_COLUMNS if c in columns]
    unknown = sorted(c for c in columns if c not in CANONICAL_COLUMNS)
    return known + unknown


def _hrv_stat(results: Dict, metric: str) -> float:
    """`results['hrv_stats']`（Domain/Metric/Value/Unit の縦持ち）から1指標を取り出す。"""
    stats = results.get('hrv_stats')
    if stats is None or getattr(stats, 'empty', True):
        return float('nan')
    matched = stats.loc[stats['Metric'] == metric, 'Value']
    return float(matched.iloc[0]) if len(matched) else float('nan')


def _extract_session_data(results: Dict) -> Dict:
    """
    分析結果からセッションログ用のデータを抽出する。

    ここで返したキーがそのままCSVの列になる。指標を追加するときは
    キーを足すだけでよい（既存CSVは自動でその列が追加され、過去行はNaN）。

    Parameters
    ----------
    results : dict
        分析結果を格納した辞書

    Returns
    -------
    dict
        セッションデータの辞書

    Raises
    ------
    ValueError
        start_timeが見つからない場合
    """
    info = results.get('data_info', {})
    mean_metrics = results.get('mean_metrics', {})
    best_metrics = results.get('best_metrics', {})
    # 非周期成分（1/f）はmean_metrics/best_metricsに載らないため、
    # resultsから直接読む（追加し忘れるとサイレントに欠落するため注意）。
    aperiodic_info = results.get('aperiodic', {})

    start_time = info.get('start_time')
    if start_time is None:
        raise ValueError('results["data_info"]["start_time"]が見つかりません')

    duration_sec = info.get('duration_sec')
    duration_min = duration_sec / 60.0 if duration_sec is not None else float('nan')

    alpha_peak = aperiodic_info.get('alpha_peak')
    theta_peak = aperiodic_info.get('theta_peak')

    artifact_summary = results.get('artifact_summary') or {}
    rejected_ratio = artifact_summary.get('rejected_ratio')

    respiration = results.get('respiration_result')

    return {
        'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
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
        # δ相対パワーと振幅除外率は低周波ドリフト混入の判定に使う
        # （両者が並走して高いときはδ/θ由来の指標を割り引いて読む）。
        'delta_rel_pct': mean_metrics.get('delta_rel_pct', float('nan')),
        'theta_rel_pct': mean_metrics.get('theta_rel_pct', float('nan')),
        'alpha_rel_pct': mean_metrics.get('alpha_rel_pct', float('nan')),
        'artifact_reject_pct': (
            rejected_ratio * 100 if rejected_ratio is not None else float('nan')
        ),
        'breathing_rate_bpm': (
            respiration.breathing_rate if respiration is not None else float('nan')
        ),
        'breathing_rate_std': (
            respiration.breathing_rate_std if respiration is not None else float('nan')
        ),
        # 呼吸周期は breathing_rate の逆数だが、超低速呼吸の議論では秒で見るほうが早い
        'respiratory_period_s': (
            60.0 / respiration.breathing_rate
            if respiration is not None and respiration.breathing_rate > 0
            else float('nan')
        ),
        'rsa_amplitude_ms': (
            respiration.rsa_amplitude_mean if respiration is not None else float('nan')
        ),
        # 呼吸追従帯のパワー。超低速呼吸では固定HF帯が空になるため、
        # 副交感神経活動の評価はこちらとRSA振幅・SD1で行う。
        'rsa_band_power_ms2': (results.get('rsa_band') or {}).get('power', float('nan')),
        'sd1': _hrv_stat(results, 'SD1'),
    }


def read_session_log(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    セッションログCSVを読み込む。

    Parameters
    ----------
    csv_path : Path, optional
        読み込むCSVパス。省略時は `logs/session_log.csv`。

    Returns
    -------
    pd.DataFrame
        timestamp昇順のDataFrame。ファイルが無い場合は空のDataFrame。
    """
    if csv_path is None:
        csv_path = default_log_path()

    if not Path(csv_path).exists():
        return pd.DataFrame(columns=['timestamp'])

    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def write_to_csv(
    results: Dict,
    csv_path: Optional[Path] = None,
) -> Path:
    """
    セッションログCSVにセッションデータを upsert する。

    同一 timestamp の行が既にある場合は追記せず上書きする。分析の再実行で
    行が重複しないようにするため（ローカル実行が主経路のため再実行は頻繁に起きる）。

    列は固定ではない。新しいキーが来れば列が増え、過去行はNaNになる。
    既存CSVにしかない列も保持される。

    Parameters
    ----------
    results : dict
        分析結果を格納した辞書。以下のキーを参照する：
        - 'data_info': {'start_time': pd.Timestamp, 'duration_sec': float}
        - 'mean_metrics' / 'best_metrics' / 'aperiodic'
        - 'artifact_summary' / 'respiration_result'（あれば）
    csv_path : Path, optional
        出力先CSVパス。省略時は `logs/session_log.csv`。

    Returns
    -------
    Path
        書き込んだCSVファイルのパス
    """
    if csv_path is None:
        csv_path = default_log_path()
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_record = _extract_session_data(results)
    df_new = pd.DataFrame([new_record])

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'timestamp' in df.columns:
            # 同一timestampの既存行を落としてから追記する（upsert）
            df = df[df['timestamp'] != new_record['timestamp']]
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    # bool列はconcatでfloat化する（既存行のNaNと混ざるため）。
    # float_formatが効いて True が 0.000/1.000 になるのを防ぐ。
    for key, value in new_record.items():
        if isinstance(value, bool):
            df[key] = df[key].map(_BOOL_MAP).astype('boolean')

    df = df[_order_columns(df.columns)]
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    df.to_csv(csv_path, index=False, float_format='%.3f')

    return csv_path
