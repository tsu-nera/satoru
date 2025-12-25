"""
高レベルEEG解析ユーティリティ

時間セグメント分析など、レポート生成で利用する追加機能を提供する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from .sensors.eeg.frontal_theta import (
    FrontalThetaResult,
    calculate_frontal_theta,
)
from .sensors.eeg.preprocessing import filter_eeg_quality
from .statistical_dataframe import get_band_power_at_time, get_band_ratio_at_time

if TYPE_CHECKING:
    import mne


# ========================================
# 総合スコア算出の重み定数
# ========================================
MEDITATION_SCORE_WEIGHTS = {
    'fmtheta': 0.3125,          # Frontal Midline Theta（瞑想深度）
    'spectral_entropy': 0.25,   # Spectral Entropy（集中度）
    'theta_alpha_ratio': 0.1875, # θ/α比（瞑想深度）
    'beta_alpha_ratio': 0.125,  # β/α比（覚醒度、低いほどリラックス）
    'iaf_stability': 0.125,     # IAF安定性（周波数特性）
}


@dataclass
class SegmentAnalysisResult:
    """時間セグメント分析の結果を保持する。"""

    segments: pd.DataFrame
    table: pd.DataFrame  # 後方互換性のため残す（metrics_tableと同じ内容）
    normalized: pd.DataFrame
    metadata: Dict[str, object]
    band_power_table: pd.DataFrame  # 新規: バンドパワー詳細テーブル
    metrics_table: pd.DataFrame  # 新規: 比率と特徴指標テーブル

    def to_markdown(self, floatfmt: str = '.3f') -> str:
        """集計表をMarkdown文字列として返す。"""
        return self.table.to_markdown(index=False, floatfmt=floatfmt)


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """0-1レンジへ正規化（定数列は0.5で埋める）。"""
    clean = series.dropna()
    if clean.empty:
        return pd.Series(np.nan, index=series.index)

    min_val = clean.min()
    max_val = clean.max()

    if pd.isna(min_val) or pd.isna(max_val) or np.isclose(max_val, min_val):
        return pd.Series(0.5, index=series.index)

    return (series - min_val) / (max_val - min_val)


def calculate_segment_analysis(
    df_clean: pd.DataFrame,
    fmtheta_series: pd.Series,
    statistical_df: Dict[str, pd.DataFrame],
    segment_minutes: int = 5,
    warmup_minutes: float = 0.0,
    exclude_first_segment: bool = False,
    exclude_last_segment: bool = False,
    smr_series: Optional[pd.Series] = None,
) -> SegmentAnalysisResult:
    """
    セッションを一定時間のセグメントに分割し、主要指標を算出する。

    Parameters
    ----------
    df_clean : pd.DataFrame
        前処理済みMind Monitor形式のデータフレーム。
    fmtheta_series : pd.Series
        Fmθの時系列データ（indexはタイムスタンプ）。
    statistical_df : dict
        create_statistical_dataframe()が返す統計DataFrame辞書。
        必須キー: 'band_powers', 'band_ratios'
    segment_minutes : int, default 5
        セグメント長（分単位）。
    warmup_minutes : float, default 0.0
        セッション開始後の除外期間（分単位）。アーティファクト除去のため。
    exclude_first_segment : bool, default False
        最初のセグメントをスコア計算・ピーク判定から除外（relaxing phase）。
    exclude_last_segment : bool, default False
        最後のセグメントをスコア計算・ピーク判定から除外（post meditation stage）。
    smr_series : pd.Series, optional
        SMR（12-15Hz）の時系列データ（indexはタイムスタンプ）。

    Returns
    -------
    SegmentAnalysisResult
        集計表・正規化値・メタデータを含む時間セグメント分析結果。

    Notes
    -----
    バンドパワー・比率・IAFはstatistical_dfから自動取得されます（MNE Epochsベース）。
    df_cleanのバンドパワー列は使用されません。
    exclude_first/last_segmentはピーク判定・best値計算のみに影響し、
    レポートのテーブルには全セグメントが表示されます。
    """
    # Statistical DFのバリデーション
    required_keys = ['band_powers', 'band_ratios', 'spectral_entropy', 'iaf']
    missing_keys = [k for k in required_keys if k not in statistical_df]
    if missing_keys:
        raise ValueError(f'statistical_dfには{missing_keys}キーが必要です。')

    band_powers_df = statistical_df['band_powers']
    band_ratios_df = statistical_df['band_ratios']
    se_df = statistical_df['spectral_entropy']
    fnirs_df = statistical_df.get('fnirs')
    hr_df = statistical_df.get('hr')
    posture_df = statistical_df.get('posture')

    if 'TimeStamp' not in df_clean.columns:
        raise ValueError('TimeStamp列が存在しません。')
    if segment_minutes <= 0:
        raise ValueError('segment_minutesは正の整数で指定してください。')

    df_clean = df_clean.copy()
    df_clean['TimeStamp'] = pd.to_datetime(df_clean['TimeStamp'], errors='coerce')
    df_clean = df_clean.dropna(subset=['TimeStamp']).sort_values('TimeStamp')
    if df_clean.empty:
        raise ValueError('有効なTimeStampを持つデータがありません。')

    # ウォームアップ期間を除外
    original_start = df_clean['TimeStamp'].iloc[0]
    session_start = original_start + pd.Timedelta(minutes=warmup_minutes)
    session_end = df_clean['TimeStamp'].iloc[-1]

    # ウォームアップ後のデータのみを使用
    df_clean = df_clean[df_clean['TimeStamp'] >= session_start]
    if df_clean.empty:
        raise ValueError(f'ウォームアップ期間（{warmup_minutes}分）除外後、有効なデータがありません。')

    segment_delta = pd.Timedelta(minutes=segment_minutes)

    # Fmθ時系列（ウォームアップ期間を除外）
    fmtheta_series = fmtheta_series.sort_index()
    fmtheta_series = fmtheta_series[fmtheta_series.index >= session_start]

    # SMR時系列（ウォームアップ期間を除外）
    if smr_series is not None:
        smr_series = smr_series.sort_index()
        smr_series = smr_series[smr_series.index >= session_start]

    # IAF時系列をStatistical DFから取得
    iaf_series = statistical_df['iaf'].sort_index()
    iaf_series = iaf_series[iaf_series.index >= session_start]

    # セグメント開始時刻のリストを取得（band_powersから）
    # band_powersのindexがセグメント開始タイムスタンプ
    segment_starts = list(band_powers_df.index)
    if not segment_starts:
        raise ValueError('Statistical DFにセグメントデータがありません。')

    records = []

    for idx, start in enumerate(segment_starts, start=1):
        end = start + segment_delta

        # Statistical DFから直接値を取得（セグメント化済み）
        # バンドパワー（dB）
        delta_mean = band_powers_df.loc[start, 'Delta'] if start in band_powers_df.index else np.nan
        theta_mean = band_powers_df.loc[start, 'Theta'] if start in band_powers_df.index else np.nan
        alpha_mean = band_powers_df.loc[start, 'Alpha'] if start in band_powers_df.index else np.nan
        beta_mean = band_powers_df.loc[start, 'Beta'] if start in band_powers_df.index else np.nan
        gamma_mean = band_powers_df.loc[start, 'Gamma'] if start in band_powers_df.index else np.nan

        # バンド比率（対数スケール: dB差分）
        theta_alpha_ratio_db = band_ratios_df.loc[start, 'theta_alpha_db'] if start in band_ratios_df.index else np.nan
        beta_alpha_ratio_db = band_ratios_df.loc[start, 'beta_alpha_db'] if start in band_ratios_df.index else np.nan
        beta_theta_ratio_db = band_ratios_df.loc[start, 'beta_theta_db'] if start in band_ratios_df.index else np.nan

        # バンド比率（実数値）
        theta_alpha_ratio = band_ratios_df.loc[start, 'theta_alpha'] if start in band_ratios_df.index else np.nan
        beta_alpha_ratio = band_ratios_df.loc[start, 'beta_alpha'] if start in band_ratios_df.index else np.nan
        beta_theta_ratio = band_ratios_df.loc[start, 'beta_theta'] if start in band_ratios_df.index else np.nan

        # Spectral Entropy
        se_mean = se_df.loc[start, 'spectral_entropy'] if start in se_df.index else np.nan

        # Fmθ平均の計算（外れ値除去）
        fm_slice = fmtheta_series.loc[(fmtheta_series.index >= start) & (fmtheta_series.index < end)]
        fm_clean = fm_slice.dropna()
        if len(fm_clean) > 3:
            z_scores = np.abs(stats.zscore(fm_clean))
            fm_filtered = fm_clean[z_scores < 3.0]
            fm_mean = fm_filtered.mean() if len(fm_filtered) > 0 else fm_clean.mean()
        else:
            fm_mean = fm_clean.mean() if len(fm_clean) > 0 else np.nan

        # SMR平均の計算（外れ値除去）
        smr_mean = np.nan
        if smr_series is not None:
            smr_slice = smr_series.loc[(smr_series.index >= start) & (smr_series.index < end)]
            smr_clean = smr_slice.dropna()
            if len(smr_clean) > 3:
                z_scores = np.abs(stats.zscore(smr_clean))
                smr_filtered = smr_clean[z_scores < 3.0]
                smr_mean = smr_filtered.mean() if len(smr_filtered) > 0 else smr_clean.mean()
            else:
                smr_mean = smr_clean.mean() if len(smr_clean) > 0 else np.nan

        # IAF平均（Statistical DFから自動取得済み）
        iaf_mean = np.nan
        iaf_cv = np.nan
        iaf_slice = iaf_series.loc[(iaf_series.index >= start) & (iaf_series.index < end)]
        iaf_mean = iaf_slice.mean()
        # IAF変動係数
        if len(iaf_slice) > 1:
            iaf_std = iaf_slice.std()
            iaf_val = iaf_slice.mean()
            if pd.notna(iaf_val) and iaf_val != 0:
                iaf_cv = iaf_std / iaf_val

        # fNIRS値を取得（オプション）
        hbo_mean = np.nan
        hbr_mean = np.nan
        if fnirs_df is not None and start in fnirs_df.index:
            hbo_mean = fnirs_df.loc[start, 'hbo_mean']
            hbr_mean = fnirs_df.loc[start, 'hbr_mean']

        # HR値を取得（オプション）
        hr_mean = np.nan
        if hr_df is not None and start in hr_df.index:
            hr_mean = hr_df.loc[start, 'hr_mean']

        # Yaw RMS値を取得（オプション）
        yaw_rms = np.nan
        if posture_df is not None and start in posture_df.index:
            yaw_rms = posture_df.loc[start, 'yaw_rms']

        # 相対パワー（%）の計算
        # dB → パワーに変換（10^(dB/10)）
        delta_power = 10 ** (delta_mean / 10) if pd.notna(delta_mean) else 0.0
        theta_power = 10 ** (theta_mean / 10) if pd.notna(theta_mean) else 0.0
        alpha_power = 10 ** (alpha_mean / 10) if pd.notna(alpha_mean) else 0.0
        beta_power = 10 ** (beta_mean / 10) if pd.notna(beta_mean) else 0.0
        gamma_power = 10 ** (gamma_mean / 10) if pd.notna(gamma_mean) else 0.0

        # 合計パワー
        total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power

        # 相対パワー（%）
        if total_power > 0:
            delta_relative = (delta_power / total_power) * 100
            theta_relative = (theta_power / total_power) * 100
            alpha_relative = (alpha_power / total_power) * 100
            beta_relative = (beta_power / total_power) * 100
            gamma_relative = (gamma_power / total_power) * 100
        else:
            delta_relative = np.nan
            theta_relative = np.nan
            alpha_relative = np.nan
            beta_relative = np.nan
            gamma_relative = np.nan

        # 総合スコア計算（利用可能な指標のみ）
        segment_score_result = calculate_meditation_score(
            fmtheta=fm_mean,
            spectral_entropy=se_mean,  # Statistical DFから取得
            theta_alpha_ratio=theta_alpha_ratio,  # 実数比率を使用
            faa=None,  # セグメント単位では未対応
            beta_alpha_ratio=beta_alpha_ratio,  # 実数値を使用
            iaf_cv=iaf_cv,
            hsi_quality=None,  # セグメント単位では未対応
        )
        meditation_score = segment_score_result['total_score']

        records.append({
            'segment_index': idx,
            'segment_start': start,
            'segment_end': end,
            'fmtheta_mean': fm_mean,
            'smr_mean': smr_mean,
            'spectral_entropy': se_mean,
            'iaf_mean': iaf_mean,
            'delta_mean': delta_mean,
            'theta_mean': theta_mean,
            'alpha_mean': alpha_mean,
            'beta_mean': beta_mean,
            'gamma_mean': gamma_mean,
            'delta_relative': delta_relative,
            'theta_relative': theta_relative,
            'alpha_relative': alpha_relative,
            'beta_relative': beta_relative,
            'gamma_relative': gamma_relative,
            'theta_alpha_ratio': theta_alpha_ratio,
            'theta_alpha_ratio_db': theta_alpha_ratio_db,
            'beta_alpha_ratio': beta_alpha_ratio,
            'beta_alpha_ratio_db': beta_alpha_ratio_db,
            'beta_theta_ratio': beta_theta_ratio,
            'beta_theta_ratio_db': beta_theta_ratio_db,
            'hbo_mean': hbo_mean,
            'hbr_mean': hbr_mean,
            'hr_mean': hr_mean,
            'yaw_rms': yaw_rms,
            'meditation_score': meditation_score,
        })

    segment_frame = pd.DataFrame(records)
    segment_frame['label'] = [
        f"{row['segment_start'].strftime('%H:%M')} - {row['segment_end'].strftime('%H:%M')}"
        for _, row in segment_frame.iterrows()
    ]
    # 経過時間ラベル（開始時間のみ、例: 0, 3, 6）
    segment_frame['elapsed_label'] = [
        int(row['segment_index'] * segment_minutes)
        for _, row in segment_frame.iterrows()
    ]

    if segment_frame.empty:
        raise ValueError('時間セグメント分析の結果が空です。')

    # 正規化スコア
    numeric_cols = ['fmtheta_mean', 'iaf_mean', 'alpha_mean', 'beta_mean', 'theta_alpha_ratio']
    metrics_df = segment_frame.set_index('segment_index')[numeric_cols]
    normalized = pd.DataFrame(
        {col: _min_max_normalize(metrics_df[col]) for col in metrics_df.columns}
    )
    normalized = normalized.reindex(metrics_df.index)

    # スコア計算・ピーク判定対象のセグメントインデックスを決定
    all_indices = segment_frame['segment_index'].tolist()
    excluded_indices = set()
    if exclude_first_segment and len(all_indices) > 0:
        excluded_indices.add(all_indices[0])
    if exclude_last_segment and len(all_indices) > 0:
        excluded_indices.add(all_indices[-1])
    scoring_indices = [idx for idx in all_indices if idx not in excluded_indices]

    # ピーク判定（総合スコアベース、除外セグメントを除く）
    meditation_scores = segment_frame.set_index('segment_index')['meditation_score']
    scoring_scores = meditation_scores.loc[meditation_scores.index.isin(scoring_indices)]
    if scoring_scores.dropna().empty:
        peak_idx = None
        peak_score = pd.Series(dtype=float)
    else:
        peak_score = meditation_scores  # 全体を保持（表示用）
        peak_idx = int(scoring_scores.idxmax())

    # 表形式（日本語ラベル）
    # テーブル1: バンドパワー詳細（絶対パワー + 相対パワー）
    # テーブル2: 比率と特徴指標（Theta/Alpha/Beta dBを除く）
    band_power_rows = []
    metrics_rows = []
    first_idx = all_indices[0] if all_indices else None
    last_idx = all_indices[-1] if all_indices else None

    for idx, row in segment_frame.iterrows():
        seg_idx = int(row['segment_index'])

        # 備考列: 最初/最後/ピークを表示
        note = ''
        if exclude_first_segment and seg_idx == first_idx:
            note = 'relaxing'
        elif exclude_last_segment and seg_idx == last_idx:
            note = 'post meditation'
        elif peak_idx is not None and seg_idx == peak_idx:
            note = 'peak'

        # テーブル1: バンドパワー詳細（11列）
        band_power_row = {
            'min': row['elapsed_label'],
            'δ (dB)': row['delta_mean'],
            'θ (dB)': row['theta_mean'],
            'α (dB)': row['alpha_mean'],
            'β (dB)': row['beta_mean'],
            'γ (dB)': row['gamma_mean'],
            'δ (%)': row['delta_relative'],
            'θ (%)': row['theta_relative'],
            'α (%)': row['alpha_relative'],
            'β (%)': row['beta_relative'],
            'γ (%)': row['gamma_relative'],
        }
        band_power_rows.append(band_power_row)

        # テーブル2: 比率と特徴指標（13列）
        metrics_row = {
            'min': row['elapsed_label'],
            'θ/α': row['theta_alpha_ratio'],
            'β/α': row['beta_alpha_ratio'],
            'β/θ': row['beta_theta_ratio'],
            'Fmθ (dB)': row['fmtheta_mean'],
            'SMR (dB)': row['smr_mean'],
            'SE': row['spectral_entropy'],
            'IAF (Hz)': row['iaf_mean'],
            'HbO': row['hbo_mean'],
            'HbR': row['hbr_mean'],
            'HR': row['hr_mean'],
            'Yaw RMS': row['yaw_rms'],
            '備考': note,
        }
        metrics_rows.append(metrics_row)

    band_power_table = pd.DataFrame(band_power_rows)
    metrics_table = pd.DataFrame(metrics_rows)

    # 後方互換性のため、tableはmetrics_tableと同じ内容にする
    table = metrics_table.copy()

    metadata = {
        'segment_minutes': segment_minutes,
        'session_start': session_start,
        'session_end': session_end,
        'peak_segment_index': peak_idx,
        'peak_time_range': (
            segment_frame.set_index('segment_index').loc[peak_idx, 'elapsed_label']
            if peak_idx is not None
            else None
        ),
        'peak_score': float(peak_score.loc[peak_idx]) if peak_idx is not None else None,
        'excluded_indices': list(excluded_indices),
        'scoring_indices': scoring_indices,
    }

    segments = segment_frame.set_index('segment_index')
    segments['segment_index'] = segments.index

    return SegmentAnalysisResult(
        segments=segments,
        table=table,
        normalized=normalized,
        metadata=metadata,
        band_power_table=band_power_table,
        metrics_table=metrics_table,
    )


def _normalize_indicator(
    value: float,
    min_val: float,
    max_val: float,
    reverse: bool = False,
) -> float:
    """
    指標を0-1に正規化する。

    Parameters
    ----------
    value : float
        正規化する値
    min_val : float
        最小値（この値が0になる）
    max_val : float
        最大値（この値が1になる）
    reverse : bool
        Trueの場合、値が低いほど良い指標として逆転（例：SE、HSI品質）

    Returns
    -------
    float
        正規化された値（0-1）
    """
    if pd.isna(value):
        return 0.5  # 欠損値はデフォルト0.5

    # クリッピング
    clipped = np.clip(value, min_val, max_val)

    # 正規化
    if np.isclose(max_val, min_val):
        normalized = 0.5
    else:
        normalized = (clipped - min_val) / (max_val - min_val)

    # 逆転（低いほど良い指標）
    if reverse:
        normalized = 1.0 - normalized

    return normalized


def calculate_meditation_score(
    fmtheta: Optional[float] = None,
    spectral_entropy: Optional[float] = None,
    theta_alpha_ratio: Optional[float] = None,
    faa: Optional[float] = None,
    beta_alpha_ratio: Optional[float] = None,
    iaf_cv: Optional[float] = None,
    hsi_quality: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    瞑想セッションの総合スコアを計算する。

    Parameters
    ----------
    fmtheta : float, optional
        Frontal Midline Thetaパワー（dB）
    spectral_entropy : float, optional
        Spectral Entropy（0-1正規化済み）
    theta_alpha_ratio : float, optional
        θ/α比（dB差）
    faa : float, optional
        Frontal Alpha Asymmetry（dB差）
    beta_alpha_ratio : float, optional
        β/α比（無次元、低いほどリラックス）
    iaf_cv : float, optional
        IAF変動係数（0-1、低いほど安定）
    hsi_quality : float, optional
        HSI品質スコア（1.0=Good, 4.0=Bad）
    weights : dict, optional
        重み辞書（指定しない場合はMEDITATION_SCORE_WEIGHTSを使用）

    Returns
    -------
    dict
        {
            'total_score': 総合スコア（0-100点）,
            'level': 評価レベル（優秀/良好/普通/要改善）,
            'scores': {指標名: 個別スコア（0-1）},
            'weights': 使用した重み辞書
        }
    """
    if weights is None:
        weights = MEDITATION_SCORE_WEIGHTS

    scores = {}

    # Fmθスコア（高いほど良い）
    # 旧: 50-200 μV² → 新: 17-23 dB (10*log10(50) ≈ 17, 10*log10(200) ≈ 23)
    if fmtheta is not None:
        scores['fmtheta'] = _normalize_indicator(fmtheta, min_val=17.0, max_val=23.0)
    else:
        scores['fmtheta'] = 0.5

    # SEスコア（低いほど良い、逆転）
    # 注: SEは通常0.7-1.0の範囲に収まるため、この範囲で正規化
    if spectral_entropy is not None:
        scores['spectral_entropy'] = _normalize_indicator(
            spectral_entropy, min_val=0.7, max_val=1.0, reverse=True
        )
    else:
        scores['spectral_entropy'] = 0.5

    # θ/α比スコア（高いほど良い）
    # 実数比率: 典型的な範囲は0.1〜1.0（1.0以上は稀）
    if theta_alpha_ratio is not None:
        scores['theta_alpha_ratio'] = _normalize_indicator(
            theta_alpha_ratio, min_val=0.1, max_val=1.0
        )
    else:
        scores['theta_alpha_ratio'] = 0.5

    # FAAスコア（正値ほど良い、中心化）
    # 旧: -0.5 ~ 0.5 (ln) → 新: -20.0 ~ 20.0 (dB差分)
    if faa is not None:
        scores['faa'] = _normalize_indicator(faa, min_val=-20.0, max_val=20.0)
    else:
        scores['faa'] = 0.5

    # β/α比スコア（低いほど良い、逆転）
    # β/α < 1 はリラックス状態（Alpha優位）、β/α > 1 は覚醒状態（Beta優位）
    if beta_alpha_ratio is not None:
        scores['beta_alpha_ratio'] = _normalize_indicator(
            beta_alpha_ratio, min_val=0.1, max_val=1.0, reverse=True
        )
    else:
        scores['beta_alpha_ratio'] = 0.5

    # IAF安定性スコア（変動係数が低いほど良い、逆転）
    if iaf_cv is not None:
        scores['iaf_stability'] = _normalize_indicator(
            iaf_cv, min_val=0.0, max_val=0.05, reverse=True
        )
    else:
        scores['iaf_stability'] = 0.5

    # 品質スコア（1.0=Good、4.0=Bad、逆変換）
    if hsi_quality is not None:
        scores['quality'] = _normalize_indicator(
            hsi_quality, min_val=1.0, max_val=4.0, reverse=True
        )
    else:
        scores['quality'] = 0.5

    # 重み付け平均で総合スコア算出（weightsに存在するキーのみ使用）
    total_score = sum(scores[key] * weights[key] for key in weights.keys() if key in scores)
    total_score_100 = total_score * 100.0

    # 評価レベル判定
    if total_score_100 >= 80:
        level = "優秀"
    elif total_score_100 >= 65:
        level = "良好"
    elif total_score_100 >= 50:
        level = "普通"
    else:
        level = "要改善"

    return {
        'total_score': total_score_100,
        'level': level,
        'scores': scores,
        'weights': weights,
    }


def calculate_best_metrics(segment_result: SegmentAnalysisResult) -> Dict[str, float]:
    """
    セグメント分析結果から集中瞑想に最適なbest値を抽出する。

    Parameters
    ----------
    segment_result : SegmentAnalysisResult
        calculate_segment_analysis()の戻り値

    Returns
    -------
    dict
        各指標のbest値を含む辞書:
        - fm_theta_best: Fmθ最大値（高いほど瞑想深度が深い）
        - iaf_best: IAF最大値（高いほど安定したアルファ波）
        - alpha_best: Alpha最大値（高いほどリラックス）
        - beta_best: Beta最小値（低いほど覚醒/ストレスが低い）
        - theta_alpha_best: θ/α比最大値（高いほど瞑想深度が深い）

    Notes
    -----
    集中瞑想の観点から各指標の選定基準を決定:
    - fm_theta, iaf, alpha, theta_alpha: 最大値（高いほど良い）
    - beta: 最小値（低いほど良い）
    除外セグメント（relaxing/post meditation）はbest値計算から除外されます。
    """
    segments = segment_result.segments

    # 除外セグメントを除いたデータを使用
    scoring_indices = segment_result.metadata.get('scoring_indices', list(segments.index))
    scoring_segments = segments.loc[segments.index.isin(scoring_indices)]

    # 集中瞑想に最適なbest値を計算
    best_metrics = {
        # 高いほど良い指標 → 最大値
        'fm_theta_best': scoring_segments['fmtheta_mean'].max() if 'fmtheta_mean' in scoring_segments else np.nan,
        'iaf_best': scoring_segments['iaf_mean'].max() if 'iaf_mean' in scoring_segments else np.nan,
        'alpha_best': scoring_segments['alpha_mean'].max() if 'alpha_mean' in scoring_segments else np.nan,
        'theta_alpha_best': scoring_segments['theta_alpha_ratio'].max() if 'theta_alpha_ratio' in scoring_segments else np.nan,
        # 低いほど良い指標 → 最小値
        'beta_best': scoring_segments['beta_mean'].min() if 'beta_mean' in scoring_segments else np.nan,
    }

    return best_metrics
