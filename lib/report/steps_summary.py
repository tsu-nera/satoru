"""
Statistical DataFrame / セグメント分析 / バンド比率 /
サマリーCSV / セッションログ保存の解析ステップ
"""

from lib import (
    calculate_best_metrics,
    calculate_segment_analysis,
)
from lib.sensors.eeg.visualization import plot_band_ratios
from lib.session_log import write_to_csv
from lib.statistical_dataframe import create_statistical_dataframe
from lib.visualization import plot_segment_comparison

from .step import analysis_step
from .steps_eeg import FRONTAL_CHANNELS


@analysis_step('Statistical DataFrame生成')
def build_statistical_dataframe(
    df, raw, results, warmup_minutes, fnirs_results, hr_data, hrv_result,
    respiration_result, artifact_window_samples, hsi=None,
):
    print('計算中: Statistical DataFrame（統一的なバンドパワー・比率計算）...')
    session_start = df['TimeStamp'].iloc[0]
    statistical_df = create_statistical_dataframe(
        raw,
        segment_minutes=3,
        warmup_minutes=warmup_minutes,
        session_start=session_start,
        fnirs_results=fnirs_results,
        hr_data=hr_data,
        df_timestamps=df['TimeStamp'],
        df=df,  # Posture統計量計算用
        hrv_result=hrv_result,  # HRVデータを追加
        respiration_result=respiration_result,  # 呼吸データを追加
        window_samples=artifact_window_samples,
        hsi=hsi,
    )
    results['statistical_df'] = statistical_df
    print(f'  バンドパワー: {len(statistical_df["band_powers"])} セグメント')
    print(f'  バンド比率: {len(statistical_df["band_ratios"])} セグメント')

    # 坐相統計量の取得（Statistical DataFrame から）
    if 'posture' in statistical_df and not statistical_df['posture'].empty:
        print(f'  Posture統計量: {len(statistical_df["posture"])} セグメント')
        posture_df = statistical_df['posture']

        # posture詳細テーブルを追加
        if 'posture' not in results:
            results['posture'] = {}
        results['posture']['detail_table'] = posture_df
    else:
        print('  警告: Statistical DataFrame に posture が含まれていません')

    return statistical_df


@analysis_step('HRV mean/best計算')
def compute_hrv_mean_best(results, mean_metrics, best_metrics):
    hrv_result_obj = results['hrv_result']
    if hasattr(hrv_result_obj, 'time_series') and 'rmssd' in hrv_result_obj.time_series:
        rmssd_series = hrv_result_obj.time_series['rmssd']
        rmssd_valid = rmssd_series.dropna()
        if len(rmssd_valid) > 0:
            mean_metrics['hrv_mean'] = rmssd_valid.mean()
            best_metrics['hrv_best'] = rmssd_valid.max()  # RMSSDは高いほど良い


@analysis_step('時間セグメント分析')
def analyze_segments(df_quality, fmtheta_result, smr_result, statistical_df, img_dir, results, warmup_minutes):
    print('計算中: 時間セグメント分析...')

    if statistical_df is None:
        print('警告: Statistical DFが生成されていないため、セグメント分析をスキップします。')
        raise ValueError('Statistical DFが必要です')

    # IAFはStatistical DFに含まれているため、準備不要
    segment_result = calculate_segment_analysis(
        df_quality,
        fmtheta_result.time_series,
        statistical_df,
        segment_minutes=3,
        warmup_minutes=warmup_minutes,
        exclude_first_segment=True,  # relaxing phase
        smr_series=smr_result.time_series if smr_result else None,
    )
    print('プロット中: 時間セグメント比較...')
    segment_plot_name = 'time_segment_metrics.png'
    plot_segment_comparison(
        segment_result,
        img_path=img_dir / segment_plot_name,
    )
    results['segment_table'] = segment_result.table  # 後方互換性のため残す
    results['band_power_table'] = segment_result.band_power_table
    results['metrics_table'] = segment_result.metrics_table
    results['segment_plot'] = segment_plot_name

    # best値を計算
    best_metrics = calculate_best_metrics(segment_result)
    results['best_metrics'] = best_metrics

    # mean値を計算（セグメントの平均）
    mean_metrics = _segment_mean_metrics(segment_result.segments)

    # HRV (RMSSD) の mean/best を計算
    if 'hrv_result' in results:
        compute_hrv_mean_best(results, mean_metrics, best_metrics)

    results['mean_metrics'] = mean_metrics

    return segment_result


def _segment_mean_metrics(segments):
    """セグメント分析結果からセッション平均の指標を計算する。"""
    return {
        'fm_theta_mean': segments['fmtheta_mean'].mean() if 'fmtheta_mean' in segments else None,
        'iaf_mean': segments['iaf_mean'].mean() if 'iaf_mean' in segments else None,
        'alpha_mean': segments['alpha_mean'].mean() if 'alpha_mean' in segments else None,
        'beta_mean': segments['beta_mean'].mean() if 'beta_mean' in segments else None,
        'theta_alpha_mean': segments['theta_alpha_ratio'].mean() if 'theta_alpha_ratio' in segments else None,
        # 相対パワーはドリフト混入の判定に使うためセッションログに残す
        'delta_rel_pct': segments['delta_relative'].mean() if 'delta_relative' in segments else None,
        'theta_rel_pct': segments['theta_relative'].mean() if 'theta_relative' in segments else None,
        'alpha_rel_pct': segments['alpha_relative'].mean() if 'alpha_relative' in segments else None,
    }


@analysis_step('前頭のみのセッション指標計算')
def compute_frontal_metrics(
    df, df_quality, fmtheta_result, smr_result, raw, hsi, results, warmup_minutes,
    artifact_window_samples,
):
    """
    セッションログの主要EEG列用に、AF7/AF8（Fpz基準）だけで指標を計算し直す。

    全チャネルの値はTPの接触状態でチャネル構成が変わると基準がずれ、
    セッション間で比較できない。レポート本文は全チャネルの値のまま、
    セッションログだけをこの値にする。
    """
    print('計算中: 前頭（AF7/AF8）のみのセッション指標...')
    picks = [ch for ch in FRONTAL_CHANNELS if ch in raw.ch_names]
    if not picks:
        raise ValueError('前頭チャネル（AF7/AF8）がありません')

    pick_idx = [raw.ch_names.index(ch) for ch in picks]
    raw_frontal = raw.copy().pick(picks)
    statistical_df = create_statistical_dataframe(
        raw_frontal,
        segment_minutes=3,
        warmup_minutes=warmup_minutes,
        session_start=df['TimeStamp'].iloc[0],
        window_samples=artifact_window_samples,
        hsi=hsi[pick_idx] if hsi is not None else None,
    )
    segment_result = calculate_segment_analysis(
        df_quality,
        fmtheta_result.time_series,
        statistical_df,
        segment_minutes=3,
        warmup_minutes=warmup_minutes,
        exclude_first_segment=True,
        smr_series=smr_result.time_series if smr_result else None,
    )
    valid_mask = statistical_df['valid_mask']
    frontal_metrics = {
        'mean_metrics': _segment_mean_metrics(segment_result.segments),
        'best_metrics': calculate_best_metrics(segment_result),
        'montage': '+'.join(ch.replace('RAW_', '') for ch in picks) + '/Fpz',
        'valid_coverage_pct': float(valid_mask.mean() * 100) if valid_mask.size else float('nan'),
    }
    results['frontal_metrics'] = frontal_metrics
    print(
        f"  使用チャネル: {frontal_metrics['montage']}、"
        f"有効率: {frontal_metrics['valid_coverage_pct']:.1f}%"
    )
    return frontal_metrics


@analysis_step('バンド比率プロット', show_traceback=True)
def plot_band_ratios_step(img_dir, results):
    print('プロット中: バンド比率...')
    if 'segment_table' in results:
        plot_band_ratios(
            results['segment_table'],
            img_path=img_dir / 'band_ratios.png',
        )
        results['band_ratios_img'] = 'band_ratios.png'
    else:
        print('警告: セグメントテーブルがないため、バンド比率プロットをスキップします。')


@analysis_step('セッションログCSV更新')
def write_session_log_csv(results):
    csv_path = write_to_csv(results=results)
    print(f'✓ セッションログCSV更新: {csv_path}')
    return csv_path


def save_session_log(results, save_to):
    """セッションログ保存（logs/session_log.csv への upsert）。"""
    if save_to == 'csv':
        print('更新中: セッションログ（CSV）...')
        write_session_log_csv(results)
    else:
        print('セッションログへの保存はスキップされました（--save-to オプションで指定）')
