"""
Statistical DataFrame / セグメント分析 / バンド比率 / 総合スコア /
サマリーCSV / セッションログ保存の解析ステップ
"""

import os

from lib import (
    calculate_segment_analysis,
    calculate_meditation_score,
    calculate_best_metrics,
)
from lib.session_log import write_to_csv, write_to_google_sheets
from lib.sensors.eeg.visualization import plot_band_ratios
from lib.visualization import plot_segment_comparison
from lib.statistical_dataframe import create_statistical_dataframe

from .step import analysis_step


@analysis_step('Statistical DataFrame生成')
def build_statistical_dataframe(
    df, raw, results, warmup_minutes, fnirs_results, hr_data, hrv_result,
    respiration_result, artifact_window_samples,
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
        exclude_last_segment=True,   # post meditation stage
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
    results['segment_peak_range'] = segment_result.metadata.get('peak_time_range')
    results['segment_peak_score'] = segment_result.metadata.get('peak_score')

    # best値を計算
    best_metrics = calculate_best_metrics(segment_result)
    results['best_metrics'] = best_metrics

    # mean値を計算（セグメントの平均）
    segments = segment_result.segments
    mean_metrics = {
        'fm_theta_mean': segments['fmtheta_mean'].mean() if 'fmtheta_mean' in segments else None,
        'iaf_mean': segments['iaf_mean'].mean() if 'iaf_mean' in segments else None,
        'alpha_mean': segments['alpha_mean'].mean() if 'alpha_mean' in segments else None,
        'beta_mean': segments['beta_mean'].mean() if 'beta_mean' in segments else None,
        'theta_alpha_mean': segments['theta_alpha_ratio'].mean() if 'theta_alpha_ratio' in segments else None,
    }

    # HRV (RMSSD) の mean/best を計算
    if 'hrv_result' in results:
        compute_hrv_mean_best(results, mean_metrics, best_metrics)

    results['mean_metrics'] = mean_metrics

    return segment_result


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


@analysis_step('セッション総合スコア計算')
def calculate_session_score(fmtheta_result, statistical_df, results):
    print('計算中: セッション総合スコア...')

    # 各指標から必要な値を抽出
    fmtheta_val = None
    if fmtheta_result and 'frontal_theta_stats' in results:
        # 平均値を取得
        stats_df = results['frontal_theta_stats']
        fmtheta_row = stats_df[stats_df['Metric'] == 'Mean']
        if not fmtheta_row.empty:
            fmtheta_val = fmtheta_row['Value'].iloc[0]

    se_val = None
    if 'spectral_entropy_stats' in results:
        se_stats_df = results['spectral_entropy_stats']
        se_row = se_stats_df[se_stats_df['Metric'] == 'Mean']
        if not se_row.empty:
            se_val = se_row['Value'].iloc[0]

    theta_alpha_val = None
    beta_alpha_val = None
    beta_theta_val = None

    # バンド比率: セグメント分析から取得（Statistical DFベース、最も信頼性が高い）
    if 'segment_table' in results:
        segment_df = results['segment_table']

        # θ/α比（実数比率）: 総合スコア計算用
        if 'θ/α' in segment_df.columns:
            theta_alpha_values = segment_df['θ/α'].dropna()
            if len(theta_alpha_values) > 0:
                theta_alpha_val = theta_alpha_values.mean()

        # β/α比（実数比率）
        if 'β/α' in segment_df.columns:
            beta_alpha_values = segment_df['β/α'].dropna()
            if len(beta_alpha_values) > 0:
                beta_alpha_val = beta_alpha_values.mean()

        # β/θ比（実数比率）
        if 'β/θ' in segment_df.columns:
            beta_theta_values = segment_df['β/θ'].dropna()
            if len(beta_theta_values) > 0:
                beta_theta_val = beta_theta_values.mean()
                results['beta_theta_ratio'] = beta_theta_val  # レポート用に保存

    # フォールバック: Statistical DFから直接取得
    if theta_alpha_val is None and statistical_df is not None:
        stats_df = statistical_df['statistics']
        theta_alpha_row = stats_df[stats_df['Metric'] == 'theta_alpha_db_Mean']
        if not theta_alpha_row.empty:
            theta_alpha_val = theta_alpha_row['Value'].iloc[0]

    if beta_alpha_val is None and statistical_df is not None:
        stats_df = statistical_df['statistics']
        beta_alpha_row = stats_df[stats_df['Metric'] == 'beta_alpha_Mean']
        if not beta_alpha_row.empty:
            beta_alpha_val = beta_alpha_row['Value'].iloc[0]

    faa_val = None
    if 'faa_stats' in results:
        faa_stats_df = results['faa_stats']
        faa_row = faa_stats_df[faa_stats_df['Metric'] == 'Mean FAA']
        if not faa_row.empty:
            faa_val = faa_row['Value'].iloc[0]

    iaf_cv_val = None

    # Statistical DFから直接IAF変動係数を取得（最も信頼性が高い）
    if statistical_df is not None and 'statistics' in statistical_df:
        stats_df = statistical_df['statistics']
        iaf_cv_row = stats_df[stats_df['Metric'] == 'iaf_CV']
        if not iaf_cv_row.empty:
            iaf_cv_val = iaf_cv_row['Value'].iloc[0]

    # フォールバック: セグメント分析から計算
    if iaf_cv_val is None and 'segment_table' in results:
        segment_df = results['segment_table']
        if 'IAF (Hz)' in segment_df.columns:
            iaf_values = segment_df['IAF (Hz)'].dropna()
            if len(iaf_values) > 1:
                iaf_mean = iaf_values.mean()
                iaf_std = iaf_values.std()
                if iaf_mean > 0:
                    iaf_cv_val = iaf_std / iaf_mean

    hsi_quality_val = None
    if 'hsi_stats' in results:
        hsi_stats = results['hsi_stats']
        if 'avg_quality' in hsi_stats:
            hsi_quality_val = hsi_stats['avg_quality']

    # 総合スコア計算
    session_score = calculate_meditation_score(
        fmtheta=fmtheta_val,
        spectral_entropy=se_val,
        theta_alpha_ratio=theta_alpha_val,
        faa=faa_val,
        beta_alpha_ratio=beta_alpha_val,
        iaf_cv=iaf_cv_val,
        hsi_quality=hsi_quality_val,
    )

    results['session_score'] = session_score['total_score']
    results['session_level'] = session_score['level']
    results['session_score_breakdown'] = session_score['scores']

    return session_score


@analysis_step('セッションログCSV更新')
def write_session_log_csv(results):
    csv_path = write_to_csv(results=results)
    print(f'✓ セッションログCSV更新: {csv_path}')
    return csv_path


@analysis_step('Google Sheets更新')
def write_session_log_sheets(results, spreadsheet_id):
    write_to_google_sheets(
        results=results,
        spreadsheet_id=spreadsheet_id,
    )
    print(f'✓ Google Sheets更新: {spreadsheet_id}')
    return spreadsheet_id


def save_session_log(results, save_to):
    """セッションログ保存（開発用CSV または 本番用Google Sheets）。"""
    if save_to == 'csv':
        print('更新中: セッションログ（CSV）...')
        write_session_log_csv(results)
    elif save_to == 'sheets':
        print('更新中: セッションログ（Google Sheets）...')
        spreadsheet_id = os.environ.get('GSHEET_SESSION_LOG_ID')
        if not spreadsheet_id:
            print('警告: 環境変数 GSHEET_SESSION_LOG_ID が設定されていません')
        else:
            write_session_log_sheets(results, spreadsheet_id)
    else:
        print('セッションログへの保存はスキップされました（--save-to オプションで指定）')
