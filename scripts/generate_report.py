#!/usr/bin/env python3
"""
瞑想分析レポート生成スクリプト

Muse各種センサーデータ（EEG、fNIRS、ECG、IMU）を統合的に分析し、
マークダウンレポートを生成します。

Usage:
    python generate_report.py --data <CSV_PATH> [--output <REPORT_PATH>]
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# lib モジュールから関数をインポート
from lib import (
    load_mind_monitor_csv,
    calculate_band_statistics,
    calculate_hsi_statistics,
    prepare_mne_raw,
    filter_eeg_quality,
    calculate_psd,
    calculate_spectrogram,
    calculate_spectrogram_all_channels,
    calculate_paf,
    get_psd_peak_frequencies,
    calculate_frontal_theta,
    calculate_frontal_asymmetry,
    calculate_alpha_power,
    calculate_alpha_power_from_raw,
    calculate_spectral_entropy,
    calculate_spectral_entropy_time_series,
    calculate_segment_analysis,
    calculate_meditation_score,
    calculate_best_metrics,
    get_optics_data,
    analyze_fnirs,
    generate_session_summary,
    get_heart_rate_data,
    analyze_motion,
    analyze_psd_peaks,
    calculate_smr,
)
from lib.session_log import write_to_csv, write_to_google_sheets
from lib.sensors.ecg.respiration import calculate_respiratory_period

# 可視化関数をインポート
from lib.sensors.eeg.visualization import (
    plot_band_power_time_series,
    plot_psd,
    plot_spectrogram,
    plot_spectrogram_grid,
    plot_band_ratios,
    plot_paf,
    plot_raw_preview,
    plot_frontal_theta,
    plot_frontal_asymmetry,
    plot_psd_peaks,
    plot_smr,
)

from lib.visualization import (
    plot_segment_comparison,
    plot_fnirs_muse_style,
    plot_motion_heart_rate,
    create_motion_stats_table,
)
from lib.sensors.imu import PostureAnalyzer
from lib.statistical_dataframe import create_statistical_dataframe




def generate_markdown_report(data_path, output_dir, results):
    """
    マークダウンレポートを生成

    Parameters
    ----------
    data_path : Path
        入力CSVファイルパス
    output_dir : Path
        出力ディレクトリ
    results : dict
        分析結果を格納した辞書
    """
    from lib.templates import MeditationReportRenderer

    report_path = output_dir / 'REPORT.md'

    print(f'生成中: マークダウンレポート -> {report_path}')

    # テンプレートレンダラーでレポート生成
    renderer = MeditationReportRenderer()
    context = renderer.build_context(results, data_path)
    report_content = renderer.render_report(context)

    # ファイルに書き込み
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f'✓ レポート生成完了: {report_path}')


def run_full_analysis(data_path, output_dir, save_to='none', warmup_minutes=1.0, selfloops_data=None):
    """
    完全な分析を実行

    Parameters
    ----------
    data_path : Path
        入力CSVファイルパス
    output_dir : Path
        出力ディレクトリ
    save_to : str, default='none'
        セッションログの保存先
        - 'none': 保存しない（デフォルト）
        - 'csv': ローカルCSVに保存（開発用）
        - 'sheets': Google Sheetsに保存（本番用）
    warmup_minutes : float, default=1.0
        ウォームアップ除外時間（分）。短い記録の場合は0を指定。
    selfloops_data : Path, default=None
        Selfloops HRVデータファイルパス（オプション）
    """
    print('='*60)
    print('瞑想分析レポート生成')
    print('='*60)
    print()

    # 画像出力ディレクトリ
    img_dir = output_dir / 'img'
    img_dir.mkdir(exist_ok=True)

    # 分析結果を格納
    results = {}

    # データ読み込み
    print(f'Loading: {data_path}')
    df = load_mind_monitor_csv(data_path, filter_headband=False, warmup_seconds=warmup_minutes * 60)

    # データ情報を記録
    results['data_info'] = {
        'shape': df.shape,
        'start_time': df['TimeStamp'].min(),
        'end_time': df['TimeStamp'].max(),
        'duration_sec': (df['TimeStamp'].max() - df['TimeStamp'].min()).total_seconds()
    }

    print(f'データ形状: {df.shape[0]} 行 × {df.shape[1]} 列')

    # タイムスタンプ表示用
    start_time = results["data_info"]["start_time"]
    end_time = results["data_info"]["end_time"]
    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time is not None else 'N/A'
    end_str = end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time is not None else 'N/A'
    print(f'記録時間: {start_str} ~ {end_str}')

    duration_sec = results["data_info"]["duration_sec"]
    duration_min = duration_sec / 60.0 if duration_sec is not None else None
    if duration_min is not None:
        print(f'計測時間: {duration_min:.1f} 分\n')
    else:
        print('計測時間: N/A\n')

    # HSI接続品質統計
    print('計算中: 接続品質 (HSI)...')
    hsi_stats = calculate_hsi_statistics(df)
    results['hsi_stats'] = hsi_stats

    # バンド統計
    print('計算中: バンド統計量...')
    band_stats = calculate_band_statistics(df)
    results['band_statistics'] = band_stats['statistics']

    # fNIRS解析
    fnirs_results = None
    try:
        optics_data = get_optics_data(df)
        if optics_data and len(optics_data['time']) > 0:
            print('計算中: fNIRS統計...')
            fnirs_results = analyze_fnirs(optics_data)

            # fNIRS統計をDataFrame化（lateralityを除外してleft/rightのみ）
            hemisphere_stats = {k: v for k, v in fnirs_results['stats'].items() if k != 'laterality'}
            df_stats = pd.DataFrame(hemisphere_stats).T
            df_stats = df_stats.rename(
                index={"left": "左半球", "right": "右半球"},
                columns={
                    "hbo_mean": "HbO平均", "hbo_min": "HbO最小", "hbo_max": "HbO最大",
                    "hbr_mean": "HbR平均", "hbr_min": "HbR最小", "hbr_max": "HbR最大",
                    "hbt_mean": "HbT平均", "hbd_mean": "HbD平均",
                },
            )
            results['fnirs_stats'] = df_stats[
                ["HbO平均", "HbO最小", "HbO最大", "HbR平均", "HbR最小", "HbR最大", "HbT平均", "HbD平均"]
            ]
            results['fnirs_laterality'] = fnirs_results['stats']['laterality']

            print('プロット中: fNIRS時系列...')
            fig_fnirs, _ = plot_fnirs_muse_style(fnirs_results)
            fnirs_img_name = 'fnirs_muse_style.png'
            fig_fnirs.savefig(img_dir / fnirs_img_name, dpi=150, bbox_inches='tight')
            plt.close(fig_fnirs)
            results['fnirs_img'] = fnirs_img_name
    except KeyError as exc:
        print(f'警告: fNIRSデータを処理できませんでした ({exc})')

    # 動作検出（加速度・ジャイロ）と心拍数
    hr_data = None
    hr_data_source = None
    motion_result = None
    try:
        # 心拍数データ取得（Selfloops優先、なければMuse）
        if selfloops_data and selfloops_data.exists():
            # Selfloopsデータから心拍数を取得
            print(f'Loading Selfloops HR data: {selfloops_data}')
            from lib.loaders.selfloops import load_selfloops_csv, get_heart_rate_data_from_selfloops
            sl_df = load_selfloops_csv(str(selfloops_data), warmup_seconds=0.0)
            hr_data = get_heart_rate_data_from_selfloops(sl_df)
            hr_data_source = 'Selfloops'
        else:
            # Museデータから心拍数を取得
            hr_data = get_heart_rate_data(df)
            hr_data_source = 'Muse'

        results['hr_data_source'] = hr_data_source  # レポート表示用

        # 動作検出（10秒間隔）
        print('計算中: 動作検出（加速度・ジャイロ）...')
        motion_result = analyze_motion(df, interval='10s')

        # 統計情報をDataFrame化（心拍数情報を含む）
        motion_stats = create_motion_stats_table(motion_result, hr_data=hr_data)

        # 時系列プロット（動作検出のみ、心拍数は含まない）
        print('プロット中: 動作検出時系列...')
        motion_img_name = 'motion_only.png'
        fig_motion, _ = plot_motion_heart_rate(motion_result, hr_data=None, df=df)
        fig_motion.savefig(img_dir / motion_img_name, dpi=150, bbox_inches='tight')
        plt.close(fig_motion)

        # postureネスト構造で保存（テンプレート用）
        results['posture'] = {
            'motion_img': motion_img_name,
            'summary_table': motion_stats,
        }

        # 内部処理用データ
        results['motion_ratio'] = motion_result['motion_ratio']

    except Exception as exc:
        print(f'警告: 心拍数データまたは動作検出を処理できませんでした ({exc})')

    # 自律神経系分析（HRV）
    hrv_result = None
    hrv_data = None
    try:
        if selfloops_data and selfloops_data.exists():
            print('計算中: HRV解析（自律神経系）...')
            from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
            from lib.sensors.ecg.hrv import calculate_hrv_standard_set

            sl_df = load_selfloops_csv(str(selfloops_data), warmup_seconds=60.0)
            hrv_data = get_hrv_data(sl_df, clean_artifacts=True)

            # セッション時間チェック
            total_duration = hrv_data['time'][-1] - hrv_data['time'][0]
            if total_duration < 180:
                print(f'⚠️  HRV解析スキップ: 記録時間が短すぎます（{total_duration:.0f}秒 < 180秒）')
            else:
                hrv_result = calculate_hrv_standard_set(hrv_data)
                results['hrv_stats'] = hrv_result.statistics
                results['hrv_result'] = hrv_result  # 時系列データ用に保存

                print('プロット中: HRV時系列...')
                from lib.sensors.ecg.visualization.hrv_plot import plot_hrv_time_series, plot_hrv_frequency
                from lib.sensors.ecg.analysis import analyze_hrv

                hrv_img_name = 'hrv_time_series.png'
                plot_hrv_time_series(
                    hrv_result,
                    img_path=str(img_dir / hrv_img_name),
                    title='HRV Time Series Analysis',
                    hr_data=hr_data
                )
                results['hrv_img'] = hrv_img_name

                # HRV周波数解析
                print('プロット中: HRV周波数解析...')
                hrv_freq_img_name = 'hrv_frequency.png'
                hrv_indices = analyze_hrv(hrv_data, show=False)
                plot_hrv_frequency(
                    hrv_data,
                    hrv_indices=hrv_indices,
                    img_path=str(img_dir / hrv_freq_img_name)
                )
                results['hrv_freq_img'] = hrv_freq_img_name

    except Exception as e:
        print(f'⚠️  HRV解析エラー: {e}')
        import traceback
        traceback.print_exc()

    # 呼吸分析（ECG-Derived Respiration）
    respiration_result = None
    rbp_result = None
    try:
        if hrv_data is not None:
            print('計算中: 呼吸分析（ECG-Derived Respiration）...')
            from lib.sensors.ecg.respiration import estimate_resonance_breathing_pace

            respiration_result, rbp_result = estimate_resonance_breathing_pace(
                hrv_data,
                target_fs=8.0,
                peak_distance=8.0,
                window_minutes=3.0,
                bin_width=0.5
            )

            # 結果を保存（内部処理用）
            results['respiration_result'] = respiration_result
            results['rbp_result'] = rbp_result

            print(f'  平均BR: {respiration_result.breathing_rate:.1f} bpm')
            if rbp_result:
                print(f'  共鳴呼吸回数（RMSSD基準）: {rbp_result.optimal_rmssd["range"]} bpm')

    except Exception as e:
        print(f'⚠️  呼吸分析エラー: {e}')
        import traceback
        traceback.print_exc()

    # バンドパワー時系列（Museアプリ風）
    print('プロット中: バンドパワー時系列...')
    df_quality, quality_mask = filter_eeg_quality(df)
    df_for_band = df_quality if not df_quality.empty else df
    plot_band_power_time_series(
        df_for_band,
        img_path=img_dir / 'band_power_time_series.png',
        rolling_window=200,
        resample_interval='10S',
        smooth_window=5,
        clip_percentile=98.0
    )
    results['band_power_img'] = 'band_power_time_series.png'
    results['band_power_quality_ratio'] = float(quality_mask.mean())

    # MNE RAW準備
    print('準備中: MNE RAWデータ...')
    mne_dict = prepare_mne_raw(df)
    raw = None
    raw_unfiltered = None  # Fmθ/FAA用のフィルタなしraw

    if mne_dict:
        raw = mne_dict['raw']
        print(f'検出されたチャネル: {mne_dict["channels"]}')
        print(f'推定サンプリングレート: {mne_dict["sfreq"]:.2f} Hz')

        # Fmθ/FAA計算用に、バンドパスフィルタを適用しないrawデータを作成
        # (これらの関数は内部で独自のバンドパスフィルタを適用するため)
        mne_dict_unfiltered = prepare_mne_raw(df, apply_bandpass=False, apply_notch=False)
        if mne_dict_unfiltered:
            raw_unfiltered = mne_dict_unfiltered['raw']

        # Rawプレビュー
        print('プロット中: 生データプレビュー...')
        raw_preview_img = 'raw_preview.png'
        raw_duration = raw.times[-1] if raw.n_times else 0.0
        preview_duration = raw_duration if raw_duration and raw_duration < 180 else 180.0
        plot_raw_preview(
            raw,
            img_path=img_dir / raw_preview_img,
            duration_sec=preview_duration,
            start_sec=0.0,
            n_channels=min(4, len(mne_dict['channels'])),
        )
        results['raw_preview_img'] = raw_preview_img

        # PSD計算
        print('計算中: パワースペクトル密度...')
        psd_dict = calculate_psd(raw)

        # PSDプロット
        print('プロット中: パワースペクトル密度...')
        plot_psd(psd_dict, img_path=img_dir / 'psd.png')
        results['psd_img'] = 'psd.png'

        # ピーク周波数
        results['psd_peaks'] = get_psd_peak_frequencies(psd_dict)

        # スペクトログラム（全チャネル）
        # 256Hzは過剰なため、64Hzにダウンサンプリング（高速化）
        # スペクトログラムは30Hz程度までカバーできれば十分
        print('計算中: スペクトログラム（全チャネル）...')
        raw_for_tfr = raw.copy().resample(64, verbose=False)
        tfr_results = calculate_spectrogram_all_channels(raw_for_tfr)
        tfr_primary = None
        tfr_primary_channel = None

        if tfr_results:
            print('プロット中: スペクトログラム（全チャネル）...')
            plot_spectrogram_grid(tfr_results, img_path=img_dir / 'spectrogram.png')
            results['spectrogram_img'] = 'spectrogram.png'

            # 時系列解析で優先的に使うチャネル（TP9優先、なければ最初のチャネル）
            preferred_channels = ('RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10')
            for channel in preferred_channels:
                if channel in tfr_results:
                    tfr_primary = tfr_results[channel]
                    tfr_primary_channel = channel
                    break
            if tfr_primary is None:
                tfr_primary_channel, tfr_primary = next(iter(tfr_results.items()))

        # PAF分析
        print('計算中: Peak Alpha Frequency...')
        paf_dict = calculate_paf(psd_dict)

        # PAFプロット
        print('プロット中: PAF...')
        plot_paf(paf_dict, img_path=img_dir / 'paf.png')
        results['paf_img'] = 'paf.png'

        # IAFサマリー
        iaf_summary = []
        for ch_label, paf_result in paf_dict['paf_by_channel'].items():
            iaf_summary.append({
                'チャネル': ch_label,
                'Peak (Hz)': paf_result['PAF'],
                'CoG (Hz)': paf_result['CoG'],
                'Power (μV²/Hz)': paf_result['Power']
            })
        results['paf_summary'] = pd.DataFrame(iaf_summary)
        results['iaf'] = {
            'value': paf_dict['iaf'],
            'std': paf_dict['iaf_std'],
            'peak': paf_dict['iaf_peak'],
            'cog': paf_dict['iaf_cog']
        }

        # PSDピーク分析（SMR含む）
        try:
            print('計算中: PSDピーク分析...')
            iaf_for_peaks = paf_dict.get('iaf_peak', paf_dict.get('iaf'))
            psd_peaks_result = analyze_psd_peaks(psd_dict, iaf=iaf_for_peaks)
            results['harmonics_table'] = psd_peaks_result.peaks_table
            results['harmonics_stats'] = psd_peaks_result.statistics
            results['harmonics_result'] = psd_peaks_result

            print('プロット中: PSDピーク分析...')
            plot_psd_peaks(psd_peaks_result, psd_dict, img_path=img_dir / 'psd_peaks.png')
            results['harmonics_img'] = 'psd_peaks.png'
        except Exception as exc:
            print(f'警告: PSDピーク分析に失敗しました ({exc})')

        # Alpha Power (Brain Recharge Score) 解析
        try:
            print('計算中: Alpha Power (Brain Recharge Score)...')
            # Alpha列があるか確認（Mind Monitor形式）
            alpha_cols = ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']
            has_alpha_data = all(
                col in df.columns and df[col].notna().any()
                for col in alpha_cols
            )
            if has_alpha_data:
                alpha_power_result = calculate_alpha_power(df)
            else:
                # RAW EEGからAlpha Powerを計算（Muse App OSC形式）
                print('  Alpha列が空のため、RAW EEGから計算...')
                alpha_power_result = calculate_alpha_power_from_raw(df)
            results['alpha_power_score'] = alpha_power_result.score
            results['alpha_power_db'] = alpha_power_result.alpha_db
            results['alpha_power_stats'] = alpha_power_result.statistics
            results['alpha_power_metadata'] = alpha_power_result.metadata
        except Exception as exc:
            print(f'警告: Alpha Power解析に失敗しました ({exc})')

        # FAA解析
        try:
            print('計算中: Frontal Alpha Asymmetry...')
            faa_result = calculate_frontal_asymmetry(df, raw=raw_unfiltered)
            print('プロット中: Frontal Alpha Asymmetry...')
            plot_frontal_asymmetry(
                faa_result,
                img_path=img_dir / 'frontal_alpha_asymmetry.png'
            )
            results['faa_img'] = 'frontal_alpha_asymmetry.png'
            results['faa_stats'] = faa_result.statistics
        except Exception as exc:
            print(f'警告: FAA解析に失敗しました ({exc})')

        # Spectral Entropy解析
        try:
            print('計算中: Spectral Entropy...')

            # PSDから全体のエントロピーを計算
            se_result = calculate_spectral_entropy(psd_dict)

            # 時系列エントロピーの計算（スペクトログラムから）
            if tfr_primary:
                session_start = df['TimeStamp'].iloc[0]
                se_time_result = calculate_spectral_entropy_time_series(
                    tfr_primary,
                    start_time=pd.to_datetime(session_start)
                )

                results['spectral_entropy_stats'] = se_time_result.statistics
                results['spectral_entropy_change'] = se_time_result.metadata.get('change_percent')
        except Exception as exc:
            print(f'警告: Spectral Entropy解析に失敗しました ({exc})')

    # Frontal Midline Theta解析
    fmtheta_result = None
    try:
        print('計算中: Frontal Midline Theta...')
        fmtheta_result = calculate_frontal_theta(df, raw=raw_unfiltered if raw_unfiltered else None)
        print('プロット中: Frontal Midline Theta...')
        plot_frontal_theta(
            fmtheta_result,
            img_path=img_dir / 'frontal_midline_theta.png'
        )
        results['frontal_theta_img'] = 'frontal_midline_theta.png'
        results['frontal_theta_stats'] = fmtheta_result.statistics
        results['frontal_theta_increase'] = fmtheta_result.metadata.get('increase_rate_percent')
    except Exception as exc:
        print(f'警告: Fmθ解析に失敗しました ({exc})')

    # SMR解析（12-15Hz, AF領域）
    smr_result = None
    try:
        print('計算中: SMR (12-15Hz, AF領域)...')
        smr_result = calculate_smr(df, raw=raw_unfiltered if raw_unfiltered else None)
        print('プロット中: SMR...')
        plot_smr(
            smr_result,
            img_path=img_dir / 'smr.png'
        )
        results['smr_img'] = 'smr.png'
        results['smr_stats'] = smr_result.statistics
        results['smr_increase'] = smr_result.metadata.get('increase_rate_percent')
    except Exception as exc:
        print(f'警告: SMR解析に失敗しました ({exc})')

    # Statistical DataFrame生成（統一的なバンドパワー・比率計算）
    statistical_df = None
    if raw is not None:
        try:
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
            )
            results['statistical_df'] = statistical_df
            print(f'  バンドパワー: {len(statistical_df["band_powers"])} セグメント')
            print(f'  バンド比率: {len(statistical_df["band_ratios"])} セグメント')

            # 坐相統計量の取得（Statistical DataFrame から）
            if 'posture' in statistical_df and not statistical_df['posture'].empty:
                print(f'  Posture統計量: {len(statistical_df["posture"])} セグメント')
                posture_df = statistical_df['posture']

                # 後方互換性のため、PostureAnalyzer でもサマリーを計算
                posture_analyzer = PostureAnalyzer()
                posture_summary = posture_analyzer.compute_summary(df)

                # posture詳細テーブルを追加
                if 'posture' not in results:
                    results['posture'] = {}
                results['posture']['detail_table'] = posture_df
            else:
                print('  警告: Statistical DataFrame に posture が含まれていません')

        except Exception as exc:
            print(f'警告: Statistical DataFrame生成に失敗しました ({exc})')

    # 時間セグメント分析
    try:
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
            try:
                hrv_result_obj = results['hrv_result']
                if hasattr(hrv_result_obj, 'time_series') and 'rmssd' in hrv_result_obj.time_series:
                    rmssd_series = hrv_result_obj.time_series['rmssd']
                    rmssd_valid = rmssd_series.dropna()
                    if len(rmssd_valid) > 0:
                        mean_metrics['hrv_mean'] = rmssd_valid.mean()
                        best_metrics['hrv_best'] = rmssd_valid.max()  # RMSSDは高いほど良い
            except Exception as e:
                print(f'警告: HRV mean/best計算に失敗しました ({e})')

        results['mean_metrics'] = mean_metrics

    except Exception as exc:
        print(f'警告: 時間セグメント分析に失敗しました ({exc})')

    # バンド比率（Statistical DFから取得）
    if statistical_df is not None:
        print('バンド比率統計をStatistical DFから取得...')
        results['band_ratios_stats'] = statistical_df['statistics']

        # セグメントテーブルからバンド比率をプロット
        try:
            print('プロット中: バンド比率...')
            if 'segment_table' in results:
                plot_band_ratios(
                    results['segment_table'],
                    img_path=img_dir / 'band_ratios.png',
                )
                results['band_ratios_img'] = 'band_ratios.png'
            else:
                print('警告: セグメントテーブルがないため、バンド比率プロットをスキップします。')
        except Exception as exc:
            print(f'警告: バンド比率プロットに失敗しました ({exc})')
            import traceback
            traceback.print_exc()
    else:
        print('警告: Statistical DFが生成されていないため、バンド比率をスキップします。')

    # セッション総合スコア計算
    try:
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

    except Exception as exc:
        print(f'警告: 総合スコア計算に失敗しました ({exc})')

    # レポート生成
    generate_markdown_report(data_path, output_dir, results)

    # サマリーCSV生成
    print('生成中: サマリーCSV...')
    summary_result = generate_session_summary(data_path, results)
    summary_csv_path = output_dir / 'summary.csv'
    summary_result.summary.to_csv(summary_csv_path, index=False, encoding='utf-8')
    print(f'✓ サマリーCSV生成完了: {summary_csv_path}')

    # セッションログ保存（開発用CSV または 本番用Google Sheets）
    if save_to == 'csv':
        print('更新中: セッションログ（CSV）...')
        try:
            csv_path = write_to_csv(results=results)
            print(f'✓ セッションログCSV更新: {csv_path}')
        except Exception as exc:
            print(f'警告: セッションログCSV更新に失敗しました ({exc})')
    elif save_to == 'sheets':
        print('更新中: セッションログ（Google Sheets）...')
        try:
            spreadsheet_id = os.environ.get('GSHEET_SESSION_LOG_ID')
            if not spreadsheet_id:
                print('警告: 環境変数 GSHEET_SESSION_LOG_ID が設定されていません')
            else:
                write_to_google_sheets(
                    results=results,
                    spreadsheet_id=spreadsheet_id,
                )
                print(f'✓ Google Sheets更新: {spreadsheet_id}')
        except Exception as exc:
            print(f'警告: Google Sheets更新に失敗しました ({exc})')
    else:
        print('セッションログへの保存はスキップされました（--save-to オプションで指定）')

    print()
    print('='*60)
    print('分析完了!')
    print('='*60)
    print(f'レポート: {output_dir / "REPORT.md"}')
    print(f'サマリー: {summary_csv_path}')
    print(f'画像: {img_dir}/')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='Muse各種センサーデータ（EEG、fNIRS、ECG、IMU）の統合的な瞑想分析とレポート生成'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='入力CSVファイルパス'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent,
        help='出力ディレクトリ（デフォルト: スクリプトと同じディレクトリ）'
    )
    parser.add_argument(
        '--save-to',
        type=str,
        choices=['none', 'csv', 'sheets'],
        default='none',
        help='セッションログの保存先: none=保存しない（デフォルト）, csv=ローカルCSV（開発用）, sheets=Google Sheets（本番用）'
    )
    parser.add_argument(
        '--warmup',
        type=float,
        default=1.0,
        help='ウォームアップ除外時間（分）。短い記録の場合は0を指定（デフォルト: 1.0）'
    )
    parser.add_argument(
        '--selfloops-data',
        type=Path,
        default=None,
        help='Selfloops HRVデータファイルパス（オプション）。指定された場合、Muse心拍数の代わりに使用'
    )

    args = parser.parse_args()

    # パスの検証
    if not args.data.exists():
        print(f'エラー: データファイルが見つかりません: {args.data}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    # 分析実行
    run_full_analysis(
        args.data,
        args.output,
        save_to=args.save_to,
        warmup_minutes=args.warmup,
        selfloops_data=args.selfloops_data
    )

    return 0


if __name__ == '__main__':
    exit(main())
