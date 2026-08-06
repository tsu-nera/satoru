#!/usr/bin/env python3
"""
瞑想分析レポート生成スクリプト

Muse各種センサーデータ（EEG、fNIRS、ECG、IMU）を統合的に分析し、
マークダウンレポートを生成します。

Usage:
    python generate_report.py --data <CSV_PATH> [--output <REPORT_PATH>]
"""

import sys
from pathlib import Path
import argparse

import matplotlib
# lib 側が pyplot を import する前にバックエンドを固定する
matplotlib.use("Agg")

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# lib モジュールから関数をインポート
from lib import (
    load_mind_monitor_csv,
    calculate_band_statistics,
    calculate_hsi_statistics,
    generate_session_summary,
)
from lib.sensors.eeg.artifact import ARTIFACT_WINDOW_SAMPLES
from lib.sensors.eeg.band_power import compute_band_powers_from_raw, needs_band_power_computation

# 解析ステップ（fNIRS / 動作+心拍 / HRV / 呼吸 / EEGスペクトル / Statistical DF /
# セグメント / 総合スコア / ログ保存）
from lib.report import (
    analyze_fnirs,
    analyze_motion_and_hr,
    analyze_hrv,
    analyze_respiration,
    plot_band_power_series,
    prepare_mne_and_spectral,
    analyze_frontal_theta_step,
    analyze_smr_step,
    build_statistical_dataframe,
    analyze_segments,
    plot_band_ratios_step,
    calculate_session_score,
    save_session_log,
)




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


def run_full_analysis(data_path, output_dir, save_to='none', warmup_minutes=1.0, selfloops_data=None,
                      artifact_window_samples=ARTIFACT_WINDOW_SAMPLES):
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
    artifact_window_samples : int, default=ARTIFACT_WINDOW_SAMPLES
        アーチファクト判定兼PSDのWelch窓長（サンプル）。窓長を変えると除外率が
        大きく動くため、比較実験用に上書きできる（docs/adr/001）
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

    # バンドパワー列が空の場合、RAW EEGから計算（OSC CSV対応）
    if needs_band_power_computation(df):
        print('計算中: バンドパワー（RAW EEGから計算）...')
        df = compute_band_powers_from_raw(df)
        print('✓ バンドパワー列を付与しました')

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
    fnirs_results = analyze_fnirs(df, img_dir, results)

    # 動作検出（加速度・ジャイロ）と心拍数
    hr_data = analyze_motion_and_hr(df, img_dir, results, selfloops_data)

    # 自律神経系分析（HRV）
    hrv_pair = analyze_hrv(df, img_dir, results, selfloops_data, hr_data)
    hrv_result, hrv_data = hrv_pair if hrv_pair is not None else (None, None)

    # 呼吸分析（ECG-Derived Respiration）
    respiration_result = analyze_respiration(hrv_data, results)

    # バンドパワー時系列（Museアプリ風）
    df_quality = plot_band_power_series(df, img_dir, results)

    # MNE RAW準備 + PSD/スペクトログラム/PAF/ITF + EEGスペクトル解析
    # （PSDピーク・Alpha Power・FAA・Spectral Entropyを含む）
    raw, raw_unfiltered, psd_dict, paf_dict, tfr_primary = prepare_mne_and_spectral(
        df, img_dir, results, artifact_window_samples
    )

    # Frontal Midline Theta解析
    fmtheta_result = analyze_frontal_theta_step(df, raw_unfiltered, results)

    # SMR解析（12-15Hz, AF領域）
    smr_result = analyze_smr_step(df, raw_unfiltered, results)

    # Statistical DataFrame生成（統一的なバンドパワー・比率計算）
    statistical_df = None
    if raw is not None:
        statistical_df = build_statistical_dataframe(
            df, raw, results, warmup_minutes, fnirs_results, hr_data, hrv_result,
            respiration_result, artifact_window_samples,
        )

    # 時間セグメント分析
    analyze_segments(df_quality, fmtheta_result, smr_result, statistical_df, img_dir, results, warmup_minutes)

    # バンド比率（Statistical DFから取得）
    if statistical_df is not None:
        print('バンド比率統計をStatistical DFから取得...')
        results['band_ratios_stats'] = statistical_df['statistics']

        # セグメントテーブルからバンド比率をプロット
        plot_band_ratios_step(img_dir, results)
    else:
        print('警告: Statistical DFが生成されていないため、バンド比率をスキップします。')

    # セッション総合スコア計算
    calculate_session_score(fmtheta_result, statistical_df, results)

    # レポート生成
    generate_markdown_report(data_path, output_dir, results)

    # サマリーCSV生成
    print('生成中: サマリーCSV...')
    summary_result = generate_session_summary(data_path, results)
    summary_csv_path = output_dir / 'summary.csv'
    summary_result.summary.to_csv(summary_csv_path, index=False, encoding='utf-8')
    print(f'✓ サマリーCSV生成完了: {summary_csv_path}')

    # セッションログ保存（開発用CSV または 本番用Google Sheets）
    save_session_log(results, save_to)

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
        '--artifact-window-samples',
        type=int,
        default=ARTIFACT_WINDOW_SAMPLES,
        help=f'アーチファクト判定兼PSDのWelch窓長（サンプル、@256Hzで1024=4秒）。'
             f'窓長を変えると除外率が大きく動く（デフォルト: {ARTIFACT_WINDOW_SAMPLES}）'
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
        selfloops_data=args.selfloops_data,
        artifact_window_samples=args.artifact_window_samples,
    )

    return 0


if __name__ == '__main__':
    exit(main())
