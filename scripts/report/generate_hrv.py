#!/usr/bin/env python3
"""
HRVレポート生成スクリプト

Selfloops HRVデータを読み込み、HRV解析を実行してレポートを生成します。
実験的な機能を試すために、generate_report.pyから独立させています。

Usage:
    # デフォルト出力（tmp/hrv/）
    python generate_hrv.py --data <SELFLOOPS_CSV_PATH>

    # 出力先指定
    python generate_hrv.py --data <SELFLOOPS_CSV_PATH> --output <OUTPUT_DIR>
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data, get_heart_rate_data_from_selfloops
from lib.sensors.ecg.hrv import calculate_hrv_standard_set
from lib.sensors.ecg.segment_analysis_hrv import calculate_segment_hrv_analysis
from lib.sensors.ecg.visualization.hrv_plot import plot_hrv_time_series, plot_hrv_frequency, plot_hrv_nonlinear
from lib.sensors.ecg.analysis import analyze_hrv
from lib.sensors.ecg.respiration import estimate_resonance_breathing_pace
from lib.templates.filters import (
    number_format,
    format_percent,
    format_timestamp,
    format_duration,
    df_to_markdown,
)
from lib.templates.formatters import format_respiratory_stats


def generate_hrv_report(data_path, output_dir, results):
    """
    HRV解析レポート（Markdown）を生成

    Parameters
    ----------
    data_path : Path
        入力CSVファイルパス
    output_dir : Path
        出力ディレクトリ
    results : dict
        HRV解析結果を格納した辞書
    """
    report_path = output_dir / 'HRV_REPORT.md'

    print(f'\n生成中: HRVレポート -> {report_path}')

    # テンプレートディレクトリ（プロジェクトルート/templates）
    template_dir = project_root / 'templates'

    # Jinja2環境設定
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True
    )

    # カスタムフィルタ登録
    env.filters['number_format'] = number_format
    env.filters['format_percent'] = format_percent
    env.filters['format_timestamp'] = format_timestamp
    env.filters['format_duration'] = format_duration
    env.filters['df_to_markdown'] = df_to_markdown

    # コンテキスト構築
    context = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': data_path.name,
        'start_time': results.get('start_time'),
        'end_time': results.get('end_time'),
        'duration_sec': results.get('duration_sec'),
        'ecg': results.get('ecg', {}),
    }

    # テンプレート読み込み＆レンダリング
    template = env.get_template('hrv_report.md.j2')
    report_content = template.render(**context)

    # ファイル書き込み
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f'✓ HRVレポート生成完了: {report_path}')


def analyze_hrv_session(data_path, output_dir, warmup_seconds=60.0):
    """
    HRVセッション解析を実行

    Parameters
    ----------
    data_path : Path
        Selfloops CSVファイルパス
    output_dir : Path
        出力ディレクトリ
    warmup_seconds : float, default=60.0
        ウォームアップ除外時間（秒）
    """
    print('='*60)
    print('HRV解析')
    print('='*60)
    print()

    # 画像出力ディレクトリ
    img_dir = output_dir / 'img'
    img_dir.mkdir(exist_ok=True, parents=True)

    # データ読み込み
    print(f'Loading: {data_path}')
    sl_df = load_selfloops_csv(str(data_path), warmup_seconds=warmup_seconds)

    print(f'データ形状: {sl_df.shape[0]} 行 × {sl_df.shape[1]} 列')

    # タイムスタンプ表示
    if 'TimeStamp' in sl_df.columns:
        start_time = sl_df['TimeStamp'].min()
        end_time = sl_df['TimeStamp'].max()
        duration_sec = (end_time - start_time).total_seconds()

        print(f'記録時間: {start_time.strftime("%Y-%m-%d %H:%M:%S")} ~ {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'計測時間: {duration_sec / 60.0:.1f} 分\n')

    # HRVデータ取得
    print('計算中: HRVデータ抽出...')
    hrv_data = get_hrv_data(sl_df, clean_artifacts=True)

    # 心拍数データ取得
    hr_data = get_heart_rate_data_from_selfloops(sl_df)

    # セッション時間チェック
    total_duration = hrv_data['time'][-1] - hrv_data['time'][0]
    print(f'HRVデータ時間: {total_duration:.0f} 秒')

    if total_duration < 180:
        print(f'⚠️  警告: 記録時間が短すぎます（{total_duration:.0f}秒 < 180秒）')
        print('   HRV解析の精度が低下する可能性があります')

    # HRV標準セット解析
    print('\n計算中: HRV標準セット解析...')
    hrv_result = calculate_hrv_standard_set(hrv_data)

    # 統計情報表示
    print('\nHRV統計情報:')
    print(hrv_result.statistics.to_string(index=False))

    # HRV時系列プロット
    print('\nプロット中: HRV時系列...')
    hrv_img_name = 'hrv_time_series.png'
    plot_hrv_time_series(
        hrv_result,
        img_path=str(img_dir / hrv_img_name),
        title='HRV Time Series Analysis',
        hr_data=hr_data
    )
    print(f'✓ 保存: {img_dir / hrv_img_name}')

    # HRV周波数解析
    print('\n計算中: HRV周波数解析...')
    hrv_indices = analyze_hrv(hrv_data, show=False)

    print('\nHRV周波数解析結果:')
    for key, value in hrv_indices.items():
        if isinstance(value, (int, float)):
            print(f'  {key}: {value:.2f}')

    print('\nプロット中: HRV周波数解析...')
    hrv_freq_img_name = 'hrv_frequency.png'
    plot_hrv_frequency(
        hrv_data,
        hrv_indices=hrv_indices,
        img_path=str(img_dir / hrv_freq_img_name)
    )
    print(f'✓ 保存: {img_dir / hrv_freq_img_name}')

    # HRV非線形解析プロット
    print('\nプロット中: HRV非線形解析（Poincaré + DFA）...')
    hrv_nonlinear_img_name = 'hrv_nonlinear.png'
    plot_hrv_nonlinear(
        hrv_data,
        result=hrv_result,
        img_path=str(img_dir / hrv_nonlinear_img_name)
    )
    print(f'✓ 保存: {img_dir / hrv_nonlinear_img_name}')

    # 呼吸分析（ECG-Derived Respiration）
    print('\n計算中: 呼吸分析（ECG-Derived Respiration）...')
    respiration_result, rbp_result = estimate_resonance_breathing_pace(
        hrv_data,
        target_fs=8.0,
        peak_distance=8.0,
        window_minutes=3.0,
        bin_width=0.5
    )

    print(f'\n呼吸分析結果:')
    print(f'  平均BR: {respiration_result.breathing_rate:.1f} bpm')
    if rbp_result:
        print(f'  共鳴呼吸回数（RMSSD基準）: {rbp_result.optimal_rmssd["range"]} bpm')
        # RMSSD最大値は存在する場合のみ表示
        if "max_rmssd" in rbp_result.optimal_rmssd:
            print(f'  RMSSD最大値: {rbp_result.optimal_rmssd["max_rmssd"]:.1f} ms')

    # HRV時間セグメント分析
    print('\n計算中: HRV時間セグメント分析（3分ごと）...')
    segment_hrv_result = calculate_segment_hrv_analysis(
        hrv_data,
        segment_minutes=3.0,
        warmup_seconds=warmup_seconds
    )
    print(f'✓ セグメント分析完了: {segment_hrv_result.metadata["num_segments"]}セグメント')

    # レポート生成用のresults辞書を構築
    results = {}

    # データ情報
    if 'TimeStamp' in sl_df.columns:
        start_time = sl_df['TimeStamp'].min()
        end_time = sl_df['TimeStamp'].max()
        duration_sec = (end_time - start_time).total_seconds()
        results['start_time'] = start_time
        results['end_time'] = end_time
        results['duration_sec'] = duration_sec

    # ECG/HRVデータ
    ecg = {}
    ecg['hrv_img'] = hrv_img_name
    ecg['hrv_freq_img'] = hrv_freq_img_name
    ecg['hrv_nonlinear_img'] = hrv_nonlinear_img_name

    # hrv_statsを時間領域、周波数領域、非線形に分割
    hrv_stats_df = hrv_result.statistics
    if not hrv_stats_df.empty and 'Domain' in hrv_stats_df.columns:
        time_domain_df = hrv_stats_df[hrv_stats_df['Domain'] == 'Time Domain']
        freq_domain_df = hrv_stats_df[hrv_stats_df['Domain'] == 'Frequency Domain']
        nonlinear_df = hrv_stats_df[hrv_stats_df['Domain'] == 'Nonlinear']

        if not time_domain_df.empty:
            ecg['hrv_time_stats'] = time_domain_df.drop(columns=['Domain'])
        if not freq_domain_df.empty:
            ecg['hrv_freq_stats'] = freq_domain_df.drop(columns=['Domain'])
        if not nonlinear_df.empty:
            ecg['hrv_nonlinear_stats'] = nonlinear_df.drop(columns=['Domain'])

    # 呼吸データ
    ecg['respiratory_stats'] = format_respiratory_stats(respiration_result)
    ecg['respiratory_period'] = respiration_result

    # 共鳴呼吸データ
    if rbp_result and hasattr(rbp_result, 'bin_statistics'):
        ecg['rbp_stats'] = rbp_result.bin_statistics

    # HRVセグメント分析
    ecg['segment_hrv_table'] = segment_hrv_result.table

    results['ecg'] = ecg

    # Markdownレポート生成
    generate_hrv_report(data_path, output_dir, results)

    print()
    print('='*60)
    print('HRV解析完了!')
    print('='*60)
    print(f'出力先: {img_dir}/')
    print(f'レポート: {output_dir / "HRV_REPORT.md"}')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='Selfloops HRVデータのHRV解析レポート生成'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Selfloops CSVファイルパス'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=project_root / 'tmp' / 'hrv',
        help='出力ディレクトリ（デフォルト: tmp/hrv/）'
    )
    parser.add_argument(
        '--warmup',
        type=float,
        default=60.0,
        help='ウォームアップ除外時間（秒）（デフォルト: 60.0）'
    )

    args = parser.parse_args()

    # パスの検証
    if not args.data.exists():
        print(f'エラー: データファイルが見つかりません: {args.data}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    # HRV解析実行
    analyze_hrv_session(
        args.data,
        args.output,
        warmup_seconds=args.warmup
    )

    return 0


if __name__ == '__main__':
    exit(main())
