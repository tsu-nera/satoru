#!/usr/bin/env python3
"""
Muse脳波データ基本分析スクリプト

lib/eeg.py の関数を使用してマークダウンレポートを生成します。

Usage:
    python generate_report.py --data <CSV_PATH> [--output <REPORT_PATH>]
"""

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
    calculate_paf_time_evolution,
    plot_band_power_time_series,
    plot_psd,
    plot_psd_time_series,
    plot_spectrogram,
    plot_spectrogram_grid,
    plot_band_ratios,
    plot_paf,
    plot_paf_time_evolution,
    plot_raw_preview,
    get_psd_peak_frequencies,
    calculate_frontal_theta,
    plot_frontal_theta,
    calculate_frontal_asymmetry,
    plot_frontal_asymmetry,
    calculate_spectral_entropy,
    calculate_spectral_entropy_time_series,
    calculate_segment_analysis,
    plot_segment_comparison,
    calculate_meditation_score,
    get_optics_data,
    analyze_fnirs,
    plot_fnirs_muse_style,
    generate_session_summary,
)
from lib.statistical_dataframe import create_statistical_dataframe


def format_timestamp_for_report(value):
    """Datetime表示を秒精度に整形"""
    if value is None:
        return 'N/A'
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    if hasattr(value, 'strftime'):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    return str(value)


def seconds_to_minutes(value):
    """秒を分表示用に変換"""
    try:
        return float(value) / 60.0
    except (TypeError, ValueError):
        return None


def generate_fnirs_stats_table(fnirs_stats: dict) -> pd.DataFrame:
    """fNIRS統計情報をDataFrame化して整形"""
    df_stats = pd.DataFrame(fnirs_stats).T
    df_stats = df_stats.rename(
        index={"left": "左半球", "right": "右半球"},
        columns={
            "hbo_mean": "HbO平均",
            "hbo_std": "HbO標準偏差",
            "hbo_min": "HbO最小",
            "hbo_max": "HbO最大",
            "hbr_mean": "HbR平均",
            "hbr_std": "HbR標準偏差",
            "hbr_min": "HbR最小",
            "hbr_max": "HbR最大",
        },
    )
    return df_stats[
        [
            "HbO平均",
            "HbO標準偏差",
            "HbO最小",
            "HbO最大",
            "HbR平均",
            "HbR標準偏差",
            "HbR最小",
            "HbR最大",
        ]
    ]


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
    report_path = output_dir / 'REPORT.md'

    print(f'生成中: マークダウンレポート -> {report_path}')

    info = results.get('data_info', {})

    start_time = format_timestamp_for_report(info.get('start_time'))
    end_time = format_timestamp_for_report(info.get('end_time'))
    duration_min = seconds_to_minutes(info.get('duration_sec'))
    duration_str = f"{duration_min:.1f} 分" if duration_min is not None else "N/A"

    report = f"""# Muse脳波データ分析レポート

- **生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **データファイル**: `{data_path.name}`
- **記録時間**: {start_time} ~ {end_time}
- **計測時間**: {duration_str}

---

"""

    # ========================================
    # 接続品質（HSI）
    # ========================================
    if 'hsi_stats' in results:
        hsi_data = results['hsi_stats']
        overall_quality = hsi_data.get('overall_quality')
        good_ratio = hsi_data.get('good_ratio', 0.0) * 100

        report += "## 📡 接続品質\n\n"

        # 全体評価
        if overall_quality is not None:
            report += f"- **総合品質スコア**: {overall_quality:.2f}\n"
            report += f"- **Good品質率**: {good_ratio:.1f}%\n\n"

        # チャネル別詳細
        if not hsi_data['statistics'].empty:
            report += "### チャネル別詳細\n\n"
            report += hsi_data['statistics'].to_markdown(index=False, floatfmt='.2f')
            report += "\n\n"
            report += "> **注**: 1.0=Good, 2.0=Medium, 4.0=Bad\n\n"

    # ========================================
    # 生データプレビュー
    # ========================================
    if 'raw_preview_img' in results:
        report += "## 🧾 生データプレビュー\n\n"
        report += f"![生データプロット](img/{results['raw_preview_img']})\n\n"
        report += "> **注**: フィルタ適用後EEGの初期数分（μV表示）。異常波形の早期チェック用。\n\n"

    # ========================================
    # 分析サマリー
    # ========================================
    report += "## 📊 分析サマリー\n\n"

    # 総合スコア
    if 'session_score' in results:
        report += "### 総合評価\n\n"
        report += f"- **総合スコア**: {results['session_score']:.1f}/100\n"

        # スコア内訳
        if 'session_score_breakdown' in results:
            report += "\n**スコア内訳**\n\n"
            breakdown = results['session_score_breakdown']
            score_labels = {
                'fmtheta': '瞑想深度 (Fmθ)',
                'spectral_entropy': '集中度 (SE)',
                'theta_alpha_ratio': '瞑想深度 (θ/α)',
                'alpha_beta_ratio': 'リラックス度 (α/β)',
                'iaf_stability': '周波数安定性 (IAF)',
            }
            for key, label in score_labels.items():
                if key in breakdown:
                    score_100 = breakdown[key] * 100
                    report += f"- {label}: {score_100:.1f}/100\n"
        report += "\n"

    # セッション総合評価
    if 'band_ratios_stats' in results:
        report += "### セッション総合評価\n\n"
        ratios_df = results['band_ratios_stats']

        # 新形式（縦長）か旧形式（横長）かを判定
        if 'DisplayName' in ratios_df.columns:
            # 新形式：Mean行のみを表示
            mean_rows = ratios_df[ratios_df['Metric'].str.contains('Mean', na=False)]
            for _, row in mean_rows.iterrows():
                ratio_name = row['DisplayName']
                avg_value = row['Value']
                report += f"- **{ratio_name}**: {avg_value:.3f}\n"
        else:
            # 旧形式（後方互換）
            for _, row in ratios_df.iterrows():
                ratio_name = row.get('指標', row.get('Metric', '不明'))
                avg_value = row.get('平均値', row.get('Value', 0.0))
                report += f"- **{ratio_name}**: {avg_value:.3f}\n"

        # Fmθ平均を追加
        if 'frontal_theta_stats' in results:
            fmtheta_df = results['frontal_theta_stats']
            fmtheta_mean_row = fmtheta_df[fmtheta_df['Metric'] == 'Mean']
            if not fmtheta_mean_row.empty:
                fmtheta_value = fmtheta_mean_row['Value'].iloc[0]
                report += f"- **Fmθ平均 (μV²)**: {fmtheta_value:.3f}\n"

        # IAF平均を追加
        if 'iaf' in results:
            iaf_data = results['iaf']
            iaf_value = iaf_data['value']
            iaf_std = iaf_data['std']
            report += f"- **IAF平均 (Hz)**: {iaf_value:.2f} ± {iaf_std:.2f}\n"

        report += "\n"

    # ピークパフォーマンス区間
    segment_keys = {'segment_table', 'segment_plot', 'segment_peak_range'}
    if any(key in results for key in segment_keys):
        peak_range = results.get('segment_peak_range')
        peak_score = results.get('segment_peak_score')
        if peak_range:
            report += "### ピークパフォーマンス\n\n"
            if peak_score is not None:
                report += f"- **最高パフォーマンス区間**: {peak_range}\n"
                report += f"- **スコア**: {peak_score:.1f}/100\n\n"
            else:
                report += f"- **最高パフォーマンス区間**: {peak_range}\n\n"

    # ========================================
    # 周波数帯域分析
    # ========================================
    band_power_keys = {
        'band_power_img',
        'psd_img',
        'spectrogram_img'
    }
    if any(key in results for key in band_power_keys):
        report += "## 🧠 周波数帯域分析\n\n"

        if 'band_power_img' in results:
            report += "### バンドパワー時系列\n\n"
            report += f"![バンドパワー時系列](img/{results['band_power_img']})\n\n"

        if 'psd_img' in results:
            report += "### パワースペクトル密度（PSD）\n\n"
            report += f"![PSD](img/{results['psd_img']})\n\n"

        if 'spectrogram_img' in results:
            report += "### スペクトログラム\n\n"
            report += f"![スペクトログラム](img/{results['spectrogram_img']})\n\n"

    # ========================================
    # 特徴的指標分析
    # ========================================
    fmtheta_keys = {'frontal_theta_img', 'frontal_theta_stats', 'frontal_theta_increase'}
    paf_keys = {'paf_img', 'paf_summary', 'iaf', 'paf_time_img', 'paf_time_stats'}
    faa_keys = {'faa_img', 'faa_stats'}
    band_ratio_keys = {'band_ratios_img', 'band_ratios_stats'}

    if any(key in results for key in (fmtheta_keys | paf_keys | faa_keys | band_ratio_keys)):
        report += "## 🎯 特徴的指標分析\n\n"

        # Frontal Midline Theta
        if any(key in results for key in fmtheta_keys):
            report += "### Frontal Midline Theta (Fmθ)\n\n"

            if 'frontal_theta_img' in results:
                report += f"![Frontal Midline Theta](img/{results['frontal_theta_img']})\n\n"

            if 'frontal_theta_stats' in results:
                report += results['frontal_theta_stats'].to_markdown(index=False, floatfmt='.3f')
                report += "\n\n"

            if 'frontal_theta_increase' in results:
                inc = results['frontal_theta_increase']
                if pd.notna(inc):
                    report += f"セッション後半の平均Fmθは前半比で **{inc:+.1f}%** 変化しました。\n\n"

        # Peak Alpha Frequency
        if any(key in results for key in paf_keys):
            report += "### Peak Alpha Frequency (PAF)\n\n"

            if 'paf_img' in results:
                report += f"![PAF](img/{results['paf_img']})\n\n"

            if 'iaf' in results:
                iaf_data = results['iaf']
                report += f"**Individual Alpha Frequency (IAF)**: {iaf_data['value']:.2f} ± {iaf_data['std']:.2f} Hz\n\n"

            if 'paf_summary' in results:
                report += "**チャネル別詳細**\n\n"
                report += results['paf_summary'].to_markdown(index=False, floatfmt='.2f')
                report += "\n\n"

        # Frontal Alpha Asymmetry
        if any(key in results for key in faa_keys):
            report += "### Frontal Alpha Asymmetry (FAA)\n\n"

            if 'faa_img' in results:
                report += f"![Frontal Alpha Asymmetry](img/{results['faa_img']})\n\n"

            if 'faa_stats' in results:
                report += results['faa_stats'].to_markdown(index=False, floatfmt='.3f')
                report += "\n\n"
                report += "> **解釈**: FAA = ln(右) - ln(左)。正値は左半球優位（接近動機・ポジティブ感情）、負値は右半球優位（回避動機・ネガティブ感情）を示唆します。\n\n"

        # Spectral Entropy
        if 'spectral_entropy_stats' in results:
            report += "### Spectral Entropy (SE)\n\n"
            report += results['spectral_entropy_stats'].to_markdown(index=False, floatfmt='.3f')
            report += "\n\n"

            if 'spectral_entropy_change' in results:
                change = results['spectral_entropy_change']
                if pd.notna(change):
                    interpretation = "低下（注意集中）" if change < 0 else "上昇（注意散漫）"
                    report += f"セッション後半のエントロピーは前半比で **{change:+.1f}%** 変化しました。\n"
                    report += f"**解釈**: {interpretation}\n\n"

            report += "> **解釈**: Spectral Entropyは周波数成分の多様性を示します。低い値は特定の周波数帯に集中（集中状態）、高い値は広帯域に分散（散漫状態）を示唆します。\n\n"

        # バンド比率
        if any(key in results for key in band_ratio_keys):
            report += "### バンド比率指標\n\n"

            if 'band_ratios_img' in results:
                report += f"![バンド比率](img/{results['band_ratios_img']})\n\n"

    # ========================================
    # 血流動態分析 (fNIRS)
    # ========================================
    if "fnirs_stats" in results or "fnirs_img" in results:
        report += "## 🩸 血流動態分析 (fNIRS)\n\n"

        if "fnirs_img" in results:
            report += "### HbO/HbR時系列\n\n"
            report += f"![fNIRS時系列](img/{results['fnirs_img']})\n\n"

        if "fnirs_stats" in results:
            report += "### 統計サマリー\n\n"
            report += results["fnirs_stats"].to_markdown(floatfmt=".2f")
            report += "\n\n"

    # ========================================
    # 時間経過分析
    # ========================================
    if any(key in results for key in segment_keys):
        report += "## ⏱️ 時間経過分析\n\n"

        if 'segment_plot' in results:
            report += "### セグメント別パフォーマンス\n\n"
            report += f"![時間セグメント比較](img/{results['segment_plot']})\n\n"

        if 'segment_table' in results:
            report += "### 詳細データ\n\n"
            report += results['segment_table'].to_markdown(index=False, floatfmt='.3f')
            report += "\n\n"

    # ファイルに書き込み
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'✓ レポート生成完了: {report_path}')


def run_full_analysis(data_path, output_dir):
    """
    完全な分析を実行

    Parameters
    ----------
    data_path : Path
        入力CSVファイルパス
    output_dir : Path
        出力ディレクトリ
    """
    print('='*60)
    print('Muse脳波データ基本分析')
    print('='*60)
    print()

    # 画像出力ディレクトリ
    img_dir = output_dir / 'img'
    img_dir.mkdir(exist_ok=True)

    # 分析結果を格納
    results = {}

    # データ読み込み
    print(f'Loading: {data_path}')
    df = load_mind_monitor_csv(data_path, filter_headband=False)

    # データ情報を記録
    results['data_info'] = {
        'shape': df.shape,
        'start_time': df['TimeStamp'].min(),
        'end_time': df['TimeStamp'].max(),
        'duration_sec': (df['TimeStamp'].max() - df['TimeStamp'].min()).total_seconds()
    }

    print(f'データ形状: {df.shape[0]} 行 × {df.shape[1]} 列')
    print(
        f'記録時間: '
        f'{format_timestamp_for_report(results["data_info"]["start_time"])} '
        f'~ {format_timestamp_for_report(results["data_info"]["end_time"])}'
    )
    duration_min = seconds_to_minutes(results["data_info"]["duration_sec"])
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
    try:
        optics_data = get_optics_data(df)
        if optics_data and len(optics_data['time']) > 0:
            print('計算中: fNIRS統計...')
            fnirs_results = analyze_fnirs(optics_data)
            results['fnirs_stats'] = generate_fnirs_stats_table(fnirs_results['stats'])

            print('プロット中: fNIRS時系列...')
            fig_fnirs, _ = plot_fnirs_muse_style(fnirs_results)
            fnirs_img_name = 'fnirs_muse_style.png'
            fig_fnirs.savefig(img_dir / fnirs_img_name, dpi=150, bbox_inches='tight')
            plt.close(fig_fnirs)
            results['fnirs_img'] = fnirs_img_name
    except KeyError as exc:
        print(f'警告: fNIRSデータを処理できませんでした ({exc})')

    # バンドパワー時系列（Museアプリ風）
    print('プロット中: バンドパワー時系列...')
    df_quality, quality_mask = filter_eeg_quality(df)
    df_for_band = df_quality if not df_quality.empty else df
    plot_band_power_time_series(
        df_for_band,
        img_path=img_dir / 'band_power_time_series.png',
        rolling_window=200,
        resample_interval='2S',
        smooth_window=9,
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

        # PSD時系列
        print('プロット中: PSDの時間推移...')
        psd_time_img_name = 'psd_time_series.png'
        plot_psd_time_series(
            raw,
            channels=mne_dict['channels'],
            img_path=img_dir / psd_time_img_name,
            fmin=1.0,
            fmax=40.0,
            window_sec=8.0,
            step_sec=2.0,
            clip_percentile=90.0,
            smooth_window=7
        )
        results['psd_time_series_img'] = psd_time_img_name

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

        # PAFサマリー
        paf_summary = []
        for ch_label, paf_result in paf_dict['paf_by_channel'].items():
            paf_summary.append({
                'チャネル': ch_label,
                'PAF (Hz)': paf_result['PAF'],
                'Power (μV²/Hz)': paf_result['Power']
            })
        results['paf_summary'] = pd.DataFrame(paf_summary)
        results['iaf'] = {'value': paf_dict['iaf'], 'std': paf_dict['iaf_std']}

        # PAF時間推移（優先チャネルを使用）
        if tfr_primary:
            print(f'計算中: PAF時間推移... (チャネル: {tfr_primary_channel})')
            paf_time_dict = calculate_paf_time_evolution(tfr_primary, paf_dict)

            print('プロット中: PAF時間推移...')
            plot_paf_time_evolution(paf_time_dict, df, paf_dict, img_path=img_dir / 'paf_time_evolution.png')
            results['paf_time_img'] = 'paf_time_evolution.png'
            results['paf_time_stats'] = paf_time_dict['stats']

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

    # Statistical DataFrame生成（統一的なバンドパワー・比率計算）
    statistical_df = None
    if raw is not None:
        try:
            print('計算中: Statistical DataFrame（統一的なバンドパワー・比率計算）...')
            session_start = df['TimeStamp'].iloc[0]
            statistical_df = create_statistical_dataframe(
                raw,
                segment_minutes=3,
                warmup_minutes=1.0,
                session_start=session_start,
            )
            results['statistical_df'] = statistical_df
            print(f'  バンドパワー: {len(statistical_df["band_powers"])} セグメント')
            print(f'  バンド比率: {len(statistical_df["band_ratios"])} セグメント')
        except Exception as exc:
            print(f'警告: Statistical DataFrame生成に失敗しました ({exc})')

    # 時間セグメント分析
    try:
        print('計算中: 時間セグメント分析...')

        if statistical_df is None:
            print('警告: Statistical DFが生成されていないため、セグメント分析をスキップします。')
            raise ValueError('Statistical DFが必要です')

        # IAF時系列の準備（PAF時間推移から）
        iaf_series = None
        if 'paf_time_img' in results and paf_time_dict:
            # PAF時間推移のタイムスタンプとIAF値をSeriesに変換
            session_start = df['TimeStamp'].iloc[0]
            iaf_times = pd.to_datetime(session_start) + pd.to_timedelta(paf_time_dict['times'], unit='s')
            iaf_series = pd.Series(paf_time_dict['paf_smoothed'], index=iaf_times)

        segment_result = calculate_segment_analysis(
            df_quality,
            fmtheta_result.time_series,
            statistical_df,
            segment_minutes=3,
            iaf_series=iaf_series,
            warmup_minutes=1.0,  # 最初の1分間を除外（アーティファクト対策）
        )
        print('プロット中: 時間セグメント比較...')
        segment_plot_name = 'time_segment_metrics.png'
        plot_segment_comparison(
            segment_result,
            img_path=img_dir / segment_plot_name,
        )
        results['segment_table'] = segment_result.table
        results['segment_plot'] = segment_plot_name
        results['segment_peak_range'] = segment_result.metadata.get('peak_time_range')
        results['segment_peak_score'] = segment_result.metadata.get('peak_score')
    except Exception as exc:
        print(f'警告: 時間セグメント分析に失敗しました ({exc})')

    # バンド比率（Statistical DFから取得）
    if statistical_df is not None:
        print('バンド比率統計をStatistical DFから取得...')
        results['band_ratios_stats'] = statistical_df['statistics']

        # プロット用にStatistical DFからデータを準備
        try:
            print('プロット中: バンド比率...')
            # plot_band_ratios関数が期待する形式に変換
            band_ratios_df = statistical_df['band_ratios'].copy()
            band_ratios_df = band_ratios_df.reset_index()
            band_ratios_df = band_ratios_df.rename(columns={'index': 'TimeStamp'})

            # カラム名を旧形式に合わせる（plot_band_ratios関数との互換性のため）
            band_ratios_df = band_ratios_df.rename(columns={
                'alpha_beta': 'Alpha/Beta',
                'beta_theta': 'Beta/Theta',
                'theta_alpha': 'Theta/Alpha',
            })

            plot_band_ratios(
                {
                    'ratios': band_ratios_df,
                    'statistics': statistical_df['statistics'],
                },
                img_path=img_dir / 'band_ratios.png',
                clip_percentile=95.0,
                smooth_window=5
            )
            results['band_ratios_img'] = 'band_ratios.png'
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
        alpha_beta_val = None
        beta_theta_val = None

        # バンド比率: セグメント分析から取得（Statistical DFベース、最も信頼性が高い）
        if 'segment_table' in results:
            segment_df = results['segment_table']

            # θ/α比（Bels差分）: 総合スコア計算用
            if 'θ/α比 (Bels)' in segment_df.columns:
                theta_alpha_values = segment_df['θ/α比 (Bels)'].dropna()
                if len(theta_alpha_values) > 0:
                    theta_alpha_val = theta_alpha_values.mean()

            # α/β比（実数値）
            if 'α/β比' in segment_df.columns:
                alpha_beta_values = segment_df['α/β比'].dropna()
                if len(alpha_beta_values) > 0:
                    alpha_beta_val = alpha_beta_values.mean()

            # β/θ比（実数値）- 新規追加
            if 'β/θ比' in segment_df.columns:
                beta_theta_values = segment_df['β/θ比'].dropna()
                if len(beta_theta_values) > 0:
                    beta_theta_val = beta_theta_values.mean()
                    results['beta_theta_ratio'] = beta_theta_val  # レポート用に保存

        # フォールバック: Statistical DFから直接取得
        if theta_alpha_val is None and statistical_df is not None:
            stats_df = statistical_df['statistics']
            theta_alpha_row = stats_df[stats_df['Metric'] == 'theta_alpha_bels_Mean']
            if not theta_alpha_row.empty:
                theta_alpha_val = theta_alpha_row['Value'].iloc[0]

        if alpha_beta_val is None and statistical_df is not None:
            stats_df = statistical_df['statistics']
            alpha_beta_row = stats_df[stats_df['Metric'] == 'alpha_beta_Mean']
            if not alpha_beta_row.empty:
                alpha_beta_val = alpha_beta_row['Value'].iloc[0]

        faa_val = None
        if 'faa_stats' in results:
            faa_stats_df = results['faa_stats']
            faa_row = faa_stats_df[faa_stats_df['Metric'] == 'Mean FAA']
            if not faa_row.empty:
                faa_val = faa_row['Value'].iloc[0]

        iaf_cv_val = None

        # セグメント分析からIAF変動係数を優先的に計算（より安定した評価）
        if 'segment_table' in results:
            segment_df = results['segment_table']
            if 'IAF平均 (Hz)' in segment_df.columns:
                iaf_values = segment_df['IAF平均 (Hz)'].dropna()
                if len(iaf_values) > 1:
                    iaf_mean = iaf_values.mean()
                    iaf_std = iaf_values.std()
                    if iaf_mean > 0:
                        iaf_cv_val = iaf_std / iaf_mean

        # セグメント分析で取得できない場合、PAF時間推移から取得
        if iaf_cv_val is None and 'paf_time_stats' in results:
            paf_stats = results['paf_time_stats']
            if '変動係数 (%)' in paf_stats:
                iaf_cv_val = paf_stats['変動係数 (%)'] / 100.0  # パーセントから0-1に変換

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
            alpha_beta_ratio=alpha_beta_val,
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
        description='Muse脳波データの基本分析とレポート生成（リファクタリング版）'
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

    args = parser.parse_args()

    # パスの検証
    if not args.data.exists():
        print(f'エラー: データファイルが見つかりません: {args.data}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    # 分析実行
    run_full_analysis(args.data, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
