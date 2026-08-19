"""
EEGスペクトル解析の解析ステップ

band power時系列プロット・MNE RAW準備・PSD/スペクトログラム・PAF/ITF・
PSDピーク（SMR harmonics含む）・Alpha Power・FAA・Spectral Entropy・
Frontal Midline Theta・SMR を扱う。

`prepare_mne_and_spectral` の内部（RAW準備・アーチファクト要約・生データ
プレビュー・PSD計算・スペクトログラム・PAF・ITF）は元コードで try/except に
包まれていないため、ここでも例外は捕捉せずそのまま伝播させる
（失敗時は run_full_analysis 全体がクラッシュする現行挙動を維持する）。
"""

import pandas as pd

from lib import (
    analyze_psd_peaks as compute_psd_peaks,
)
from lib import (
    calculate_alpha_power,
    calculate_alpha_power_from_raw,
    calculate_frontal_asymmetry,
    calculate_frontal_theta,
    calculate_itf,
    calculate_paf,
    calculate_psd,
    calculate_smr,
    calculate_spectral_entropy,
    calculate_spectral_entropy_time_series,
    calculate_spectrogram_all_channels,
    filter_eeg_quality,
    find_band_peak,
    fit_aperiodic,
    get_psd_peak_frequencies,
    oscillatory_band_power,
    prepare_mne_raw,
)
from lib.sensors.eeg.artifact import summarize_artifacts
from lib.sensors.eeg.constants import FREQ_BANDS
from lib.sensors.eeg.visualization import (
    plot_aperiodic_fit,
    plot_band_power_time_series,
    plot_paf,
    plot_psd,
    plot_psd_peaks,
    plot_raw_preview,
    plot_spectrogram_grid,
)

from .step import analysis_step


def plot_band_power_series(df, img_dir, results):
    """バンドパワー時系列（Museアプリ風）をプロットする。

    元コードで try/except に包まれていないため、失敗時は例外をそのまま伝播させる。
    """
    print('プロット中: バンドパワー時系列...')
    df_quality, quality_mask = filter_eeg_quality(df)
    df_for_band = df_quality if not df_quality.empty else df
    plot_band_power_time_series(
        df_for_band,
        img_path=img_dir / 'band_power_time_series.png',
        rolling_window=200,
        resample_interval='10s',
        smooth_window=5,
        clip_percentile=98.0
    )
    results['band_power_img'] = 'band_power_time_series.png'
    results['band_power_quality_ratio'] = float(quality_mask.mean())
    return df_quality


@analysis_step('PSDピーク分析')
def analyze_psd_peaks(psd_dict, paf_dict, img_dir, results):
    print('計算中: PSDピーク分析...')
    iaf_for_peaks = paf_dict.get('iaf_peak', paf_dict.get('iaf'))
    psd_peaks_result = compute_psd_peaks(psd_dict, iaf=iaf_for_peaks)
    results['harmonics_table'] = psd_peaks_result.peaks_table
    results['harmonics_stats'] = psd_peaks_result.statistics
    results['harmonics_result'] = psd_peaks_result

    print('プロット中: PSDピーク分析...')
    plot_psd_peaks(psd_peaks_result, psd_dict, img_path=img_dir / 'psd_peaks.png')
    results['harmonics_img'] = 'psd_peaks.png'
    return psd_peaks_result


@analysis_step('Alpha Power解析')
def analyze_alpha_power(df, results):
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
    return alpha_power_result


@analysis_step('FAA解析')
def analyze_faa(df, raw_unfiltered, results):
    print('計算中: Frontal Alpha Asymmetry...')
    faa_result = calculate_frontal_asymmetry(df, raw=raw_unfiltered)
    results['faa_stats'] = faa_result.statistics
    results['faa_interpretation'] = faa_result.metadata.get('interpretation')
    return faa_result


@analysis_step('Spectral Entropy解析')
def analyze_spectral_entropy(psd_dict, tfr_primary, df, results):
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
    return se_result


def prepare_mne_and_spectral(df, img_dir, results, artifact_window_samples):
    """MNE RAW準備、PSD/スペクトログラム/PAF/ITF計算、および付随するEEGスペクトル
    解析ステップ（PSDピーク・Alpha Power・FAA・Spectral Entropy）を実行する。

    RAW準備〜ITF計算までは元コードで try/except に包まれていないため、失敗時は
    例外をそのまま伝播させる（run_full_analysis 全体がクラッシュする現行挙動を維持）。

    Returns
    -------
    tuple
        (raw, raw_unfiltered, psd_dict, paf_dict, tfr_primary)
    """
    print('準備中: MNE RAWデータ...')
    mne_dict = prepare_mne_raw(df)
    raw = None
    raw_unfiltered = None  # Fmθ/FAA用のフィルタなしraw
    psd_dict = None
    paf_dict = None
    tfr_primary = None

    if mne_dict:
        raw = mne_dict['raw']
        print(f'検出されたチャネル: {mne_dict["channels"]}')
        print(f'推定サンプリングレート: {mne_dict["sfreq"]:.2f} Hz')

        # RAW振幅ベースの品質統計（HSIでは検出できないアーチファクトの可視化）
        print('計算中: RAW振幅品質...')
        artifact_summary = summarize_artifacts(
            raw.get_data() * 1e6,
            mne_dict['channels'],
            window_samples=artifact_window_samples,
        )
        results['artifact_summary'] = artifact_summary
        print(f'  振幅による除外率: {artifact_summary["rejected_ratio"] * 100:.1f}%')

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

        # ITF分析
        print('計算中: Individual Theta Frequency...')
        itf_dict = calculate_itf(psd_dict)
        results['itf'] = {
            'value': itf_dict['itf'],
            'std': itf_dict['itf_std'],
            'peak': itf_dict['itf_peak'],
            'cog': itf_dict['itf_cog']
        }

        # 非周期成分（1/f）分離
        # 全チャネル平均PSDに対してspecparamでフィットする。
        # IAF/ITFの窓内argmaxが1/fの単調減少域で窓端に張り付く問題（Issue #31）を
        # 避けるため、ピーク検出はspecparamのイテレーティブなピーク除去に委ねる。
        print('計算中: 非周期成分（1/f）分離...')
        mean_psd = psd_dict['psds'].mean(axis=0)
        aperiodic_result = fit_aperiodic(psd_dict['freqs'], mean_psd)
        if aperiodic_result is not None:
            theta_band = FREQ_BANDS['Theta'][:2]
            alpha_band = FREQ_BANDS['Alpha'][:2]
            results['aperiodic'] = {
                'result': aperiodic_result,
                'offset': aperiodic_result.offset,
                'exponent': aperiodic_result.exponent,
                'r_squared': aperiodic_result.r_squared,
                'error': aperiodic_result.error,
                'n_peaks': aperiodic_result.n_peaks,
                'peaks': aperiodic_result.peaks,
                'theta_peak': find_band_peak(aperiodic_result, theta_band),
                'alpha_peak': find_band_peak(aperiodic_result, alpha_band),
                'theta_osc_db': oscillatory_band_power(
                    psd_dict['freqs'], mean_psd, aperiodic_result, theta_band
                ),
                'alpha_osc_db': oscillatory_band_power(
                    psd_dict['freqs'], mean_psd, aperiodic_result, alpha_band
                ),
            }

            print('プロット中: 非周期成分（1/f）フィット...')
            plot_aperiodic_fit(
                psd_dict['freqs'], mean_psd, aperiodic_result,
                img_path=img_dir / 'aperiodic.png',
            )
            results['aperiodic_img'] = 'aperiodic.png'
        else:
            print('  警告: 非周期成分フィットに失敗したため、非周期成分セクションをスキップします。')

        # PSDピーク分析（SMR含む）
        analyze_psd_peaks(psd_dict, paf_dict, img_dir, results)

        # Alpha Power (Brain Recharge Score) 解析
        analyze_alpha_power(df, results)

        # FAA解析
        analyze_faa(df, raw_unfiltered, results)

        # Spectral Entropy解析
        analyze_spectral_entropy(psd_dict, tfr_primary, df, results)

    return raw, raw_unfiltered, psd_dict, paf_dict, tfr_primary


@analysis_step('Fmθ解析')
def analyze_frontal_theta_step(df, raw_unfiltered, results):
    print('計算中: Frontal Midline Theta...')
    fmtheta_result = calculate_frontal_theta(df, raw=raw_unfiltered if raw_unfiltered else None)
    results['frontal_theta_stats'] = fmtheta_result.statistics
    results['frontal_theta_increase'] = fmtheta_result.metadata.get('increase_rate_percent')
    return fmtheta_result


@analysis_step('SMR解析')
def analyze_smr_step(df, raw_unfiltered, results):
    print('計算中: SMR (12-15Hz, AF領域)...')
    smr_result = calculate_smr(df, raw=raw_unfiltered if raw_unfiltered else None)
    results['smr_stats'] = smr_result.statistics
    results['smr_increase'] = smr_result.metadata.get('increase_rate_percent')
    return smr_result
