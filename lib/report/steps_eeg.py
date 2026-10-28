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

import numpy as np
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
from lib.sensors.eeg.artifact import expand_window_mask, summarize_artifacts
from lib.sensors.eeg.constants import FREQ_BANDS
from lib.sensors.eeg.frequency import calculate_psd_masked, mask_spectrogram
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

# セッションログの横断比較用に固定するチャネル（Muse既定のFpz基準のまま使う）
FRONTAL_CHANNELS = ('RAW_AF7', 'RAW_AF8')

# 耳側（TP9/TP10）のIAFを記録するのに必要な有効窓の合計（秒、2チャネルの合計）。
# TPは冬季に接触が悪い日があり、短い有効窓から出したIAFは信頼できないため空欄にする
TP_CHANNELS = ('RAW_TP9', 'RAW_TP10')
TP_MIN_VALID_SEC = 180

# 耳側IAFを探すα帯域（Hz）
TP_IAF_BAND = (7.0, 13.0)


def plot_band_power_series(df, img_dir, results, tap_df=None):
    """バンドパワー時系列（Museアプリ風）をプロットする。

    `tap_df` があればタップ時刻を縦線でオーバーレイする（任意データ）。

    元コードで try/except に包まれていないため、失敗時は例外をそのまま伝播させる。
    """
    print('プロット中: バンドパワー時系列...')
    df_quality, quality_mask = filter_eeg_quality(df)
    df_for_band = df_quality if not df_quality.empty else df

    event_times = None
    if tap_df is not None:
        event_times = tap_df.loc[tap_df['event'] == 'tap', 'TimeStamp']

    plot_band_power_time_series(
        df_for_band,
        img_path=img_dir / 'band_power_time_series.png',
        rolling_window=200,
        resample_interval='10s',
        smooth_window=5,
        clip_percentile=98.0,
        event_times=event_times
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


def mask_invalid_rows(df, columns_by_channel, mne_dict, valid_mask, window_samples):
    """
    合成マスクで無効なチャネル×窓に当たる行の値をNaNにしたdfのコピーを返す。

    rawに入らなかった行（HSIがBadの行）も無効として扱う。

    Parameters
    ----------
    columns_by_channel : dict
        RAWチャネル名 → NaNにする列名（例: {'RAW_TP9': 'Alpha_TP9'}）
    """
    masked = df.copy()
    row_index = mne_dict['row_index']
    for ch, col in columns_by_channel.items():
        if col not in masked.columns or ch not in mne_dict['channels']:
            continue
        ch_idx = mne_dict['channels'].index(ch)
        sample_valid = expand_window_mask(valid_mask[ch_idx], len(row_index), window_samples)
        keep = pd.Series(False, index=masked.index)
        keep.loc[row_index[sample_valid]] = True
        masked.loc[~keep, col] = np.nan
    return masked


@analysis_step('Alpha Power解析')
def analyze_alpha_power(df, results, mne_dict, valid_mask, window_samples):
    print('計算中: Alpha Power (Brain Recharge Score)...')
    sites = ('TP9', 'AF7', 'AF8', 'TP10')
    # Alpha列があるか確認（Mind Monitor形式）
    alpha_cols = [f'Alpha_{s}' for s in sites]
    has_alpha_data = all(
        col in df.columns and df[col].notna().any()
        for col in alpha_cols
    )
    if has_alpha_data:
        df_masked = mask_invalid_rows(
            df, {f'RAW_{s}': f'Alpha_{s}' for s in sites}, mne_dict, valid_mask, window_samples
        )
        usable_cols = [col for col in alpha_cols if df_masked[col].notna().any()]
        alpha_power_result = calculate_alpha_power(df_masked, channels=usable_cols or alpha_cols)
    else:
        # RAW EEGからAlpha Powerを計算（Muse App OSC形式）
        # サンプル単位のフィルタ処理なので窓単位のマスクは当てず、全窓が無効なチャネルだけ外す
        print('  Alpha列が空のため、RAW EEGから計算...')
        usable = [
            ch for idx, ch in enumerate(mne_dict['channels']) if valid_mask[idx].any()
        ]
        alpha_power_result = calculate_alpha_power_from_raw(
            df, channels=usable or list(mne_dict['channels'])
        )
    results['alpha_power_score'] = alpha_power_result.score
    results['alpha_power_db'] = alpha_power_result.alpha_db
    results['alpha_power_stats'] = alpha_power_result.statistics
    results['alpha_power_metadata'] = alpha_power_result.metadata
    return alpha_power_result


@analysis_step('FAA解析')
def analyze_faa(df, raw_unfiltered, results, valid_mask, window_samples):
    print('計算中: Frontal Alpha Asymmetry...')
    # TP9とTP10が両方有効な窓だけlinked mastoid基準にし、それ以外の窓はMuse既定の基準（Fpz）のまま
    channels = raw_unfiltered.ch_names if raw_unfiltered is not None else []
    mastoids_present = {'RAW_TP9', 'RAW_TP10'} <= set(channels)
    mastoid_valid = None
    if mastoids_present:
        mastoid_valid = (
            valid_mask[channels.index('RAW_TP9')] & valid_mask[channels.index('RAW_TP10')]
        )
    faa_result = calculate_frontal_asymmetry(
        df,
        raw=raw_unfiltered,
        use_mastoid_reference=mastoids_present,
        mastoid_valid=mastoid_valid,
        window_samples=window_samples,
    )
    results['faa_stats'] = faa_result.statistics
    results['faa_interpretation'] = faa_result.metadata.get('interpretation')
    return faa_result


def compute_tp_metrics(raw, valid_mask, window_samples):
    """
    耳側（TP9/TP10）の有効率とIAFを返す（セッションログの追加列用）。

    tp9_valid_pct / tp10_valid_pct は全窓に対するそのチャネルの有効窓の割合（%）。
    チャネルが無ければNaN。tp_iaf_hz はTP9とTP10の有効窓の合計が TP_MIN_VALID_SEC
    以上のときだけ、線スペクトル除去後の平均PSDのα帯域ピークを返し、足りなければNaN。

    Parameters
    ----------
    raw : mne.io.RawArray
        フィルタ済みraw（valid_maskの行と同じチャネル並び）
    valid_mask : np.ndarray
        (n_channels, n_windows) のbool配列
    window_samples : int
        マスクの窓長（サンプル）
    """
    nan = float('nan')
    metrics = {'tp9_valid_pct': nan, 'tp10_valid_pct': nan, 'tp_iaf_hz': nan}
    present = [ch for ch in TP_CHANNELS if ch in raw.ch_names]
    if not present or valid_mask.shape[1] == 0:
        return metrics

    tp_idx = [raw.ch_names.index(ch) for ch in present]
    for ch, idx in zip(present, tp_idx):
        metrics[f'{ch.replace("RAW_", "").lower()}_valid_pct'] = float(valid_mask[idx].mean() * 100)

    sfreq = raw.info['sfreq']
    valid_sec = float(valid_mask[tp_idx].sum()) * window_samples / sfreq
    if valid_sec < TP_MIN_VALID_SEC:
        return metrics

    psd_dict = calculate_psd_masked(raw.copy().pick(present), valid_mask[tp_idx], window_samples)
    if psd_dict is None:
        return metrics

    freqs = psd_dict['freqs']
    band = (freqs >= TP_IAF_BAND[0]) & (freqs <= TP_IAF_BAND[1])
    mean_psd = psd_dict['psds'].mean(axis=0)
    metrics['tp_iaf_hz'] = float(freqs[band][mean_psd[band].argmax()])
    return metrics


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

    接触不良や振幅アーチファクトのあるチャネル×窓は、チャネルごと捨てずに
    合成マスク（振幅判定 AND クリップ判定）で窓単位に除外する。HSIは除外に使わない。

    Returns
    -------
    tuple
        (raw, raw_unfiltered, psd_dict, paf_dict, tfr_primary, unfiltered_uv)
        unfiltered_uv は raw とサンプルが揃ったフィルタ前の生値（n_channels, n_samples, μV）
    """
    print('準備中: MNE RAWデータ...')
    mne_dict = prepare_mne_raw(df)
    raw = None
    raw_unfiltered = None  # Fmθ/FAA用のフィルタなしraw
    psd_dict = None
    paf_dict = None
    tfr_primary = None
    unfiltered_uv = None

    if mne_dict:
        raw = mne_dict['raw']
        print(f'検出されたチャネル: {mne_dict["channels"]}')
        print(f'推定サンプリングレート: {mne_dict["sfreq"]:.2f} Hz')

        # フィルタ前のraw（クリップ判定とFmθ/FAA計算用）
        # (Fmθ/FAAの関数は内部で独自のバンドパスフィルタを適用するため、フィルタなしを渡す)
        # 行の絞り込みと補間はrawと同じで、フィルタを掛けないので値は元のμVのまま
        # （0/1450μVへの張り付きが残る）。合成マスクの窓境界もそのまま使える
        mne_dict_unfiltered = prepare_mne_raw(df, apply_bandpass=False, apply_notch=False)
        if mne_dict_unfiltered:
            raw_unfiltered = mne_dict_unfiltered['raw']
            unfiltered_uv = raw_unfiltered.get_data() * 1e6

        # 振幅（HSIでは検出できないアーチファクト）とクリップ（増幅器の飽和）を合成した有効マスク。
        # HSIは参考列としてだけ表示する
        print('計算中: RAW振幅・クリップ品質...')
        artifact_summary = summarize_artifacts(
            raw.get_data() * 1e6,
            mne_dict['channels'],
            window_samples=artifact_window_samples,
            unfiltered_uv=unfiltered_uv,
            hsi=mne_dict['hsi'],
        )
        results['artifact_summary'] = artifact_summary
        valid_mask = artifact_summary['valid_mask']
        print(f'  除外率（振幅＋クリップ）: {artifact_summary["rejected_ratio"] * 100:.1f}%')
        for ch, ch_valid in zip(mne_dict['channels'], valid_mask):
            if ch_valid.size and not ch_valid.all():
                print(f'  {ch}: 有効率 {ch_valid.mean() * 100:.1f}%')

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

        # PSD計算（チャネルごとに有効な窓だけを平均する）
        print('計算中: パワースペクトル密度...')
        psd_dict = calculate_psd_masked(raw, valid_mask, artifact_window_samples)
        if psd_dict is None:
            print('  警告: 有効な窓が無いため、マスクを使わずにPSDを計算します。')
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
        # 無効な窓の時間帯は空白にし、全窓が無効なチャネルは出さない
        coverage = {}
        for channel in list(tfr_results):
            ch_valid = valid_mask[raw.ch_names.index(channel)]
            if not ch_valid.any():
                del tfr_results[channel]
                continue
            coverage[channel] = float(ch_valid.mean())
            mask_spectrogram(
                tfr_results[channel], ch_valid, artifact_window_samples, raw.info['sfreq']
            )

        if tfr_results:
            print('プロット中: スペクトログラム（全チャネル）...')
            plot_spectrogram_grid(tfr_results, img_path=img_dir / 'spectrogram.png')
            results['spectrogram_img'] = 'spectrogram.png'

            # 時系列解析で使うチャネル: 有効率が最も高いもの（同率ならTP9→AF7→AF8→TP10の順）
            preferred_channels = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
            tfr_primary_channel = max(
                tfr_results,
                key=lambda ch: (
                    coverage[ch],
                    -preferred_channels.index(ch) if ch in preferred_channels else -99,
                ),
            )
            tfr_primary = tfr_results[tfr_primary_channel]

        # 耳側（TP9/TP10）の有効率とIAF（セッションログの追加列）
        results['tp_metrics'] = compute_tp_metrics(raw, valid_mask, artifact_window_samples)

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
        aperiodic_summary = summarize_aperiodic(psd_dict['freqs'], mean_psd)

        # セッションログ用: チャネル構成に左右されないよう前頭（AF7/AF8）だけで別にフィットする
        frontal_idx = [
            idx for idx, ch in enumerate(psd_dict['channels']) if ch in FRONTAL_CHANNELS
        ]
        if frontal_idx:
            results['aperiodic_frontal'] = summarize_aperiodic(
                psd_dict['freqs'], psd_dict['psds'][frontal_idx].mean(axis=0)
            )

        if aperiodic_summary is not None:
            results['aperiodic'] = aperiodic_summary
            aperiodic_result = aperiodic_summary['result']

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
        analyze_alpha_power(df, results, mne_dict, valid_mask, artifact_window_samples)

        # FAA解析
        analyze_faa(df, raw_unfiltered, results, valid_mask, artifact_window_samples)

        # Spectral Entropy解析
        analyze_spectral_entropy(psd_dict, tfr_primary, df, results)

    return raw, raw_unfiltered, psd_dict, paf_dict, tfr_primary, unfiltered_uv


def summarize_aperiodic(freqs, mean_psd):
    """平均PSDにspecparamをフィットし、レポート・セッションログ用の値をまとめる。失敗時はNone。"""
    aperiodic_result = fit_aperiodic(freqs, mean_psd)
    if aperiodic_result is None:
        return None

    theta_band = FREQ_BANDS['Theta'][:2]
    alpha_band = FREQ_BANDS['Alpha'][:2]
    return {
        'result': aperiodic_result,
        'offset': aperiodic_result.offset,
        'exponent': aperiodic_result.exponent,
        'r_squared': aperiodic_result.r_squared,
        'error': aperiodic_result.error,
        'n_peaks': aperiodic_result.n_peaks,
        'peaks': aperiodic_result.peaks,
        'theta_peak': find_band_peak(aperiodic_result, theta_band),
        'alpha_peak': find_band_peak(aperiodic_result, alpha_band),
        'theta_osc_db': oscillatory_band_power(freqs, mean_psd, aperiodic_result, theta_band),
        'alpha_osc_db': oscillatory_band_power(freqs, mean_psd, aperiodic_result, alpha_band),
    }


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
