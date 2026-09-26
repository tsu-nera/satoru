"""
Phase 3のdB変換を検証するテストコード

Fmθ、FAAの計算がμV²/lnからdB (10*log10) に正しく変換されたことを確認する。
"""

import numpy as np
import pandas as pd

from lib.segment_analysis import calculate_meditation_score
from lib.sensors.eeg.frontal_asymmetry import calculate_frontal_asymmetry
from lib.sensors.eeg.frontal_theta import calculate_frontal_theta


def test_fmtheta_bels_output():
    """Fmθ計算がdB単位で出力されることを確認"""
    # サンプルデータ作成（簡易的なダミーデータ）
    np.random.seed(42)
    n_samples = 256 * 60  # 1分間のデータ（256Hz）

    df = pd.DataFrame({
        'TimeStamp': pd.date_range('2025-01-01', periods=n_samples, freq='3.90625ms'),
        'RAW_AF7': np.random.randn(n_samples) * 10 + 50,
        'RAW_AF8': np.random.randn(n_samples) * 10 + 50,
    })

    result = calculate_frontal_theta(df, band=(6.0, 7.0))

    # dB単位であることを確認
    assert result.metadata['unit'] == 'dB'
    assert result.metadata['method'] == 'mne_hilbert_db'

    # dBの妥当な範囲（10*log10(μV²)なので、数値的に合理的な範囲）
    mean_val = result.time_series.mean()
    assert -20 < mean_val < 40, f"Fmθ mean ({mean_val}) should be in dB range"

    # 時系列データがdBであることを確認（対数スケールなので広い範囲を許容）
    assert result.time_series.min() > -100  # dB下限
    assert result.time_series.max() < 100   # dB上限

    print("✓ Fmθ dB conversion test passed")
    print(f"  Mean: {mean_val:.2f} dB")
    print(f"  Range: {result.time_series.min():.2f} - {result.time_series.max():.2f} dB")


def test_faa_bels_output():
    """FAA計算がdB差分で出力されることを確認"""
    np.random.seed(42)
    n_samples = 256 * 60  # 1分間のデータ

    # use_mastoid_reference がデフォルトで True のため、TP9/TP10 も必要
    df = pd.DataFrame({
        'TimeStamp': pd.date_range('2025-01-01', periods=n_samples, freq='3.90625ms'),
        'RAW_AF7': np.random.randn(n_samples) * 10 + 50,
        'RAW_AF8': np.random.randn(n_samples) * 10 + 55,  # 右が少し高め
        'RAW_TP9': np.random.randn(n_samples) * 10 + 50,
        'RAW_TP10': np.random.randn(n_samples) * 10 + 50,
    })

    result = calculate_frontal_asymmetry(df)

    # dB単位であることを確認
    assert result.metadata['unit'] == 'dB'
    assert result.metadata['method'] == 'mne_hilbert_db'

    # 統計データがdB単位であることを確認
    stats_df = result.statistics
    assert (stats_df['Unit'] == 'dB').sum() == 5  # 5つのdB単位指標（Mean, Median, Std, First, Second）

    # dB差分の妥当な範囲（対数スケールなので広めに）
    mean_faa = result.time_series.mean()
    assert -20 < mean_faa < 20, f"FAA mean ({mean_faa}) should be in dB diff range"

    # 左右パワーがdBであることを確認（対数スケールなので広い範囲を許容）
    assert result.left_power.min() > -100
    assert result.left_power.max() < 100
    assert result.right_power.min() > -100
    assert result.right_power.max() < 100

    print("✓ FAA dB conversion test passed")
    print(f"  Mean FAA: {mean_faa:.2f} dB")
    print(f"  Left power range: {result.left_power.min():.2f} - {result.left_power.max():.2f} dB")
    print(f"  Right power range: {result.right_power.min():.2f} - {result.right_power.max():.2f} dB")


def test_fmtheta_is_gain_invariant():
    """Fmθは全帯域に対する相対値なので、信号全体のゲインが変わっても値が変わらない"""
    rng = np.random.default_rng(0)
    n_samples = 256 * 60
    t = np.arange(n_samples) / 256
    base = {ch: rng.standard_normal(n_samples) * 10 + 5 * np.sin(2 * np.pi * 6.5 * t) for ch in ('RAW_AF7', 'RAW_AF8')}
    stamps = pd.date_range('2025-01-01', periods=n_samples, freq='3.90625ms')

    def fm_mean(gain):
        df = pd.DataFrame({'TimeStamp': stamps, **{ch: v * gain for ch, v in base.items()}})
        return calculate_frontal_theta(df).time_series.mean()

    # ゲイン3倍は絶対パワーで+9.5dBに相当する
    assert abs(fm_mean(3.0) - fm_mean(1.0)) < 0.1


def test_fmtheta_ignores_line_noise_on_unfiltered_raw():
    """ノッチ未適用のrawを渡しても、電源ノイズで分母が膨らまない（レポートはこの経路で呼ぶ）"""
    from lib.sensors.eeg.preprocessing import prepare_mne_raw

    rng = np.random.default_rng(0)
    n_samples = 256 * 60
    t = np.arange(n_samples) / 256
    stamps = pd.date_range('2025-01-01', periods=n_samples, freq='3.90625ms')
    clean = {ch: rng.standard_normal(n_samples) * 10 + 5 * np.sin(2 * np.pi * 6.5 * t) for ch in ('RAW_AF7', 'RAW_AF8')}

    def fm_mean(line_amp):
        line = line_amp * np.sin(2 * np.pi * 50 * t)
        df = pd.DataFrame({'TimeStamp': stamps, **{ch: v + line for ch, v in clean.items()}})
        raw = prepare_mne_raw(df, apply_bandpass=False, apply_notch=False)['raw']
        return calculate_frontal_theta(df, raw=raw).time_series.mean()

    assert abs(fm_mean(100.0) - fm_mean(0.0)) < 1.0


def test_meditation_score_normalization():
    """総合スコアの正規化範囲がdBに対応していることを確認"""
    # Fmθ（全帯域に対する相対値）: -15.0 ~ -7.0 dB の範囲でテスト（実測分布に基づくレンジ）
    score_min = calculate_meditation_score(fmtheta=-15.0)
    score_max = calculate_meditation_score(fmtheta=-7.0)
    score_mid = calculate_meditation_score(fmtheta=-11.0)

    assert score_min['scores']['fmtheta'] == 0.0  # min値で0
    assert score_max['scores']['fmtheta'] == 1.0  # max値で1
    assert 0.4 < score_mid['scores']['fmtheta'] < 0.6  # 中間値で約0.5

    # 実測レンジのFmθがクリップされず中間域に入ること（回帰防止）
    for observed in (-14.57, -11.29, -9.87):
        s = calculate_meditation_score(fmtheta=observed)['scores']['fmtheta']
        assert 0.0 < s < 1.0, f'Fmθ={observed} がクリップされている'

    # FAA: -20.0 ~ 20.0 dB の範囲でテスト
    faa_min = calculate_meditation_score(faa=-20.0)
    faa_max = calculate_meditation_score(faa=20.0)
    faa_mid = calculate_meditation_score(faa=0.0)

    assert faa_min['scores']['faa'] == 0.0
    assert faa_max['scores']['faa'] == 1.0
    assert 0.4 < faa_mid['scores']['faa'] < 0.6

    print("✓ Meditation score normalization test passed")
    print(
        f"  Fmθ normalization: {score_min['scores']['fmtheta']:.2f} / "
        f"{score_mid['scores']['fmtheta']:.2f} / {score_max['scores']['fmtheta']:.2f}"
    )
    print(
        f"  FAA normalization: {faa_min['scores']['faa']:.2f} / "
        f"{faa_mid['scores']['faa']:.2f} / {faa_max['scores']['faa']:.2f}"
    )


def test_bels_conversion_consistency():
    """μV²とBelsの変換が数学的に一貫していることを確認"""
    # μV²の値とそれに対応するBels値
    test_cases = [
        (1.0, 0.0),      # 10*log10(1) = 0
        (10.0, 10.0),    # 10*log10(10) = 10
        (100.0, 20.0),   # 10*log10(100) = 20
        (1000.0, 30.0),  # 10*log10(1000) = 30
    ]

    for uv2, expected_bels in test_cases:
        calculated_bels = 10 * np.log10(uv2)
        assert np.isclose(calculated_bels, expected_bels, atol=0.01), \
            f"10*log10({uv2}) should be {expected_bels}, got {calculated_bels}"

    # Bels差分とln差分の関係を確認
    # ln(b) - ln(a) = ln(b/a)
    # 10*log10(b) - 10*log10(a) = 10*log10(b/a)
    a, b = 50.0, 100.0
    ln_diff = np.log(b) - np.log(a)
    bels_diff = 10 * np.log10(b) - 10 * np.log10(a)

    # ln(2) ≈ 0.693, 10*log10(2) ≈ 3.01
    assert np.isclose(ln_diff, 0.693, atol=0.01)
    assert np.isclose(bels_diff, 3.01, atol=0.01)

    print("✓ Bels conversion consistency test passed")
    for uv2, _expected_bels in test_cases:
        print(f"  10*log10({uv2}) = {10 * np.log10(uv2):.2f} Bels")


if __name__ == '__main__':
    print("=== Phase 3 Bels Conversion Tests ===\n")

    test_fmtheta_bels_output()
    print()
    test_faa_bels_output()
    print()
    test_meditation_score_normalization()
    print()
    test_bels_conversion_consistency()
    print()

    print("=== All tests passed! ===")
