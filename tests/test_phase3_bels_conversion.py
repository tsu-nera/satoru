"""
Phase 3のBels変換を検証するテストコード

Fmθ、FAAの計算がμV²/lnからBelsに正しく変換されたことを確認する。
"""

import numpy as np
import pandas as pd

from lib.sensors.eeg.frontal_theta import calculate_frontal_theta
from lib.sensors.eeg.frontal_asymmetry import calculate_frontal_asymmetry
from lib.segment_analysis import calculate_meditation_score


def test_fmtheta_bels_output():
    """Fmθ計算がBels単位で出力されることを確認"""
    # サンプルデータ作成（簡易的なダミーデータ）
    np.random.seed(42)
    n_samples = 256 * 60  # 1分間のデータ（256Hz）

    df = pd.DataFrame({
        'TimeStamp': pd.date_range('2025-01-01', periods=n_samples, freq='3.90625ms'),
        'RAW_AF7': np.random.randn(n_samples) * 10 + 50,
        'RAW_AF8': np.random.randn(n_samples) * 10 + 50,
    })

    result = calculate_frontal_theta(df, band=(6.0, 7.0))

    # Bels単位であることを確認
    assert result.metadata['unit'] == 'Bels'
    assert result.metadata['method'] == 'mne_hilbert_bels'

    # 統計データがBels単位であることを確認
    stats_df = result.statistics
    assert (stats_df['Unit'] == 'Bels').sum() == 6  # 6つのBels単位指標（Mean, Median, Std, First, Second, Increase）

    # Belsの妥当な範囲（10*log10(μV²)なので、数値的に合理的な範囲）
    mean_val = result.time_series.mean()
    assert -20 < mean_val < 40, f"Fmθ mean ({mean_val}) should be in Bels range"

    # 時系列データがBelsであることを確認（対数スケールなので広い範囲を許容）
    assert result.time_series.min() > -100  # Bels下限
    assert result.time_series.max() < 100   # Bels上限

    print(f"✓ Fmθ Bels conversion test passed")
    print(f"  Mean: {mean_val:.2f} Bels")
    print(f"  Range: {result.time_series.min():.2f} - {result.time_series.max():.2f} Bels")


def test_faa_bels_output():
    """FAA計算がBels差分で出力されることを確認"""
    np.random.seed(42)
    n_samples = 256 * 60  # 1分間のデータ

    df = pd.DataFrame({
        'TimeStamp': pd.date_range('2025-01-01', periods=n_samples, freq='3.90625ms'),
        'RAW_AF7': np.random.randn(n_samples) * 10 + 50,
        'RAW_AF8': np.random.randn(n_samples) * 10 + 55,  # 右が少し高め
    })

    result = calculate_frontal_asymmetry(df)

    # Bels単位であることを確認
    assert result.metadata['unit'] == 'Bels'
    assert result.metadata['method'] == 'mne_hilbert_bels'

    # 統計データがBels単位であることを確認
    stats_df = result.statistics
    assert (stats_df['Unit'] == 'Bels').sum() == 5  # 5つのBels単位指標（Mean, Median, Std, First, Second）

    # Bels差分の妥当な範囲（対数スケールなので広めに）
    mean_faa = result.time_series.mean()
    assert -20 < mean_faa < 20, f"FAA mean ({mean_faa}) should be in Bels diff range"

    # 左右パワーがBelsであることを確認（対数スケールなので広い範囲を許容）
    assert result.left_power.min() > -100
    assert result.left_power.max() < 100
    assert result.right_power.min() > -100
    assert result.right_power.max() < 100

    print(f"✓ FAA Bels conversion test passed")
    print(f"  Mean FAA: {mean_faa:.2f} Bels")
    print(f"  Left power range: {result.left_power.min():.2f} - {result.left_power.max():.2f} Bels")
    print(f"  Right power range: {result.right_power.min():.2f} - {result.right_power.max():.2f} Bels")


def test_meditation_score_normalization():
    """総合スコアの正規化範囲がBelsに対応していることを確認"""
    # Fmθ: 17-23 Bels の範囲でテスト
    score_min = calculate_meditation_score(fmtheta=17.0)
    score_max = calculate_meditation_score(fmtheta=23.0)
    score_mid = calculate_meditation_score(fmtheta=20.0)

    assert score_min['scores']['fmtheta'] == 0.0  # min値で0
    assert score_max['scores']['fmtheta'] == 1.0  # max値で1
    assert 0.4 < score_mid['scores']['fmtheta'] < 0.6  # 中間値で約0.5

    # FAA: -2.0 ~ 2.0 Bels の範囲でテスト
    faa_min = calculate_meditation_score(faa=-2.0)
    faa_max = calculate_meditation_score(faa=2.0)
    faa_mid = calculate_meditation_score(faa=0.0)

    assert faa_min['scores']['faa'] == 0.0
    assert faa_max['scores']['faa'] == 1.0
    assert 0.4 < faa_mid['scores']['faa'] < 0.6

    print("✓ Meditation score normalization test passed")
    print(f"  Fmθ normalization: {score_min['scores']['fmtheta']:.2f} / {score_mid['scores']['fmtheta']:.2f} / {score_max['scores']['fmtheta']:.2f}")
    print(f"  FAA normalization: {faa_min['scores']['faa']:.2f} / {faa_mid['scores']['faa']:.2f} / {faa_max['scores']['faa']:.2f}")


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
    for uv2, expected_bels in test_cases:
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
