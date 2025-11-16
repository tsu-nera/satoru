"""
Statistical DataFrame IAF統合のテスト

Phase 2でIAF計算がstatistical_dataframe.pyに統合されたことを確認するテスト
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# テスト対象モジュール
from lib.statistical_dataframe import create_statistical_dataframe
from lib import prepare_mne_raw, load_mind_monitor_csv


class TestStatisticalDataframeIAF:
    """Statistical DataFrame IAF統合のテストクラス"""

    def test_statistical_dataframe_includes_iaf_key(self, sample_statistical_df):
        """Statistical DFに'iaf'キーが含まれることを確認"""
        assert 'iaf' in sample_statistical_df, "Statistical DFに'iaf'キーが含まれていません"

    def test_iaf_is_series(self, sample_statistical_df):
        """IAFがpandas Seriesであることを確認"""
        iaf = sample_statistical_df['iaf']
        assert isinstance(iaf, pd.Series), f"IAFはpd.Seriesであるべきですが、{type(iaf)}です"

    def test_iaf_has_values(self, sample_statistical_df):
        """IAFに値が含まれることを確認"""
        iaf = sample_statistical_df['iaf']
        assert len(iaf) > 0, "IAFに値が含まれていません"
        assert not iaf.isna().all(), "IAFの全値がNaNです"

    def test_iaf_values_in_alpha_range(self, sample_statistical_df):
        """IAFの値がアルファ帯域（8-13Hz）に収まることを確認"""
        iaf_values = sample_statistical_df['iaf'].dropna()

        assert len(iaf_values) > 0, "有効なIAF値が存在しません"

        min_iaf = iaf_values.min()
        max_iaf = iaf_values.max()

        assert min_iaf >= 8.0, f"IAFが8Hz未満の値（{min_iaf:.2f}Hz）を含んでいます"
        assert max_iaf <= 13.0, f"IAFが13Hzを超える値（{max_iaf:.2f}Hz）を含んでいます"

    def test_iaf_statistics_included(self, sample_statistical_df):
        """統計量にIAF指標が含まれることを確認"""
        stats_df = sample_statistical_df['statistics']
        iaf_stats = stats_df[stats_df['Category'] == 'IAF']

        assert len(iaf_stats) > 0, "統計量にIAFカテゴリが含まれていません"

        # 必須統計量の確認
        required_metrics = ['iaf_Mean', 'iaf_Median', 'iaf_Std', 'iaf_CV']
        actual_metrics = iaf_stats['Metric'].tolist()

        for metric in required_metrics:
            assert metric in actual_metrics, f"{metric}が統計量に含まれていません"

    def test_iaf_cv_calculation(self, sample_statistical_df):
        """IAF変動係数が正しく計算されることを確認"""
        # 手動でCV計算
        iaf_values = sample_statistical_df['iaf'].dropna()

        if len(iaf_values) == 0:
            pytest.skip("IAF値が存在しないためスキップ")

        expected_cv = iaf_values.std() / iaf_values.mean()

        # Statistical DFからCV取得
        stats_df = sample_statistical_df['statistics']
        iaf_cv_row = stats_df[stats_df['Metric'] == 'iaf_CV']

        assert not iaf_cv_row.empty, "iaf_CVが統計量に含まれていません"

        actual_cv = iaf_cv_row['Value'].iloc[0]

        assert np.isclose(actual_cv, expected_cv, rtol=1e-5), \
            f"IAF変動係数の計算が不正確です（期待値: {expected_cv:.6f}, 実際: {actual_cv:.6f}）"

    def test_iaf_index_matches_timestamps(self, sample_statistical_df):
        """IAFのインデックスがタイムスタンプであることを確認"""
        iaf = sample_statistical_df['iaf']

        assert isinstance(iaf.index, pd.DatetimeIndex), \
            "IAFのインデックスはpd.DatetimeIndexであるべきです"

        # band_powersと同じインデックスを持つことを確認
        band_powers = sample_statistical_df['band_powers']
        assert iaf.index.equals(band_powers.index), \
            "IAFとband_powersのインデックスが一致しません"

    def test_iaf_segment_count_matches_band_powers(self, sample_statistical_df):
        """IAFのセグメント数がバンドパワーと一致することを確認"""
        iaf = sample_statistical_df['iaf']
        band_powers = sample_statistical_df['band_powers']

        assert len(iaf) == len(band_powers), \
            f"IAFのセグメント数（{len(iaf)}）とband_powers（{len(band_powers)}）が一致しません"

    def test_iaf_mean_reasonable_value(self, sample_statistical_df):
        """IAF平均値が妥当な範囲（一般的に9-11Hz）にあることを確認"""
        iaf_values = sample_statistical_df['iaf'].dropna()

        if len(iaf_values) == 0:
            pytest.skip("IAF値が存在しないためスキップ")

        iaf_mean = iaf_values.mean()

        # 一般的なIAF範囲は9-11Hzだが、個人差があるため8-12Hzで許容
        assert 8.0 <= iaf_mean <= 12.0, \
            f"IAF平均値（{iaf_mean:.2f}Hz）が一般的な範囲（8-12Hz）外です"

    def test_iaf_statistics_display_names(self, sample_statistical_df):
        """IAF統計量のDisplayNameが日本語で設定されていることを確認"""
        stats_df = sample_statistical_df['statistics']
        iaf_stats = stats_df[stats_df['Category'] == 'IAF']

        for _, row in iaf_stats.iterrows():
            display_name = row['DisplayName']
            assert 'IAF' in display_name, f"DisplayNameにIAFが含まれていません: {display_name}"
            # 日本語が含まれることを確認（簡易的にひらがな・カタカナをチェック）
            assert any(c in display_name for c in '平均中央値標準偏差変動係数'), \
                f"DisplayNameに日本語が含まれていません: {display_name}"


# フィクスチャ
@pytest.fixture(scope="session")
def sample_csv_path():
    """
    テスト用CSVファイルパスを返すフィクスチャ

    Note: 実際のテストデータが存在する場合はそのパスを使用してください
    データが存在しない場合、このテストはスキップされます
    """
    # 実際のテストデータパス（環境に合わせて調整）
    possible_paths = [
        Path("/home/tsu-nera/repo/satoru/data/sample.csv"),
        Path("/home/tsu-nera/repo/satoru/data/test.csv"),
        Path("./data/sample.csv"),
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    # テストデータが見つからない場合はスキップ
    pytest.skip("テスト用CSVデータが見つかりませんでした")


@pytest.fixture(scope="session")
def sample_statistical_df(sample_csv_path):
    """
    サンプルデータからStatistical DataFrameを生成するフィクスチャ

    このフィクスチャはセッションスコープで1回だけ実行され、
    複数のテストで再利用されます
    """
    # データ読み込み
    df = load_mind_monitor_csv(sample_csv_path, filter_headband=False)

    # MNE RawArray準備
    mne_result = prepare_mne_raw(df)

    if mne_result is None or 'raw' not in mne_result:
        pytest.skip("MNE Rawデータの準備に失敗しました")

    # Statistical DF生成
    session_start = df['TimeStamp'].iloc[0]
    statistical_df = create_statistical_dataframe(
        mne_result['raw'],
        segment_minutes=3,
        warmup_minutes=0.0,
        session_start=session_start
    )

    return statistical_df


if __name__ == "__main__":
    # このファイルを直接実行した場合はpytestを実行
    pytest.main([__file__, "-v"])
