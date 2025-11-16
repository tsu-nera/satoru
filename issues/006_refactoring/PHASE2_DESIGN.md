# Phase 2 詳細設計: データ層の統一（IAF計算統合）

**Issue**: #006
**Phase**: 2
**作成日**: 2025-11-16
**ステータス**: 設計完了・実装待ち

---

## 📋 背景

### 現状の課題

**調査結果**（2025-11-16実施）:
- ✅ バンドパワー計算は既に`statistical_dataframe.py`に統一済み
- ✅ Mind Monitor CSV列の直接参照はほぼ存在しない
- ⚠️ **IAF計算が未統合**: `paf.py`が独立し、`segment_analysis.py`で外部から`iaf_series`を受け取る設計
- ⚠️ IAF変動係数が総合スコアで**12.5%**の重みを持つにも関わらず、計算が非統一

### Phase 2の目的

**IAF計算を`statistical_dataframe.py`に統合し、データ層の一貫性を完成させる**

---

## 🎯 実装計画

### タスク一覧

| タスク | ファイル | 所要時間 | 優先度 |
|-------|---------|---------|-------|
| 2.1: IAF計算を統合 | `statistical_dataframe.py` | 2-3時間 | 最高 |
| 2.2: segment_analysis簡略化 | `segment_analysis.py` | 1時間 | 高 |
| 2.3: generate_report簡略化 | `generate_report.py` | 1時間 | 高 |
| 2.4: テスト追加 | `tests/test_statistical_dataframe_iaf.py` | 1-2時間 | 中 |
| 2.5: 動作確認 | 全体 | 1時間 | 必須 |

**合計**: 6-8時間（1日）

---

## 📝 詳細実装仕様

### Task 2.1: IAF計算を`statistical_dataframe.py`に統合

#### ファイル: `lib/statistical_dataframe.py`

#### 変更内容

**1. `create_statistical_dataframe()`関数にIAF時系列計算を追加**

```python
def create_statistical_dataframe(
    raw: 'mne.io.RawArray',
    segment_minutes: int = 3,
    warmup_minutes: float = 0.0,
    session_start: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    """
    統一的なStatistical DataFrameを生成する。

    Returns
    -------
    dict
        {
            'band_powers': DataFrame,      # セグメント別バンドパワー時系列（Bels）
            'band_ratios': DataFrame,      # セグメント別バンド比率時系列
            'spectral_entropy': DataFrame, # セグメント別Spectral Entropy時系列
            'iaf': Series,                 # 🆕 Individual Alpha Frequency時系列（Hz）
            'statistics': DataFrame        # 統計サマリー（縦長形式）
        }
    """
```

**2. IAF計算ロジック（Spectral Entropy計算の直後に追加）**

```python
# Spectral Entropy計算（L144-165の後）
# ...

# 🆕 IAF（Individual Alpha Frequency）計算
# Epochsごとにアルファ帯域のピーク周波数を計算
iaf_values = []
alpha_range = (8.0, 13.0)

for epoch_idx in range(len(epochs)):
    # このエポックのPSD (n_channels, n_freqs)
    psd_epoch = psds[epoch_idx]

    # アルファ帯域のマスク
    alpha_mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
    alpha_freqs = freqs[alpha_mask]

    # 全チャネルの平均PSD（アルファ帯域）
    psd_alpha_avg = psd_epoch[:, alpha_mask].mean(axis=0)

    # ピーク周波数を検出
    peak_idx = psd_alpha_avg.argmax()
    iaf = alpha_freqs[peak_idx]
    iaf_values.append(iaf)

# IAF時系列をSeriesに変換
iaf_series = pd.Series(iaf_values, index=timestamps)
```

**3. IAF統計量を`statistics_df`に追加（L193-315の統計量計算セクション内）**

```python
# Spectral Entropy統計（L233-265の後）
# ...

# 🆕 IAF統計
iaf_clean = iaf_series.dropna()
if len(iaf_clean) > 0:
    # Z-score外れ値除去（閾値3.0）
    if len(iaf_clean) > 3:
        z_scores = np.abs(stats.zscore(iaf_clean))
        filtered_iaf = iaf_clean[z_scores < 3.0]
        if len(filtered_iaf) > 0:
            iaf_clean = filtered_iaf

    statistics_rows.extend([
        {
            'Category': 'IAF',
            'Metric': 'iaf_Mean',
            'Value': iaf_clean.mean(),
            'Unit': 'Hz',
            'DisplayName': 'IAF平均 (Hz)',
        },
        {
            'Category': 'IAF',
            'Metric': 'iaf_Median',
            'Value': iaf_clean.median(),
            'Unit': 'Hz',
            'DisplayName': 'IAF中央値 (Hz)',
        },
        {
            'Category': 'IAF',
            'Metric': 'iaf_Std',
            'Value': iaf_clean.std(),
            'Unit': 'Hz',
            'DisplayName': 'IAF標準偏差 (Hz)',
        },
        {
            'Category': 'IAF',
            'Metric': 'iaf_CV',
            'Value': iaf_clean.std() / iaf_clean.mean() if iaf_clean.mean() > 0 else np.nan,
            'Unit': 'ratio',
            'DisplayName': 'IAF変動係数',
        },
    ])
```

**4. 戻り値に`iaf_series`を追加（L317-322）**

```python
return {
    'band_powers': band_powers_df,
    'band_ratios': band_ratios_df,
    'spectral_entropy': se_df,
    'iaf': iaf_series,  # 🆕 追加
    'statistics': statistics_df,
}
```

---

### Task 2.2: `segment_analysis.py`を簡略化

#### ファイル: `lib/segment_analysis.py`

#### 変更内容

**1. 関数シグネチャの変更（L68-75）**

```python
# Before
def calculate_segment_analysis(
    df_clean: pd.DataFrame,
    fmtheta_series: pd.Series,
    statistical_df: Dict[str, pd.DataFrame],
    segment_minutes: int = 5,
    iaf_series: Optional[pd.Series] = None,  # ❌ 削除
    warmup_minutes: float = 0.0,
) -> SegmentAnalysisResult:

# After
def calculate_segment_analysis(
    df_clean: pd.DataFrame,
    fmtheta_series: pd.Series,
    statistical_df: Dict[str, pd.DataFrame],
    segment_minutes: int = 5,
    warmup_minutes: float = 0.0,
) -> SegmentAnalysisResult:
    """
    セッションを一定時間のセグメントに分割し、主要指標を算出する。

    Notes
    -----
    バンドパワー・比率・IAFはstatistical_dfから自動取得されます。
    df_cleanのバンドパワー列は使用されません。
    """
```

**2. バリデーション更新（L105-109）**

```python
# Before
required_keys = ['band_powers', 'band_ratios', 'spectral_entropy']

# After
required_keys = ['band_powers', 'band_ratios', 'spectral_entropy', 'iaf']  # 🆕 iaf追加
missing_keys = [k for k in required_keys if k not in statistical_df]
if missing_keys:
    raise ValueError(f'statistical_dfには{missing_keys}キーが必要です。')
```

**3. IAF取得方法の変更（L142-145削除、新規追加）**

```python
# Before（L142-145）
# IAF時系列（渡されている場合、ウォームアップ期間を除外）
if iaf_series is not None:
    iaf_series = iaf_series.sort_index()
    iaf_series = iaf_series[iaf_series.index >= session_start]

# After（L142あたりに追加）
# 🆕 IAFをStatistical DFから取得
iaf_series = statistical_df['iaf'].sort_index()
iaf_series = iaf_series[iaf_series.index >= session_start]
```

**4. IAFセグメント計算の更新（L187-199）**

```python
# IAF平均（statistical_dfから自動取得済み）
iaf_mean = np.nan
iaf_cv = np.nan

# セグメント範囲内のIAF値を取得
iaf_slice = iaf_series.loc[(iaf_series.index >= start) & (iaf_series.index < end)]
iaf_mean = iaf_slice.mean()

# IAF変動係数
if len(iaf_slice) > 1:
    iaf_std = iaf_slice.std()
    iaf_val = iaf_slice.mean()
    if pd.notna(iaf_val) and iaf_val != 0:
        iaf_cv = iaf_std / iaf_val
```

---

### Task 2.3: `generate_report.py`を簡略化

#### ファイル: `scripts/generate_report.py`

#### 変更内容

**1. PAF時間推移計算の削除（L678-684を削除）**

```python
# Before（L678-684）❌ 削除
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
    iaf_series=iaf_series,  # ❌ 削除
    warmup_minutes=1.0,
)

# After（L678あたり）✅ 簡潔に
# IAFはStatistical DFに含まれているため、準備不要
segment_result = calculate_segment_analysis(
    df_quality,
    fmtheta_result.time_series,
    statistical_df,
    segment_minutes=3,
    warmup_minutes=1.0,
)
```

**2. IAF統計取得の簡略化（L242-246を更新）**

```python
# Before（L242-246）
# Fmθ平均を追加
if 'frontal_theta_stats' in results:
    # ...

# IAF平均を追加
if 'iaf' in results:
    iaf_data = results['iaf']
    iaf_value = iaf_data['value']
    iaf_std = iaf_data['std']
    report += f"- **IAF平均 (Hz)**: {iaf_value:.2f} ± {iaf_std:.2f}\n"

# After（L242あたり）🆕 Statistical DFから取得
# Fmθ平均を追加
if 'frontal_theta_stats' in results:
    # ...

# IAF平均を追加（Statistical DFから）
if statistical_df is not None and 'iaf' in statistical_df:
    iaf_series = statistical_df['iaf']
    iaf_value = iaf_series.mean()
    iaf_std = iaf_series.std()
    report += f"- **IAF平均 (Hz)**: {iaf_value:.2f} ± {iaf_std:.2f}\n"
```

**3. 総合スコア計算のIAF変動係数取得（L813-828を更新）**

```python
# Before（L813-828）
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
        iaf_cv_val = paf_stats['変動係数 (%)'] / 100.0

# After（L813あたり）🆕 Statistical DFから直接取得
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
    if 'IAF平均 (Hz)' in segment_df.columns:
        iaf_values = segment_df['IAF平均 (Hz)'].dropna()
        if len(iaf_values) > 1:
            iaf_mean = iaf_values.mean()
            iaf_std = iaf_values.std()
            if iaf_mean > 0:
                iaf_cv_val = iaf_std / iaf_mean
```

---

### Task 2.4: テストの追加

#### 新規ファイル: `tests/test_statistical_dataframe_iaf.py`

```python
"""
Statistical DataFrame IAF統合のテスト
"""
import pytest
import numpy as np
import pandas as pd
from lib.statistical_dataframe import create_statistical_dataframe
from lib import prepare_mne_raw, load_mind_monitor_csv


def test_statistical_dataframe_includes_iaf(sample_csv_path):
    """Statistical DFにIAFが含まれることを確認"""
    # テストデータ読み込み
    df = load_mind_monitor_csv(sample_csv_path)
    mne_result = prepare_mne_raw(df)

    # Statistical DF生成
    statistical_df = create_statistical_dataframe(
        mne_result['raw'],
        segment_minutes=3,
        warmup_minutes=0.0
    )

    # IAFが含まれることを確認
    assert 'iaf' in statistical_df
    assert isinstance(statistical_df['iaf'], pd.Series)
    assert len(statistical_df['iaf']) > 0


def test_iaf_values_in_alpha_range(sample_csv_path):
    """IAFの値がアルファ帯域（8-13Hz）に収まることを確認"""
    df = load_mind_monitor_csv(sample_csv_path)
    mne_result = prepare_mne_raw(df)

    statistical_df = create_statistical_dataframe(
        mne_result['raw'],
        segment_minutes=3,
        warmup_minutes=0.0
    )

    iaf_values = statistical_df['iaf'].dropna()
    assert (iaf_values >= 8.0).all(), "IAFが8Hz未満の値を含んでいます"
    assert (iaf_values <= 13.0).all(), "IAFが13Hzを超える値を含んでいます"


def test_iaf_statistics_included(sample_csv_path):
    """統計量にIAF指標が含まれることを確認"""
    df = load_mind_monitor_csv(sample_csv_path)
    mne_result = prepare_mne_raw(df)

    statistical_df = create_statistical_dataframe(
        mne_result['raw'],
        segment_minutes=3,
        warmup_minutes=0.0
    )

    stats_df = statistical_df['statistics']
    iaf_stats = stats_df[stats_df['Category'] == 'IAF']

    # 必須統計量の確認
    required_metrics = ['iaf_Mean', 'iaf_Median', 'iaf_Std', 'iaf_CV']
    actual_metrics = iaf_stats['Metric'].tolist()

    for metric in required_metrics:
        assert metric in actual_metrics, f"{metric}が統計量に含まれていません"


def test_iaf_cv_calculation(sample_csv_path):
    """IAF変動係数が正しく計算されることを確認"""
    df = load_mind_monitor_csv(sample_csv_path)
    mne_result = prepare_mne_raw(df)

    statistical_df = create_statistical_dataframe(
        mne_result['raw'],
        segment_minutes=3,
        warmup_minutes=0.0
    )

    # 手動でCV計算
    iaf_values = statistical_df['iaf'].dropna()
    expected_cv = iaf_values.std() / iaf_values.mean()

    # Statistical DFからCV取得
    stats_df = statistical_df['statistics']
    iaf_cv_row = stats_df[stats_df['Metric'] == 'iaf_CV']
    actual_cv = iaf_cv_row['Value'].iloc[0]

    assert np.isclose(actual_cv, expected_cv, rtol=1e-5), \
        f"IAF変動係数の計算が不正確です（期待値: {expected_cv}, 実際: {actual_cv}）"


def test_iaf_consistency_across_segments(sample_csv_path):
    """異なるセグメント長でもIAFが一貫していることを確認"""
    df = load_mind_monitor_csv(sample_csv_path)
    mne_result = prepare_mne_raw(df)

    # 3分セグメント
    stat_df_3min = create_statistical_dataframe(
        mne_result['raw'],
        segment_minutes=3,
        warmup_minutes=0.0
    )

    # 5分セグメント
    stat_df_5min = create_statistical_dataframe(
        mne_result['raw'],
        segment_minutes=5,
        warmup_minutes=0.0
    )

    # 両方のIAF平均が近い値であることを確認（セグメント長が異なっても）
    iaf_3min_mean = stat_df_3min['iaf'].mean()
    iaf_5min_mean = stat_df_5min['iaf'].mean()

    # 20%以内の誤差を許容
    assert np.isclose(iaf_3min_mean, iaf_5min_mean, rtol=0.2), \
        f"セグメント長によってIAF平均が大きく変化しています（3分: {iaf_3min_mean}, 5分: {iaf_5min_mean}）"


@pytest.fixture
def sample_csv_path():
    """テスト用CSVファイルパスを返すフィクスチャ（実装時に調整）"""
    # 実際のテストデータパスに置き換え
    return "/path/to/test/data.csv"
```

---

## 🔄 後方互換性

### Deprecation Warning（一時的措置）

`segment_analysis.py`の`iaf_series`パラメータは、移行期間中（1-2ヶ月）は残しておき、deprecation warningを表示します。

```python
def calculate_segment_analysis(
    df_clean: pd.DataFrame,
    fmtheta_series: pd.Series,
    statistical_df: Dict[str, pd.DataFrame],
    segment_minutes: int = 5,
    iaf_series: Optional[pd.Series] = None,  # deprecated
    warmup_minutes: float = 0.0,
) -> SegmentAnalysisResult:
    """
    ...

    Parameters
    ----------
    iaf_series : pd.Series, optional
        ⚠️ **非推奨**: このパラメータは将来削除されます。
        IAFはstatistical_df['iaf']から自動取得されます。
    """
    # Deprecation warning
    if iaf_series is not None:
        import warnings
        warnings.warn(
            'iaf_series引数は非推奨です。'
            'IAFはstatistical_df["iaf"]から自動取得されます。'
            'このパラメータは将来のバージョンで削除されます。',
            DeprecationWarning,
            stacklevel=2
        )

    # 新しい方法でIAF取得（優先）
    if 'iaf' in statistical_df:
        iaf_series = statistical_df['iaf']
```

**移行期間後（2026年1月以降）に完全削除**

---

## 📊 期待される効果

### 定量的効果

| 指標 | 現状 | Phase 2完了後 |
|------|------|--------------|
| IAF計算箇所 | 3箇所（`paf.py`, `generate_report.py`, `segment_analysis.py`） | 1箇所（`statistical_dataframe.py`） |
| `generate_report.py`の行数 | 約900行 | 約870行（-30行） |
| `segment_analysis.py`のパラメータ数 | 6個 | 5個（-1個） |
| Statistical DFの戻り値キー数 | 4個 | 5個（+1個: `iaf`） |

### 定性的効果

1. **一貫性向上**: 全解析で同一のIAF計算ロジックを使用
2. **保守性向上**: IAF計算の修正が1箇所で完結
3. **使いやすさ向上**: セグメント分析呼び出しが簡潔に
4. **統計処理の統一**: Z-score外れ値除去が自動適用
5. **テスト容易性**: IAF計算のユニットテストが独立して実施可能

---

## ⚠️ Breaking Changes

### 影響を受けるAPI

**`calculate_segment_analysis()`**:
```python
# ❌ 旧（Phase 2完了後は非推奨、2026年1月以降削除）
calculate_segment_analysis(
    df_clean,
    fmtheta_series,
    statistical_df,
    segment_minutes=3,
    iaf_series=my_iaf_series,  # 非推奨
    warmup_minutes=1.0
)

# ✅ 新（推奨）
calculate_segment_analysis(
    df_clean,
    fmtheta_series,
    statistical_df,
    segment_minutes=3,
    warmup_minutes=1.0
)  # IAFはstatistical_dfから自動取得
```

### 移行チェックリスト

- [ ] `statistical_df`に`'iaf'`キーが含まれることを確認
- [ ] `calculate_segment_analysis()`の呼び出しから`iaf_series`引数を削除
- [ ] 既存のテストケースを更新

---

## 🔍 動作確認手順

### 1. ユニットテストの実行

```bash
pytest tests/test_statistical_dataframe_iaf.py -v
```

### 2. 統合テスト（generate_reportの実行）

```bash
python scripts/generate_report.py \
    --data data/sample.csv \
    --output output/test_phase2
```

### 3. 確認項目

- [ ] レポートに「IAF平均 (Hz)」が表示される
- [ ] セグメント分析テーブルに「IAF平均 (Hz)」列が含まれる
- [ ] 総合スコアにIAF変動係数が反映される（スコア内訳で確認）
- [ ] エラーやwarningが発生しない

---

## 📈 次のPhaseへの準備

Phase 2完了後、以下が可能になります：

### Phase 3（スケーリング単位の統一）への橋渡し
- IAF変動係数の計算が統一され、総合スコアの精度が向上
- Statistical DFにすべての主要指標が集約され、スケーリング変換の対象が明確化

### データ層の完全統一
- バンドパワー（Bels）
- バンド比率（Bels差分 + 実数値）
- Spectral Entropy（正規化済み）
- **IAF（Hz）** ← 🆕 Phase 2で追加

これにより、Phase 3でのスケーリング単位統一がスムーズに進められます。

---

## 参考資料

- `/home/tsu-nera/repo/satoru/lib/statistical_dataframe.py` - 統一的バンドパワー計算（修正対象）
- `/home/tsu-nera/repo/satoru/lib/segment_analysis.py` - セグメント分析（簡略化対象）
- `/home/tsu-nera/repo/satoru/scripts/generate_report.py` - レポート生成（簡略化対象）
- `/home/tsu-nera/repo/satoru/lib/sensors/eeg/paf.py` - PAF計算（参考、統合後も残す）

---

**最終更新**: 2025-11-16
