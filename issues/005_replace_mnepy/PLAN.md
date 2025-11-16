# MNE-Python活用強化計画

**Issue**: #005_replace_mnepy
**作成日**: 2025-11-11
**ステータス**: 計画中

---

## 📋 目的

独自実装からMNE-Pythonの標準機能への移行を進め、以下を実現する：

1. **科学的妥当性の向上** - 学術界で検証済みのアルゴリズムを使用
2. **標準化** - 他の研究との比較可能性を確保
3. **保守性の向上** - コミュニティによる継続的な改善を活用
4. **機能拡張** - MNEの高度な機能（ICA、Epochs、統計解析など）を活用

---

## 🔍 現状分析

### ✅ 既にMNE-Pythonを使用している部分

| 機能 | 実装場所 | MNE関数 | 状態 |
|------|---------|---------|------|
| RawArray作成 | `lib/sensors/eeg/preprocessing.py:138` | `mne.io.RawArray` | ✓ |
| チャネル情報 | `lib/sensors/eeg/preprocessing.py:133` | `mne.create_info` | ✓ |
| ハイパスフィルタ | `lib/sensors/eeg/preprocessing.py:142` | `raw.filter()` | ✓ |
| PSD計算 | `lib/sensors/eeg/frequency.py:45` | `raw.compute_psd()` | ✓ |
| スペクトログラム | `lib/sensors/eeg/frequency.py:112` | `mne.time_frequency.tfr_array_morlet()` | ✓ |

**評価**: 主要な信号処理は既にMNE-Pythonを使用 🎉

### 🟡 独自実装（これは適切）

| 機能 | 実装場所 | 理由 |
|------|---------|------|
| CSV読み込み | `lib/data/loader.py` | Mind Monitor固有フォーマット |
| HSI品質評価 | `lib/sensors/eeg/preprocessing.py:48` | Mind Monitor固有の品質指標 |
| サンプリングレート推定 | `lib/sensors/eeg/preprocessing.py:16` | 不均一データへの対応 |
| バンド比率計算 | `lib/sensors/eeg/ratios.py:12` | 独自指標（α/β、β/θ、θ/α） |
| Fmθ解析 | `lib/sensors/eeg/frontal_theta.py:56` | MNEにない高度な指標 |
| FAA解析 | `lib/sensors/eeg/frontal_asymmetry.py:34` | MNEにない高度な指標 |
| Spectral Entropy | `lib/sensors/eeg/spectral_entropy.py:60` | MNEにない高度な指標 |
| PAF解析 | `lib/sensors/eeg/paf.py:9` | MNEにない高度な指標 |

**評価**: これらの独自実装は妥当。MNEの結果を**入力として使用**している 👍

---

## 🎯 改善提案

### Phase 1: 前処理の強化（優先度: 高）

#### 1.1 ノッチフィルタの追加

**目的**: 電源ノイズ（50Hz/60Hz）を除去

**現状**: ハイパスフィルタのみ実装（1Hz以上）

**改善案**:
```python
# lib/sensors/eeg/preprocessing.py に追加
def apply_notch_filter(raw, freqs=[50, 60]):
    """
    電源ノイズ除去のためのノッチフィルタ適用

    Parameters
    ----------
    raw : mne.io.RawArray
    freqs : list
        除去する周波数リスト（デフォルト: [50, 60] Hz）
    """
    return raw.notch_filter(freqs=freqs, verbose=False)
```

**影響**:
- ファイル: `lib/sensors/eeg/preprocessing.py`
- テスト: ノッチフィルタ前後のPSDを比較

**参考**:
- [muse-lsl examples/utils.py](https://github.com/alexandrebarachant/muse-lsl/blob/master/examples/utils.py)
- MNE-Python公式ドキュメント

---

#### 1.2 バンドパスフィルタの明示化

**目的**: 解析対象周波数帯域を明確に定義

**現状**: ハイパスフィルタのみ（1Hz-）

**改善案**:
```python
# デフォルトで1-50Hzに制限（Museの有効帯域）
raw = raw.filter(l_freq=1.0, h_freq=50.0, fir_design='firwin', verbose=False)
```

**影響**:
- ファイル: `lib/sensors/eeg/preprocessing.py:142`
- テスト: フィルタ後の周波数特性を確認

---

### Phase 2: Epochベース解析の導入（優先度: 中）

#### 2.1 時間セグメント分析の改善

**目的**: 現在のパンダスベースのセグメント分析をMNE Epochsに移行

**現状**: DataFrameを時間で分割（`scripts/generate_report.py`）

**改善案**:
```python
from mne import Epochs, make_fixed_length_events

# 固定長イベントを作成（例: 5分ごと）
events = make_fixed_length_events(raw, duration=300)  # 300秒 = 5分

# Epochsオブジェクト作成
epochs = Epochs(raw, events, tmin=0, tmax=300, baseline=None,
                preload=True, verbose=False)

# セグメントごとにPSDを計算
psds, freqs = epochs.compute_psd(method='welch').get_data(return_freqs=True)
```

**メリット**:
- MNEの統計関数が使える
- メモリ効率が良い
- 標準的な解析フロー

**影響**:
- 新ファイル: `lib/sensors/eeg/epochs.py`
- 更新: `scripts/generate_report.py`

---

#### 2.2 イベント関連電位（ERP）への拡張

**目的**: 将来的な刺激実験への対応

**改善案**:
```python
# Mind Monitorのマーカーからイベントを抽出
# （将来機能 - Mind Monitorがマーカーをサポートした場合）
events = mne.find_events(raw, stim_channel='STI')
epochs = Epochs(raw, events, event_id={'target': 1}, ...)
```

**優先度**: 低（マーカー機能が必要）

---

### Phase 3: 可視化の強化（優先度: 低）

#### 4.1 MNE標準プロット関数の活用

**目的**: 標準的な脳波可視化

**改善案**:
```python
# PSDプロット
spectrum.plot(picks='eeg', average=True)

# トポグラフィマップ（電極配置）
raw.plot_sensors(show_names=True)

# 生データプロット（インタラクティブ）
raw.plot(duration=60, n_channels=4)
```

**メリット**:
- 標準的な表示形式
- インタラクティブな探索

**デメリット**:
- レポート生成との統合が必要
- 現在のカスタムプロットと並存

---

## 📊 コミュニティ事例

**⭐ 詳細分析**: [MUSE_LSL_ANALYSIS.md](./MUSE_LSL_ANALYSIS.md) - muse-lslプロジェクトの徹底分析

**結論**: muse-lslは素晴らしいプロジェクト（708スター）だが、**リアルタイム処理特化**のため我々の目的（オフライン分析）には不適合。**MNE-Pythonを使い続けるべき**。ただし、ノッチフィルタ実装とニューロフィードバック指標は参考にできる。

### 参考リポジトリ

1. **muse-lsl** (Alexandre Barachant) ⭐708
   - URL: https://github.com/alexandrebarachant/muse-lsl
   - 特徴: リアルタイムストリーミング・ニューロフィードバック特化
   - 参考箇所: `examples/utils.py`, `examples/neurofeedback.py`
   - **注意**: 独自FFT実装（MNE不使用）、リアルタイム処理向き

2. **muse-lsl-python** (Uri Shaked)
   - URL: https://github.com/urish/muse-lsl-python
   - 特徴: Jupyter Notebook形式の実験例
   - 参考箇所: `notebooks/SSVEP with Muse.ipynb`, `notebooks/P300 with Muse.ipynb`

3. **Muse-MotorImageryClassification** (Vinayak R)
   - URL: https://github.com/vinayakr99/Muse-MotorImageryClassification
   - 特徴: Mind Monitor CSV → MNE Epochs → ML
   - 参考箇所: `MotorImagery_Training.ipynb`

### コミュニティの共通認識

- ✓ MNEを使うのが**標準**（"industry standard"）
- ✓ サンプリングレート: **256Hz（Constant録音モード）**
- ✓ Mind Monitorは**フィルタリングしない**（生データ）
- ✓ 前処理はMNEに任せる

---

## 🗓️ 実装スケジュール

### Phase 1: 前処理強化（1-2日） ✅ **完了**

- [x] ノッチフィルタの実装（2025-11-11）
- [x] バンドパスフィルタの明示化（2025-11-11）
- [x] サンプリングレート依存の自動調整ロジック追加（2025-11-11）
- [x] Spectral Entropy／バンド比率グラフの回帰修正（2025-11-11）
- [x] ドキュメント作成（2025-11-11）
- 📄 [詳細レポート](./PHASE1_COMPLETE.md)

**成果サマリー**
- 実測サンプリングレートからNyquistを計算し、ハイカット／ノッチ周波数を安全範囲に自動制限することで、Mind Monitor由来の低サンプリングセッションでもエラーなく解析できるようにした。
- 優先チャネル選択 (`tfr_primary`) を導入し、PAF時間推移とSpectral Entropyの両方で未定義変数が発生しないように改善。
- カラム名英語化の影響で表示されなくなっていたバンド比率折れ線グラフを、内部キー（`Alpha/Beta` など）とラベル表示を分離することで復旧。

**要フォローアップ**
- Spectral Entropyやバンド比率の詳細統計を今後のスコアリングでどう扱うか方針化（Phase2以降で検討）。
- 追加のCSVでもフィルタ自動調整が想定どおり働くか継続モニタリング。

### Phase 2: Epochs導入（3-5日） ✅ **完了**

- [x] MNE Epochsによるバンドパワー計算関数を実装（2025-11-14）
- [x] `calculate_segment_analysis`関数にMNE Epochs統合（2025-11-14）
- [x] 既存レポートとの互換性確保（2025-11-14）
- [x] テストデータで動作確認（2025-11-14）
- 📄 [完了レポート](./PHASE2_COMPLETE.md)

**成果サマリー**
- MNE EpochsでセグメントごとのPSD計算を実装し、バンドパワー計算を高精度化
- `_calculate_band_power_from_epochs()` 関数で固定長Epochsを作成し、Welch法によるPSD→バンドパワー計算を自動化
- Nyquist周波数を考慮した安全なfmax自動調整により、低サンプリングレートデータでもエラーなく動作
- 既存のDataFrameベース計算は後方互換性のため保持（`use_mne_epochs=False`で切り替え可能）
- セグメント分析のループロジックを簡素化（MNE Epochsが事前にセグメント化するため）

**要フォローアップ**
- 現時点ではコード行数は増加（+169行）。これは新関数追加とレガシーパス保持のため
- 将来的にレガシーパスを削除すれば、コード削減効果が顕著になる（推定: -50行程度）
- fNIRS/心拍数などのマルチモーダル対応は別Issueで検討

### Phase 3: 可視化（2-3日）

#### 3.1 MNE標準プロット関数の活用（優先度: 高）

- [x] PSDプロット改善 - `spectrum.plot()`の活用検討（2025-11-12）
- [x] スペクトログラム改善 - MNE時間周波数プロットの活用検討（2025-11-12）
- [ ] バンドパワー時系列改善 - **実施しない**（Phase 3.1ではスコープ外と判断）

> Phase 3.1はPSD／スペクトログラム改善をもって完了。バンドパワーの強化は別Issueで必要性を再評価する。

#### 3.2 新規可視化の追加（優先度: 中）

- [ ] トポグラフィマップ追加 - **保留**（`raw.plot_sensors()`活用案は一旦キャンセル）
- [x] 生データプロット追加 - `plot_raw_preview()`で静的画像出力（2025-11-12）

> トポグラフィ表示はユーザー側で別途検証するため、本Issueではスコープ外に戻した。生データプレビューのみレポート冒頭（接続品質直後）に配置し、品質指標→波形確認の流れを維持する。

#### 3.3 統合とテスト

- [ ] レポート生成スクリプトへの統合
- [ ] ドキュメント更新

**スコープ外（別Issueで検討）**:
- 既存の高度な指標プロット（Fmθ、FAA、PAF、SE、バンド比率）の評価
- fNIRSプロットの改善検討
- 時間セグメント比較プロットの改善検討

---

## ⚠️ 注意事項

### 保持すべき独自実装

以下は**変更しない**：

1. Mind Monitor CSV読み込み（独自フォーマット）
2. HSI品質評価（Mind Monitor固有）
3. サンプリングレート推定（不均一データ対応）
4. 高度な指標計算（Fmθ、FAA、SE、PAF）
5. カスタムレポート生成

これらはMNEの**上に構築**する付加価値機能。

### 後方互換性

- 既存のレポートフォーマットを維持
- 段階的な移行（オプション機能として追加）
- 既存コードの破壊的変更を避ける

---

## 📚 参考資料

### MNE-Python公式

- [Creating MNE objects from arrays](https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html)
- [Filtering and resampling](https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html)
- [ICA for artifact correction](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html)
- [Epoching data](https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html)

### コミュニティディスカッション

- [Converting Mind Monitor CSV to EDF via MNE](https://mne.discourse.group/t/converting-mind-monitor-csv-data-from-muse-s-headband-to-edf-via-mne-python/9346)
- [Using Muse with MNE realtime](https://mne.discourse.group/t/using-muse-with-mne-realtime/4046)

### 論文

- Gramfort et al. (2013). "MEG and EEG data analysis with MNE-Python"
- Barachant et al. (2012). "Multiclass Brain-Computer Interface Classification by Riemannian Geometry"

---

## 🎯 成功指標

1. **品質**: MNE-Python標準機能の使用率 > 80%
2. **互換性**: 既存レポートが破壊されない
3. **パフォーマンス**: 処理時間が2倍以内
4. **保守性**: コード行数が20%削減
5. **文書化**: 全関数にMNE関数との対応を明記

---

## 💬 議論ポイント

### Q1: Epochsベース vs 連続データベース？

**現状**: 連続データで全て処理

**提案**: 時間セグメント分析にEpochsを使用

**メリット**:
- 統計解析が容易
- メモリ効率
- 標準的な手法

**デメリット**:
- 実装コストが高い
- 既存コードの大幅な変更

**結論**: Phase 2で段階的に導入

### Q2: ICAは必要か？

**課題**: 4チャネルでICAは効果的？

**検討**: 実データで効果を検証してから判断

**結論**: 本Issueではスコープ外。別Issueで実験・評価を行い、必要なら再計画する。

### Q3: 可視化の優先度は？

**現状**: カスタムmatplotlib実装が機能している

**提案**: MNE標準プロットは低優先度

**結論**: Phase 3で検討、必須ではない

---

## 📝 更新履歴

- 2025-11-14: Phase 2を完了（MNE Epochsによるセグメント分析のバンドパワー計算を高精度化、後方互換性維持）
- 2025-11-12: トポグラフィ対応をキャンセル（Phase 3.2のスコープから除外し、生データプレビューのみ維持）
- 2025-11-12: Phase 3.2を開始（接続品質直後に生データプレビューを挿入、plot_raw_preview導入）
- 2025-11-12: Phase 3.1を完了（PSD・スペクトログラム改善のみ実施、バンドパワー強化は見送り）
- 2025-11-12: Phase 3の詳細計画を追加（可視化タスクを3つのサブフェーズに分解、スコープ外を明確化）
- 2025-11-11: Phase 1を完了（フィルタ自動調整、Spectral Entropy/バンド比率グラフ修正）
- 2025-11-11: 初版作成（調査結果をもとに計画策定）
