# 現在のMNE-Python使用状況

**調査日**: 2025-11-11

---

## 📊 概要

**結論**: 主要な信号処理は**既にMNE-Pythonを使用している** ✅

---

## ✅ MNE-Pythonを使用している機能

### 1. データ構造（RawArray）

**ファイル**: `lib/sensors/eeg/preprocessing.py`

```python
# L133: チャネル情報の作成
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# L138: RawArrayオブジェクトの作成
raw = mne.io.RawArray(data, info, copy='auto', verbose=False)
```

**評価**: ✅ 標準的な実装

---

### 2. フィルタリング

**ファイル**: `lib/sensors/eeg/preprocessing.py`

```python
# L142: ハイパスフィルタ（DCドリフト除去）
raw = raw.filter(l_freq=1.0, h_freq=None, fir_design='firwin', verbose=False)
```

**評価**: ✅ MNE標準関数を使用

**改善提案**:
- ノッチフィルタ追加（50Hz/60Hz電源ノイズ除去）
- バンドパスフィルタの明示化（1-50Hz）

---

### 3. パワースペクトル密度（PSD）

**ファイル**: `lib/sensors/eeg/frequency.py`

```python
# L45-51: Welch法によるPSD計算
spectrum = raw.compute_psd(
    method='welch',
    n_fft=n_fft,
    n_overlap=n_fft // 2,
    fmax=fmax,
    verbose=False,
)

freqs = spectrum.freqs
psds = spectrum.get_data()
```

**評価**: ✅ 完全にMNE準拠の実装

**単位変換**: V²/Hz → μV²/Hz（L58）

---

### 4. スペクトログラム（TFR）

**ファイル**: `lib/sensors/eeg/frequency.py`

```python
# L112-119: Morlet Wavelet変換
power = mne.time_frequency.tfr_array_morlet(
    data_3d,
    sfreq=sfreq,
    freqs=freqs,
    n_cycles=n_cycles,
    output='power',
    verbose=False,
)
```

**評価**: ✅ MNE標準の時間周波数解析

**単位変換**: V² → μV²（L122）

---

## 🟡 独自実装（適切な理由あり）

### 1. データ読み込み

**ファイル**: `lib/data/loader.py`

**理由**: Mind Monitor固有のCSVフォーマット

**評価**: ✅ 独自実装が必要

**処理内容**:
- CSVパース
- TimeStamp処理
- 複数のデータタイプ（RAW, Absolute, Accelerometer, etc.）

---

### 2. 品質フィルタリング

**ファイル**: `lib/sensors/eeg/preprocessing.py:48`

```python
def filter_eeg_quality(df, require_all_good=False):
    """
    HSI品質に基づくフィルタリング
    """
    hsi_cols = [c for c in df.columns if c.startswith('HSI_')]
    # HSI値による品質判定
    ...
```

**理由**: Mind Monitor固有のHSI（Horseshoe Signal Index）品質指標

**評価**: ✅ 独自実装が必要

---

### 3. サンプリングレート推定

**ファイル**: `lib/sensors/eeg/preprocessing.py:16`

```python
def _estimate_sampling_rate(frame):
    """
    不均一なサンプリングレートからの推定
    """
    duration_seconds = (frame.index.max() - frame.index.min()).total_seconds()
    sfreq = len(frame) / duration_seconds
    return sfreq
```

**理由**: Mind MonitorのCSVは複数のサンプリングレートが混在

**評価**: ✅ 独自実装が必要

---

### 4. バンド統計

**ファイル**: `lib/sensors/eeg/statistics.py`

```python
def calculate_band_statistics(df, bands=None):
    """
    周波数バンドの基本統計
    """
    # Mind MonitorのAbsoluteパワー列から統計計算
    ...
```

**理由**: Mind Monitor出力の直接処理

**評価**: ✅ 独自実装が適切（シンプルな統計処理）

---

### 5. バンド比率

**ファイル**: `lib/sensors/eeg/ratios.py`

```python
def calculate_band_ratios(df, resample_interval='10S', bands=None):
    """
    α/β（リラックス度）、β/θ（集中度）、θ/α（瞑想深度）
    """
    # 独自の比率計算とリサンプリング
    ...
```

**理由**: 独自の指標定義

**評価**: ✅ 独自実装が必要

**改善可能**: MNEのPSD結果を入力として使用可能

---

### 6. 高度な指標

#### 6.1 Frontal Midline Theta (Fmθ)

**ファイル**: `lib/sensors/eeg/frontal_theta.py`

```python
def calculate_frontal_theta(psd_dict, channels=['RAW_AF7', 'RAW_AF8'], ...):
    """
    前頭部のθ帯域パワー（瞑想・集中の指標）
    """
    # PSD結果から前頭部θパワーを抽出
    ...
```

**入力**: MNEのPSD結果（`psd_dict`）

**評価**: ✅ MNEの上に構築された独自指標

---

#### 6.2 Frontal Alpha Asymmetry (FAA)

**ファイル**: `lib/sensors/eeg/frontal_asymmetry.py`

```python
def calculate_frontal_asymmetry(psd_dict, alpha_range=(8.0, 13.0), ...):
    """
    左右前頭部のα波非対称性（感情・動機の指標）
    """
    # ln(右) - ln(左)
    ...
```

**入力**: MNEのPSD結果（`psd_dict`）

**評価**: ✅ MNEの上に構築された独自指標

---

#### 6.3 Spectral Entropy

**ファイル**: `lib/sensors/eeg/spectral_entropy.py`

```python
def calculate_spectral_entropy(psd_dict, ...):
    """
    周波数成分の多様性（集中度の指標）
    """
    # Shannon Entropy計算
    ...
```

**入力**: MNEのPSD結果（`psd_dict`）

**評価**: ✅ MNEの上に構築された独自指標

---

#### 6.4 Peak Alpha Frequency (PAF)

**ファイル**: `lib/sensors/eeg/paf.py`

```python
def calculate_paf(psd_dict, alpha_range=(8.0, 13.0)):
    """
    α帯域のピーク周波数（認知機能の指標）
    """
    # α帯域（8-13Hz）内の最大パワー周波数を検出
    ...
```

**入力**: MNEのPSD結果（`psd_dict`）

**評価**: ✅ MNEの上に構築された独自指標

---

## 📈 使用率分析

### 信号処理機能

| カテゴリ | 総数 | MNE使用 | 使用率 |
|---------|------|---------|--------|
| データ構造 | 1 | 1 | 100% |
| フィルタリング | 1 | 1 | 100% |
| PSD | 1 | 1 | 100% |
| TFR | 1 | 1 | 100% |
| **合計** | **4** | **4** | **100%** |

✅ **主要な信号処理は100% MNE-Python使用**

---

### 全体機能

| カテゴリ | 機能数 | MNE使用 | 独自実装 |
|---------|--------|---------|----------|
| データ構造 | 1 | 1 | 0 |
| データ読み込み | 1 | 0 | 1 |
| 前処理 | 2 | 1 | 1 |
| 信号処理 | 2 | 2 | 0 |
| 統計・比率 | 2 | 0 | 2 |
| 高度な指標 | 4 | 0 | 4 |
| 可視化 | 多数 | 0 | 多数 |

**MNE使用率（信号処理）**: 100%
**MNE使用率（全体）**: 約33%（ただし、独自実装の多くはMNE結果を入力として使用）

---

## 🎯 改善の余地がある領域

### 1. フィルタリング ⭐高優先度

**現状**: ハイパスフィルタのみ

**追加すべき**:
```python
# ノッチフィルタ（電源ノイズ除去）
raw.notch_filter(freqs=[50, 60])

# バンドパスフィルタ（明示的に上限を設定）
raw.filter(l_freq=1.0, h_freq=50.0)
```

---

### 2. Epochベース解析 🟡中優先度

**現状**: 連続データのみ

**追加すべき**:
```python
from mne import make_fixed_length_events, Epochs

# 時間セグメント用のイベント作成
events = make_fixed_length_events(raw, duration=300)

# Epochsオブジェクト
epochs = Epochs(raw, events, ...)

# セグメントごとのPSD
psds = epochs.compute_psd()
```

---

### 3. アーティファクト除去 🟡中優先度

**現状**: なし

**追加すべき**:
```python
from mne.preprocessing import ICA

# ICAによるノイズ除去
ica = ICA(n_components=4)
ica.fit(raw)
raw_clean = ica.apply(raw)
```

**課題**: 4チャネルでのICA有効性を検証

---

### 4. 可視化 🔵低優先度

**現状**: カスタムmatplotlib

**追加可能**:
```python
# MNE標準プロット
spectrum.plot()
raw.plot_sensors()
raw.plot()
```

---

## 📊 依存関係マップ

```
Mind Monitor CSV
    ↓
[load_mind_monitor_csv] (独自)
    ↓
DataFrame
    ↓
[filter_eeg_quality] (独自: HSI品質フィルタ)
    ↓
[prepare_mne_raw] (MNE: RawArray作成)
    ↓
mne.io.RawArray
    ↓
├─ [raw.filter] (MNE: フィルタリング)
├─ [raw.compute_psd] (MNE: PSD計算)
│   ↓
│   ├─ [calculate_paf] (独自: PAF検出)
│   ├─ [calculate_frontal_theta] (独自: Fmθ)
│   ├─ [calculate_frontal_asymmetry] (独自: FAA)
│   └─ [calculate_spectral_entropy] (独自: SE)
│
└─ [tfr_array_morlet] (MNE: スペクトログラム)
    ↓
    └─ [calculate_paf_time_evolution] (独自: PAF時間推移)
```

**評価**: ✅ 良好なアーキテクチャ（MNEを基盤に独自指標を構築）

---

## ✅ 結論

### 現状評価

1. **信号処理**: 100% MNE-Python使用 ✅
2. **アーキテクチャ**: MNEを基盤とした独自指標 ✅
3. **独自実装**: 必要な部分のみ（Mind Monitor固有処理） ✅

### 改善の方向性

**優先度高**: フィルタリングの強化（ノッチ、バンドパス）
**優先度中**: Epochsベース解析、ICA実験
**優先度低**: MNE標準可視化の活用

### 総評

**現在の実装は既に良い状態** 🎉

さらなる改善はあるものの、基本的なアプローチは正しい。段階的に機能を追加していく方針が適切。

---

## 📝 次のアクション

1. [PLAN.md](./PLAN.md)を読む
2. Phase 1（フィルタリング強化）から開始
3. 実装しながら本ドキュメントを更新

---

**最終更新**: 2025-11-11
