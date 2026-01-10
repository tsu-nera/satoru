# SelfLoops HRVローダー実装完了

## 実装内容

### 新しい`get_hrv_data()`の設計

```python
from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data

df = load_selfloops_csv('data/selfloops.csv')
hrv_data = get_hrv_data(df)  # デフォルトで推奨設定
```

### 3つの計算モード

| モード | 設定 | 用途 | 推奨 |
|--------|------|------|------|
| **デフォルト** | `clean_artifacts=True` | R-R間隔から計算、外れ値除外 | ✅ 推奨 |
| 生データ | `clean_artifacts=False` | R-R間隔から計算、外れ値含む | デバッグ用 |
| デバイスHR | `use_device_hr=True` | SelfLoopsのHR列使用 | 非推奨 |

## テスト結果（実データ）

### 統計比較

| 指標 | デフォルト | 生データ | SelfLoops HR |
|------|----------:|--------:|-------------:|
| 平均 | 74.72 bpm | 74.88 bpm | 75.51 bpm |
| 標準偏差 | **4.96 bpm** | 6.60 bpm | 3.66 bpm |
| 最小値 | 63.22 bpm | 63.22 bpm | 68.00 bpm |
| **最大値** | **88.37 bpm** | **214.29 bpm** ⚠️ | 87.00 bpm |

### 外れ値処理の効果

- **補正されたポイント**: 3/1133 (0.3%)
- **最大補正例**:
  - 元: 280 ms → 214.3 bpm（異常値）
  - 補正後: 816 ms → 73.5 bpm（正常値）

### 標準偏差の改善

```
生データ:     6.60 bpm（不安定、外れ値の影響）
↓ 外れ値除外（0.3%のみ）
デフォルト:   4.96 bpm（安定、24.8%改善）
```

## 将来の拡張性

### Elite HRV対応（例）

```python
# Elite HRVのフォーマット
# timestamp,ibi
# 1641234567890,813
# 1641234567890,793

def load_elitehrv_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['R-R (ms)'] = df['ibi']  # 列名を統一
    df['Time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000
    return df

# 同じget_hrv_data()で処理可能
hrv_data = get_hrv_data(df)  # 全く同じコード！
```

### Polar H10対応（例）

```python
def load_polar_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['R-R (ms)'] = df['RR-interval']  # 列名を統一
    # ...
    return df

hrv_data = get_hrv_data(df)  # 同じインターフェース
```

## 外れ値除外アルゴリズム

HRV Task Force (1996) 基準:

1. **絶対的閾値**:
   - 最小R-R間隔: 300 ms (200 bpm上限)
   - 最大R-R間隔: 2000 ms (30 bpm下限)

2. **変化率チェック**:
   - 前の値との変化率: 20%以内

3. **補間方法**:
   - 線形補間で外れ値を置き換え

## 推奨事項

### ✅ 使用すべき設定

```python
# デフォルト（推奨）
hrv_data = get_hrv_data(df)

# 明示的に指定する場合
hrv_data = get_hrv_data(df, use_device_hr=False, clean_artifacts=True)
```

**理由**:
- **汎用性**: Elite HRV、Polarなど他のHRVアプリにも対応可能
- **標準化**: HRV研究のゴールドスタンダード手法
- **精度**: 学術的に検証済みの外れ値処理
- **透明性**: 処理内容が明確（SelfLoopsのHR列は仕様不明）

### ❌ 避けるべき設定

```python
# SelfLoopsのHR列を使用（非推奨）
hrv_data = get_hrv_data(df, use_device_hr=True)
```

**理由**:
- SelfLoops固有の仕様（他のアプリに移行できない）
- 処理内容が不明確
- 将来の拡張性がない

## 参考文献

Task Force of the European Society of Cardiology and the North American
Society of Pacing and Electrophysiology (1996). Heart rate variability:
standards of measurement, physiological interpretation and clinical use.
Circulation, 93(5), 1043-1065.
