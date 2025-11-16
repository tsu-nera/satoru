# Mind Monitor CSV Specification

このドキュメントは、Mind Monitor (Muse Headband用サードパーティアプリ) が出力するCSVファイルのフォーマット仕様をまとめたものです。

**公式ドキュメント**: https://mind-monitor.com/FAQ.php#csvspec

## ファイル概要

- **形式**: Excel互換CSV
- **エンコーディング**: UTF-8
- **構造**: 各行が1タイムスタンプのデータ、各列がセンサー/メトリクス
- **用途**: グラフ化や統計解析に適した形式

---

## 列（カラム）仕様

### 1. タイムスタンプ

| カラム名 | 説明 | フォーマット | 例 |
|---------|------|------------|-----|
| `TimeStamp` | 測定日時 | `YYYY-MM-DD HH:MM:SS.mmm` | `2025-10-26 08:32:21.081` |

---

### 2. 脳波データ（EEG）- 周波数帯域別

各周波数帯域について、4つのセンサー位置（TP9, AF7, AF8, TP10）のデータが記録されます。

#### センサー位置
- **TP9**: 左耳後ろ（Left Ear）
- **AF7**: 左前頭部（Left Forehead）
- **AF8**: 右前頭部（Right Forehead）
- **TP10**: 右耳後ろ（Right Ear）

#### 周波数帯域

| 周波数帯域 | カラム名パターン | 周波数範囲 | 単位 | 関連する状態 |
|-----------|----------------|-----------|------|------------|
| **Delta** (δ) | `Delta_{TP9,AF7,AF8,TP10}` | 0.5-4 Hz | Bels | 深い睡眠、無意識 |
| **Theta** (θ) | `Theta_{TP9,AF7,AF8,TP10}` | 4-8 Hz | Bels | 瞑想、創造性、記憶 |
| **Alpha** (α) | `Alpha_{TP9,AF7,AF8,TP10}` | 8-13 Hz | Bels | リラックス、目を閉じた安静状態 |
| **Beta** (β) | `Beta_{TP9,AF7,AF8,TP10}` | 13-30 Hz | Bels | 集中、活発な思考、ストレス |
| **Gamma** (γ) | `Gamma_{TP9,AF7,AF8,TP10}` | 30-44 Hz | Bels | 高次認知機能、情報処理 |

**合計**: 各帯域4センサー × 5帯域 = **20カラム**

---

### 3. RAW脳波データ

| カラム名 | 説明 | 範囲 | 単位 | 備考 |
|---------|------|------|------|------|
| `RAW_TP9` | 左耳後ろの生波形 | 0.0 - 1682.815 | µV (マイクロボルト) | 未フィルタの生データ |
| `RAW_AF7` | 左前頭部の生波形 | 0.0 - 1682.815 | µV | |
| `RAW_AF8` | 右前頭部の生波形 | 0.0 - 1682.815 | µV | |
| `RAW_TP10` | 右耳後ろの生波形 | 0.0 - 1682.815 | µV | |

---

### 4. 補助センサー（AUX）

| カラム名 | 説明 | 範囲 | 単位 | 備考 |
|---------|------|------|------|------|
| `AUX_1` | 補助USB接続センサー1 | 0.0 - 1682.815 | µV | オプション（接続時のみ） |
| `AUX_2` | 補助USB接続センサー2 | 0.0 - 1682.815 | µV | |
| `AUX_3` | 補助USB接続センサー3 | 0.0 - 1682.815 | µV | |
| `AUX_4` | 補助USB接続センサー4 | 0.0 - 1682.815 | µV | |

---

### 5. 加速度センサー

| カラム名 | 説明 | 範囲 | 単位 | 用途 |
|---------|------|------|------|------|
| `Accelerometer_X` | X軸の重力加速度 | -2 ~ +2 | g | 頭部の傾き検出 |
| `Accelerometer_Y` | Y軸の重力加速度 | -2 ~ +2 | g | |
| `Accelerometer_Z` | Z軸の重力加速度 | -2 ~ +2 | g | 垂直方向の動き |

**応用例**:
- 頭部の姿勢推定
- 体動アーチファクトの検出
- 呼吸による微細な動きの検出

---

### 6. ジャイロスコープ

| カラム名 | 説明 | 範囲 | 単位 | 特性 |
|---------|------|------|------|------|
| `Gyro_X` | X軸の角速度 | -245 ~ +245 | degrees/sec | 時間経過で0に戻る |
| `Gyro_Y` | Y軸の角速度 | -245 ~ +245 | degrees/sec | |
| `Gyro_Z` | Z軸の角速度 | -245 ~ +245 | degrees/sec | |

**応用例**:
- 頭部の回転検出
- 瞑想中の微細な動きの分析
- モーションアーチファクトの除去

---

### 7. 光学センサー

#### PPG（Photoplethysmography: 光電式容積脈波）
デバイスによってカラム数が異なります。

| デバイス | カラム名 | 説明 | 単位 | 用途 |
|---------|---------|------|------|------|
| **Muse S/S Gen2** | `Optics1` - `Optics16` | fNIRS（機能的近赤外分光法）+ PPG | 0-32 µA | 心拍数、HRV、脳血流 |
| **Muse 2** | `PPG1` - `PPG3` | PPGセンサー | 任意単位 | 心拍数測定 |

**Muse S Athena (Gen3) の光学センサー詳細**:
- **サンプリングレート**: 64 Hz
- **解像度**: 20-bit
- **波長**: 730nm (Near-IR) / 850nm (IR) / 660nm (Red) / Ambient
- **fNIRS配置**: 5-optode bilateral frontal cortex (左右前頭皮質)
- **チャンネル構成**: 16チャンネル（左内/左外/右内/右外 × 各波長）

#### Opticsチャンネルマッピング (MS-03 Athena)

デバイスの構成によって使用されるチャンネル数が異なります:

**4チャンネル構成**:
| チャンネル | 波長 | 位置 |
|----------|------|------|
| Optics1 | 730nm | Left Inner (左内側) |
| Optics2 | 730nm | Right Inner (右内側) |
| Optics3 | 850nm | Left Inner (左内側) |
| Optics4 | 850nm | Right Inner (右内側) |

**8チャンネル構成**:
| チャンネル | 波長 | 位置 |
|----------|------|------|
| Optics1 | 730nm | Left Outer (左外側) |
| Optics2 | 730nm | Right Outer (右外側) |
| Optics3 | 850nm | Left Outer (左外側) |
| Optics4 | 850nm | Right Outer (右外側) |
| Optics5 | 730nm | Left Inner (左内側) |
| Optics6 | 730nm | Right Inner (右内側) |
| Optics7 | 850nm | Left Inner (左内側) |
| Optics8 | 850nm | Right Inner (右内側) |

**16チャンネル構成**:
| チャンネル | 波長 | 位置 |
|----------|------|------|
| Optics1 | 730nm | Left Outer (左外側) |
| Optics2 | 730nm | Right Outer (右外側) |
| Optics3 | 850nm | Left Outer (左外側) |
| Optics4 | 850nm | Right Outer (右外側) |
| Optics5 | 730nm | Left Inner (左内側) |
| Optics6 | 730nm | Right Inner (右内側) |
| Optics7 | 850nm | Left Inner (左内側) |
| Optics8 | 850nm | Right Inner (右内側) |
| Optics9 | 660nm (Red) | Left Outer (左外側) |
| Optics10 | 660nm (Red) | Right Outer (右外側) |
| Optics11 | Ambient | Left Outer (左外側) |
| Optics12 | Ambient | Right Outer (右外側) |
| Optics13 | 660nm (Red) | Left Inner (左内側) |
| Optics14 | 660nm (Red) | Right Inner (右内側) |
| Optics15 | Ambient | Left Inner (左内側) |
| Optics16 | Ambient | Right Inner (右内側) |

**fNIRS解析における波長の役割**:
- **730nm (Near-IR)**: HbO/HbRの測定に使用（短波長）
- **850nm (IR)**: HbO/HbRの測定に使用（長波長） - Modified Beer-Lambert Lawにより両波長からHbO/HbRを算出
- **660nm (Red)**: 追加の酸素化指標測定
- **Ambient**: 環境光ノイズの除去用

**応用例**:
- 心拍数（HR）の測定
- 心拍変動（HRV）解析
- 呼吸数の推定（RSA: Respiratory Sinus Arrhythmia）
- 脳血流量の推定（fNIRSのみ）
- 酸素化ヘモグロビン（HbO）・脱酸素化ヘモグロビン（HbR）の測定

---

### 8. 心拍数

| カラム名 | 説明 | 範囲 | 単位 | 備考 |
|---------|------|------|------|------|
| `Heart_Rate` | 推定心拍数 | 40-200（典型的） | BPM | PPGセンサーから算出済み |

**注意**:
- デバイスが自動算出した値
- 0の場合は測定失敗または未装着を示す

---

### 9. データ品質指標

| カラム名 | 説明 | 値 | 意味 |
|---------|------|-----|------|
| `HeadBandOn` | ヘッドバンド装着状態 | `1` | 装着中（True） |
|  |  | `0` | 未装着（False） |
| `HSI_TP9` | TP9センサーの信号品質 | `1.0` | Good（良好） |
| `HSI_AF7` | AF7センサーの信号品質 | `2.0` | Medium（普通） |
| `HSI_AF8` | AF8センサーの信号品質 | `4.0` | Bad（不良） |
| `HSI_TP10` | TP10センサーの信号品質 |  |  |

**補足:**
- HSI (Horseshoe Signal Indicator) は各センサーの接触品質を示す
- 実データでは浮動小数点数 (`1.0`, `2.0`, `4.0`) で記録される
- `HeadBandOn` は整数 (`0`, `1`) で記録される
- 品質フィルタリングでは通常、HSI全チャネルが `1.0` (Good) または `≤2.0` (Good/Medium) の行を使用

**HSI (Horseshoe Indicator)**:
- 電極接触の品質を示す
- データ解析時に品質フィルタリングに使用可能

---

### 10. バッテリー

| カラム名 | 説明 | 範囲 | 単位 | フォーマット |
|---------|------|------|------|------------|
| `Battery` | バッテリー残量 | 0-100 | % | `98.51` = 98.51% |

---

### 11. イベントマーカー

| カラム名 | 説明 | 例 | 用途 |
|---------|------|-----|------|
| `Elements` | イベント/マーカー | `/muse/event/connected MuseS-FA35` | デバイス接続 |
|  |  | `blink` | 瞬き検出 |
|  |  | `jaw_clench` | 顎の食いしばり検出 |

**応用例**:
- セッション開始/終了のマーキング
- アーチファクト（瞬き、筋肉活動）の除去
- 実験プロトコルのタイミング記録

---

## データ解析時の注意点

### 1. サンプリングレート

#### ハードウェア仕様（実際のサンプリング）
- **EEG（脳波）**: 256 Hz (14-bit resolution on Muse S Athena)
- **加速度/ジャイロ**: 52 Hz (16-bit resolution)
- **PPG/fNIRS**: 64 Hz (20-bit resolution on Muse S Athena) / 256 Hz（Muse 2）

#### ⚠️ タイムスタンプの精度問題

**重要**: CSV内のタイムスタンプは**Bluetoothバッファリングの影響で不正確**です。

**公式見解（Interaxon社 + Mind Monitor開発者 James）**:
> "Interaxon has assured that on the hardware the data is actually being processed equidistantly at 256Hz"
>
> "The timestamps are not being generated by the hardware but by the LibMuse software API and are subject to processing overhead and bluetooth stack buffering"
>
> "Interaxon said to **ignore the discrepancies and treat the data as if it was 256Hz exactly**"

**具体的な現象**:
- 同じタイムスタンプが複数行で重複（例: 4サンプルが同じミリ秒）
- タイムスタンプの精度が実質64Hz程度（256Hz ÷ 4）
- データは"束ねられて"到着（クラスタリング）

**原因**:
1. Museハードウェアは正確に256Hzでサンプリング
2. Bluetoothパケットバッファに蓄積
3. Mind Monitorがバッファから取り出すタイミングでタイムスタンプを付与
4. → 複数サンプルが同じタイムスタンプになる

#### 推奨される対処法

**A. オフライン分析（推奨）**:
タイムスタンプを無視し、256Hz等間隔と仮定
```python
# ハードウェアの真のサンプリングレートを使用
sfreq = 256.0  # Muse標準（MU-01は220Hz）
raw = mne.io.RawArray(data, info)
```

**B. 高精度タイミングが必要な場合**:
CSV形式ではなく`.muse`形式（Protocol Buffer）を使用
- 各データアイテムを個別記録
- タイムスタンプの重複なし
- ただし、Mind Monitorアプリでのエクスポート設定が必要

**参考リンク**:
- [Mind Monitor Forums - Raw data timestamps](https://mind-monitor.com/forums0/viewtopic.php?p=4644)
- [Interaxon公式見解についての議論](https://mind-monitor.com/forums/viewtopic.php?t=1447)

### 1.1 CSV形式の構造的制約

CSVでは全データが統合されているため、サンプリングレートが異なるデータが混在しています。

- RAW EEG: 256Hz → 各行で値が異なる
- ブレインウェーブ（Alpha/Beta等）: 10Hz → 同じ値が25.6行続く
- ジャイロ: 52Hz → 同じ値が約5行続く

これは**CSV形式の設計上の制約**であり、不具合ではありません。

### 2. 欠損値の扱い
- 空白セル: センサーが該当時刻にデータを出力しなかった
- `0`値: 特に`Heart_Rate=0`は測定失敗を示す

### 3. 単位変換
- **Bels → デシベル (dB)**: `dB = Bels × 10`
- **g → m/s²**: `m/s² = g × 9.80665`

### 4. 品質管理
推奨フィルタリング条件:
```python
# 良質なデータのみを抽出
df_quality = df[
    (df['HeadBandOn'] == 1) &  # 装着中
    (df['HSI_TP9'] == 1) &      # 全センサー良好
    (df['HSI_AF7'] == 1) &
    (df['HSI_AF8'] == 1) &
    (df['HSI_TP10'] == 1)
]
```

---

---

**最終更新**: 2025-10-26
**バージョン**: 1.0
