# 坐相分析プロジェクト 引き継ぎドキュメント

**作成日**: 2025年12月25日
**次のタスク**: 複数セッションでの仮説検証

---

## プロジェクト概要

坐禅中の坐相（IMUセンサー）と脳波（EEG）、脳血流（fNIRS）、心拍数の関係を定量的に解析し、藤田一照氏の「不動の中の動」を科学的に検証するプロジェクトです。

---

## 完了したこと（Phase 1: 単一セッション分析）

### 📁 フォルダ構成

```
issues/008_sazou/
├── README.md                          # プロジェクト全体の説明
├── HANDOFF.md                         # この引き継ぎドキュメント
└── 2025-12-24_single_session/         # Phase 1: 単一セッション分析
    ├── REPORT.md                      # 完全版レポート（16KB）
    ├── analyze_posture.py             # 解析スクリプト（fNIRS対応）
    ├── *.csv                          # 生成データ（7ファイル）
    └── archive/                       # 旧バージョンのスクリプト・レポート
```

### 🔬 主要な発見（Phase 1）

**データ**: 2025年12月24日 07:44-08:17（約33分、10セグメント）

#### 1. Yaw RMS（左右回転の揺れ）が最強の指標

| 相関 | 係数 | p値 | 解釈 |
|------|------|-----|------|
| **Yaw RMS ⇔ HbO（脳血流）** | **ρ=-0.915*** | 0.0002 | 揺れ↑ → 脳血流↓ |
| **Yaw RMS ⇔ Beta波（緊張）** | **r=+0.930*** | 0.0001 | 揺れ↑ → 緊張↑ |
| Yaw RMS ⇔ Delta波（眠気） | r=+0.850** | 0.0019 | 揺れ↑ → 眠気↑ |
| Yaw RMS ⇔ Theta波 | r=+0.831** | 0.0029 | 揺れ↑ → Theta↑ |
| Yaw RMS ⇔ Alpha波 | r=-0.672* | 0.0334 | 揺れ↑ → リラックス↓ |

#### 2. センサーの補完的役割

| センサータイプ | 強く関連する指標 | 相関 | 解釈 |
|--------------|----------------|------|------|
| **ジャイロ系**<br>(gyro_rms, yaw_rms) | HbO（脳血流） | ρ=-0.79～-0.92 | 回転運動 → 脳血流低下 |
| **ジャイロ系** | Beta波（緊張） | r=+0.92～+0.93 | 回転運動 → 緊張増加 |
| **加速度系**<br>(motion_index) | HbR（脱酸素Hb） | ρ=+0.79 | 微細な動き → 酸素消費↑ |

#### 3. 生理学的メカニズムの仮説

```
姿勢不安定 → 頭部回転運動↑ → 前庭系活性化
                           ↓
                 前頭皮質の血流低下（HbO↓）
                           ↓
               眠気・疲労 ←→ 緊張・努力
              （Delta↑）    （Beta↑）
                           ↓
                 リラックスの喪失（Alpha↓）
```

### 📊 生成されたデータ

| ファイル | 説明 | サイズ |
|---------|------|--------|
| `REPORT.md` | 完全版レポート（主要な発見、考察、データサマリー） | 16KB |
| `analyze_posture.py` | 解析スクリプト（fNIRS対応） | 11KB |
| `combined_data.csv` | 統合データ（EEG + fNIRS + HR + Posture） | 15列×10行 |
| `correlation_results.csv` | EEG相関分析結果 | 35行 |
| `fnirs_correlation_results.csv` | fNIRS相関分析結果 | 14行 |
| `band_powers.csv` | バンドパワー時系列 | 5列×10行 |
| `fnirs_data.csv` | fNIRS統計量 | 2列×10行 |
| `hr_data.csv` | 心拍数時系列 | 1列×10行 |
| `posture_data.csv` | 坐相統計量時系列 | 7列×10行 |

---

## 次にやること（Phase 2: 複数セッション検証）

### 🎯 目的

**単一セッションで得られた仮説が、複数のデータに対して再現するかを検証する**

### 📋 検証したい仮説

1. **Yaw RMS ⇔ HbO（脳血流）の負の相関** (ρ=-0.915) が再現するか
2. **Yaw RMS ⇔ Beta波（緊張）の正の相関** (r=+0.930) が再現するか
3. **ジャイロ系とHbO、加速度系とHbRの補完的役割**が一般的か
4. **生理学的メカニズム仮説**が支持されるか

### 🗂️ 利用可能なデータ

```bash
data/mindMonitor_2025-12-24--07-44-35_*.csv  # Phase 1で使用（約33分）
data/mindMonitor_2025-12-25--07-44-46_*.csv  # 新しいデータ
data/mindMonitor_2025-12-22--07-34-33_*.csv
data/mindMonitor_2025-12-19--07-38-20_*.csv
data/mindMonitor_2025-12-11--07-36-21_*.csv
# ... 他にも複数のセッションデータあり
```

### 🔧 必要な実装

#### 1. バッチ処理スクリプトの作成

```python
# 例: batch_analyze_posture.py
# 複数のCSVファイルに対して analyze_posture.py を実行
# 結果を統合してクロスセッション分析
```

#### 2. クロスセッション統計分析

- 各セッションでの相関係数を計算
- セッション間での相関係数の一貫性を評価
- メタ分析（相関係数の平均、信頼区間）

#### 3. レポート生成

- セッションごとの結果サマリー
- 全体的な傾向の可視化
- 仮説の採択/棄却の判断

### 📁 推奨フォルダ構成（Phase 2）

```
issues/008_sazou/
├── README.md
├── HANDOFF.md
├── 2025-12-24_single_session/         # Phase 1
│   ├── REPORT.md
│   └── ...
└── multi_session_analysis/            # Phase 2（新規作成）
    ├── batch_analyze_posture.py       # バッチ処理スクリプト
    ├── cross_session_stats.py         # クロスセッション統計
    ├── sessions/                      # セッションごとの結果
    │   ├── 2025-12-24/
    │   │   ├── correlation_results.csv
    │   │   └── summary.txt
    │   ├── 2025-12-25/
    │   └── ...
    ├── CROSS_SESSION_REPORT.md        # 統合レポート
    └── meta_analysis_results.csv      # メタ分析結果
```

---

## 技術的詳細

### 使用ライブラリ

```python
from lib import (
    load_mind_monitor_csv,
    prepare_mne_raw,
    get_optics_data,          # fNIRS用
    analyze_fnirs,
    get_heart_rate_data,
)
from lib.statistical_dataframe import create_statistical_dataframe
from lib.sensors.imu import compute_posture_statistics
```

### データ処理パイプライン

```python
# 1. データ読み込み
df = load_mind_monitor_csv(csv_path)

# 2. MNE Raw作成
raw_data = prepare_mne_raw(df)
raw = raw_data['raw']

# 3. fNIRS解析
optics_data = get_optics_data(df)
fnirs_results = analyze_fnirs(optics_data)

# 4. 心拍数解析
hr_data = get_heart_rate_data(df)

# 5. Statistical DataFrame生成（坐相統計量含む）
stat_df = create_statistical_dataframe(
    raw=raw,
    segment_minutes=3,
    fnirs_results=fnirs_results,
    hr_data=hr_data,
    df=df,  # IMU統計量のため
)

# 6. 相関分析
# Pearson (EEG), Spearman (fNIRS)
```

### 重要なパラメータ

- **セグメント長**: 3分（`segment_minutes=3`）
- **ウォームアップ**: なし（`warmup_minutes=0.0`）
- **相関手法**:
  - EEG: Pearson相関係数
  - fNIRS: Spearman順位相関係数（正規分布しないため）

### 坐相統計量（7指標）

```python
posture_vars = [
    'motion_index_mean',      # 平均モーション指数 [g]
    'motion_index_max',       # 最大モーション指数 [g]
    'gyro_rms',              # ジャイロRMS [deg/s]
    'gyro_rms_corrected',    # ゼロ点補正版ジャイロRMS [deg/s]
    'pitch_angle',           # Pitch角度（前後傾き） [deg]
    'roll_angle',            # Roll角度（左右傾き） [deg]
    'yaw_rms',               # Yaw RMS（左右回転の揺れ） [deg/s]
]
```

### 主要なEEG/fNIRS指標

```python
eeg_vars = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
fnirs_vars = ['hbo_mean', 'hbr_mean']  # 酸素化Hb、脱酸素Hb
```

---

## 注意事項・Tips

### ⚠️ よくあるエラー

1. **fNIRSデータがないCSV**
   - 一部の古いデータにはfNIRS（Optics列）がない可能性
   - エラーハンドリングで `try-except` を使用

2. **prepare_mne_raw()の戻り値**
   - 辞書を返す: `raw_data['raw']` でアクセス
   - 直接 `raw` として扱わないこと

3. **セッション時間の違い**
   - セグメント数が異なる場合の扱いに注意
   - 最小セグメント数でフィルタリングを推奨

### 💡 分析のヒント

1. **効果量も計算する**
   - 相関係数だけでなく、Cohen's d などの効果量を計算
   - 実用的な意義を評価

2. **信頼区間を表示**
   - 相関係数の95%信頼区間を計算
   - セッション間のばらつきを可視化

3. **外れ値のチェック**
   - セッション全体の相関係数分布を確認
   - 極端に異なるセッションがあれば原因を調査

---

## 参考資料

### ドキュメント

- **プロジェクトガイドライン**: `docs/AI-CODING.md`
- **Mind Monitor CSV仕様**: `docs/MIND_MONITOR_CSV_SPECIFICATION.md`
- **IMUセンサー設計**: `docs/IMU_SENSOR_DESIGN.md`

### コード

- **IMUライブラリ**: `lib/sensors/imu.py`
- **fNIRSライブラリ**: `lib/sensors/fnirs.py`
- **Statistical DataFrame**: `lib/statistical_dataframe.py`
- **レポート生成**: `scripts/generate_report.py`

### Phase 1のレポート

- **完全版**: `2025-12-24_single_session/REPORT.md`
- **スクリプト**: `2025-12-24_single_session/analyze_posture.py`

---

## 実行コマンド

### Phase 1の解析を再実行する場合

```bash
cd /home/tsu-nera/repo/satoru
source venv/bin/activate
python issues/008_sazou/2025-12-24_single_session/analyze_posture.py
```

### Phase 2: バッチ処理の実装例

```bash
# 複数セッションを順次処理
for csv in data/mindMonitor_2025-*.csv; do
    python issues/008_sazou/multi_session_analysis/batch_analyze_posture.py "$csv"
done

# クロスセッション統計
python issues/008_sazou/multi_session_analysis/cross_session_stats.py
```

---

## 期待される成果（Phase 2）

### 📈 定量的評価

- [ ] 各仮説の再現性（セッション数、再現率）
- [ ] 相関係数の平均値と信頼区間
- [ ] セッション間のばらつき評価

### 📝 レポート

- [ ] クロスセッション分析レポート
- [ ] 仮説の採択/棄却の判断
- [ ] 一般化可能性の評価

### 🔬 追加の知見

- [ ] セッション間で一貫している指標の特定
- [ ] 個人差・日内変動の評価
- [ ] より頑健な生理学的モデルの構築

---

**次のステップ**: `/clear` 後、Phase 2の実装を開始してください。

**連絡先**: このドキュメントに質問があれば、Claudeに聞いてください 😊
