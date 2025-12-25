# 坐相解析プロジェクト (Issue #008)

坐禅中の体軸の揺らぎと脳波・脳血流・心拍数の関係を定量的に解析するプロジェクトです。

## 背景

藤田一照氏の「現代坐禅講義」で指摘された「不動の中の動」を科学的に検証します。Museセンサー(加速度計・ジャイロスコープ・EEG・fNIRS・PPG)を用いて、坐禅中の身体的揺らぎと心の状態の関係を定量化します。

## プロジェクト構成

### Phase 1: 単一セッション分析 ✅ 完了
- **目的**: 坐相とEEG/fNIRS/心拍数の関係性を探索
- **データ**: 2025年12月24日（約33分、10セグメント）
- **成果**: Yaw RMSとHbO（脳血流）の極めて強い負の相関（ρ=-0.915）を発見

### Phase 2: 複数セッション検証 🚧 次のステップ
- **目的**: Phase 1で得られた仮説の再現性を検証
- **データ**: 複数日のセッションデータ
- **詳細**: [`HANDOFF.md`](HANDOFF.md) を参照

## ファイル構成

```
issues/008_sazou/
├── README.md                              # このファイル
├── HANDOFF.md                             # 🔄 引き継ぎドキュメント（Phase 2への移行ガイド）
└── 2025-12-24_single_session/             # Phase 1: 単一セッション分析
    ├── REPORT.md                          # 📊 完全版レポート（16KB）
    ├── analyze_posture.py                 # 解析スクリプト（fNIRS対応）
    ├── combined_data.csv                  # 統合データ（EEG+fNIRS+HR+Posture）
    ├── correlation_results.csv            # EEG相関分析結果
    ├── fnirs_correlation_results.csv      # fNIRS相関分析結果
    ├── band_powers.csv                    # バンドパワー時系列
    ├── fnirs_data.csv                     # fNIRS統計量
    ├── hr_data.csv                        # 心拍数時系列
    ├── posture_data.csv                   # 坐相統計量時系列
    └── archive/                           # 旧バージョン（10秒間隔解析）
        ├── REPORT_old.md
        ├── analyze_mind_body_relationship_old.py
        └── *.csv                          # 旧データファイル
```

### 📖 詳細ドキュメント

- **Phase 1レポート**: [`2025-12-24_single_session/REPORT.md`](2025-12-24_single_session/REPORT.md)
  - 単一セッション分析の完全版レポート（16KB）
  - 主要な発見、考察、生理学的メカニズムの仮説
- **引き継ぎドキュメント**: [`HANDOFF.md`](HANDOFF.md)
  - Phase 2への移行ガイド
  - 技術的詳細、実装のヒント、期待される成果

## 使い方

### Phase 1の解析を再実行する場合

```bash
cd /home/tsu-nera/repo/satoru
source venv/bin/activate
python issues/008_sazou/2025-12-24_single_session/analyze_posture.py
```

### Phase 2の実装（次のステップ）

詳細は [`HANDOFF.md`](HANDOFF.md) を参照してください。

### 前提条件

- `data/mindMonitor_*.csv` が存在すること
- 必要なPythonライブラリ:
  - pandas, numpy, scipy
  - mne (EEG解析用)
  - lib/ 配下のカスタムライブラリ

## 主要な発見（Phase 1）

### 1. Yaw RMS（左右回転の揺れ）が最強の指標 ⭐

| 相関 | 係数 | 有意性 | 解釈 |
|------|------|--------|------|
| **Yaw RMS ⇔ HbO（脳血流）** | **ρ=-0.915*** | p<0.001 | 揺れ↑ → 脳血流↓ |
| **Yaw RMS ⇔ Beta波（緊張）** | **r=+0.930*** | p<0.001 | 揺れ↑ → 緊張↑ |
| Yaw RMS ⇔ Delta波（眠気） | r=+0.850** | p<0.01 | 揺れ↑ → 眠気↑ |
| Yaw RMS ⇔ Theta波 | r=+0.831** | p<0.01 | 揺れ↑ → Theta↑ |
| Yaw RMS ⇔ Alpha波 | r=-0.672* | p<0.05 | 揺れ↑ → リラックス↓ |

**意義**: 全ての生体指標の中で、Yaw RMS（左右回転の揺れ）が最も強い相関を示しました。

### 2. センサーの補完的役割の発見 🔬

| センサー | 強く関連する指標 | 相関 | 解釈 |
|---------|----------------|------|------|
| **ジャイロ系** (gyro_rms, yaw_rms) | HbO（脳血流） | ρ=-0.79～-0.92*** | 回転運動 → 脳血流低下 |
| **ジャイロ系** | Beta波（緊張） | r=+0.92～+0.93*** | 回転運動 → 緊張増加 |
| **加速度系** (motion_index) | HbR（脱酸素Hb） | ρ=+0.79** | 微細な動き → 酸素消費↑ |

**示唆**:
- **ジャイロセンサー（角速度）**: 脳血流の低下と緊張状態を反映
- **加速度センサー（微細な動き）**: 酸素消費の増加を反映
- 両者は異なる脳の代謝プロセスを捉えている

### 3. 生理学的メカニズムの仮説 🧠

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

**検証が必要**: Phase 2で複数セッションでの再現性を確認

## 研究の意義

### 学術的意義
- マインド-ボディ研究への新知見
- 身体的微細運動と脳波・脳血流の強い関連を実証
- 禅仏教の「身心一如」の科学的検証
- fNIRS応用の新展開: 脳血流と身体動態の関連を示した可能性

### 実践的応用
- 坐相の質を客観的に評価
- リアルタイムフィードバックによる坐禅指導
- 個人に最適化された瞑想法の提案
- fNIRSによる脳活動モニタリング

## 参考文献

### 書籍
- 藤田一照『現代坐禅講義』

### 学術論文

**姿勢計測**:
- Centre of Pressure Estimation Using IMUs (MDPI, 2020)
- Using Accelerometer and Gyroscopic Measures to Quantify Postural Stability (PMC)

**脳波**:
- Lomas, T., et al. (2015). "A systematic review of the neurophysiology of mindfulness on EEG oscillations."

**fNIRS**:
- Ferrari, M., & Quaresima, V. (2012). "A brief review on the history of human functional near-infrared spectroscopy (fNIRS) development and fields of application."
- Herold, F., et al. (2018). "Functional near-infrared spectroscopy in movement science: a systematic review on cortical activity in postural and walking tasks."

## ライセンス

研究目的での使用を想定しています。

## 連絡先

プロジェクト管理者: [あなたの情報]

---

**最終更新**: 2025-12-24
