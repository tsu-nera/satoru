# 複数セッション坐相分析（Phase 2）

Phase 1で発見された坐相と脳波・脳血流の関係を、複数セッションで検証したプロジェクトです。

---

## クイックスタート

### レポートを読む

📄 **[REPORT.md](REPORT.md)** - 分析結果の完全版レポート

### スクリプトを実行

```bash
# 仮想環境を有効化
source venv/bin/activate

# バッチ処理（5セッション解析）
python issues/008_sazou/multi_sessions/batch_analyze_posture.py

# クロスセッション統計分析
python issues/008_sazou/multi_sessions/cross_session_stats.py
```

---

## 主要な発見

### ✅ 強く支持された仮説（再現性≥80%）

1. **Yaw RMS ⇔ HbO（脳血流）の負の相関**
   - ρ=-0.735（平均）、再現性80%
   - 頭部の左右回転運動が増えると脳血流が低下

2. **Gyro RMS ⇔ HbO（脳血流）の負の相関**
   - ρ=-0.707（平均）、再現性80%
   - ジャイロ系指標が脳血流低下の強い予測因子

### △ 部分的に支持された仮説（再現性40%）

3. Yaw RMS ⇔ Beta波（緊張）の正の相関
4. Yaw RMS ⇔ Delta波の正の相関
5. Yaw RMS ⇔ Theta波の正の相関

### ✗ 棄却された仮説

6. Yaw RMS ⇔ Alpha波の負の相関（再現性20%）
7. Motion Index ⇔ HbR の正の相関（再現性20%）

---

## データセット

| セッション | 時間 | セグメント数 |
|-----------|------|------------|
| 2025-12-11 | 35.2分 | 11 |
| 2025-12-19 | 34.1分 | 11 |
| 2025-12-22 | 33.9分 | 11 |
| 2025-12-24 | 32.1分 | 10 |
| 2025-12-25 | 38.6分 | 12 |
| **合計** | **173.9分** | **55** |

---

## フォルダ構成

```
multi_sessions/
├── README.md                          # このファイル
├── REPORT.md                          # 完全版レポート
├── batch_analyze_posture.py           # バッチ処理スクリプト
├── cross_session_stats.py             # クロスセッション統計
├── hypothesis_validation_results.csv  # 仮説検証結果
├── all_sessions_summary.csv           # 全セッションサマリー
└── sessions/                          # 各セッションの結果
    ├── 2025-12-11--07-36-21/
    ├── 2025-12-19--07-38-20/
    ├── 2025-12-22--07-34-33/
    ├── 2025-12-24--07-44-35/
    └── 2025-12-25--07-44-46/
```

### 各セッションディレクトリの内容

- `combined_data.csv` - 統合データ（EEG + fNIRS + HR + Posture）
- `correlation_results.csv` - EEG相関分析結果
- `fnirs_correlation_results.csv` - fNIRS相関分析結果
- `summary.txt` - セッションサマリー
- `band_powers.csv` - EEGバンドパワー時系列
- `fnirs_data.csv` - fNIRS時系列
- `hr_data.csv` - 心拍数時系列
- `posture_data.csv` - 坐相統計量時系列

---

## 解析パラメータ

- **セグメント長**: 3分
- **ウォームアップ**: なし
- **EEG相関**: Pearson相関係数
- **fNIRS相関**: Spearman順位相関係数
- **有意水準**: p < 0.05

---

## 参照

- **Phase 1レポート**: `../2025-12-24_single_session/REPORT.md`
- **引き継ぎドキュメント**: `../HANDOFF.md`
- **プロジェクトREADME**: `../README.md`

---

## ライセンス

このプロジェクトは研究目的で作成されました。
