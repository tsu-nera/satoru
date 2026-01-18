# Issue #014: 共鳴周波数（Resonance Frequency）HRV分析

## クイックナビゲーション

- ⚡ **[QUICK_PROTOCOL.md](QUICK_PROTOCOL.md)** - 簡易版プロトコル（25分、3分×6回）⭐推奨
- 📋 **[MEASUREMENT_PROTOCOL.md](MEASUREMENT_PROTOCOL.md)** - 詳細版プロトコル（70分、5分×9回）
- 📊 **[REPORT.md](REPORT.md)** - 2026-01-17測定の分析結果（要再測定）
- 💻 **[rf_hrv_analysis.py](rf_hrv_analysis.py)** - 分析スクリプト
- 🔧 **[generate_measurement_protocol.py](generate_measurement_protocol.py)** - 測定順序生成スクリプト
- 📁 **[data/](data/)** - Elite HRV測定データ
- 🖼️ **[img/](img/)** - 分析結果の可視化

---

## 目的

個人の**共鳴周波数（Resonance Frequency, RF）**を特定し、
HRVバイオフィードバック訓練の最適な呼吸レートを決定することを目的としています。

共鳴周波数とは、呼吸と心拍が最も同調し、HRVが最大化される呼吸レートです。
この周波数で呼吸することで、自律神経のバランスが最適化され、
ストレス軽減、感情調整、認知機能向上などの効果が期待されます。

## 背景

### 共鳴周波数（Resonance Frequency）とは

心臓血管系には固有の共鳴周波数（通常0.08-0.12 Hz、5-7 breaths/min）があり、
この周波数で呼吸すると、呼吸と心拍が同期し、HRVが最大化されます。

**主要な特徴:**
- 個人差が大きい（4.5-7.0 breaths/min）
- LF Power (0.04-0.15 Hz) が最大化される
- 副交感神経活動が最も活性化される
- 圧受容器反射の共鳴が起こる

### HRVバイオフィードバックへの応用

共鳴周波数での呼吸訓練は、以下の効果が実証されています：

1. **ストレス軽減**: 交感神経の抑制、副交感神経の活性化
2. **感情調整**: 不安・抑うつ症状の改善
3. **認知機能向上**: 集中力、意思決定能力の向上
4. **身体症状の改善**: 高血圧、喘息、慢性疼痛など

## 分析方法

### 共鳴周波数の特定プロトコル

1. **呼吸ペーシング試験**:
   - 複数の呼吸レート（4.5-7.0 bpm）で5分間測定
   - 各測定間に2-3分の休憩
   - Elite HRVアプリでRR間隔を記録

2. **HRV指標の計算**:
   - **LF Power** (0.04-0.15 Hz): 主指標
   - **RMSSD**: 短期的HRV
   - **HR Mean / Max-Min**: 心拍数統計

3. **最適レートの特定**:
   - LF Powerが最大となる呼吸レートが共鳴周波数
   - パワースペクトル密度（PSD）で確認

### 測定上の注意点

- **測定順序効果**: 時間経過による状態変化を評価
- **再現性**: 同じ呼吸レートで複数回測定
- **環境統制**: 同じ時間帯、静かな環境で測定
- **事前準備**: 測定前10分間の安静時間

## 現状の分析結果

### 2026-01-17測定（要再測定）

⚠️ **警告**: この測定には測定順序効果（相関係数 0.560）が見られ、
結果の信頼性が低い可能性があります。

| 呼吸レート | LF Power | 評価 |
|:----------|--------:|:-----|
| 4.5 bpm   | 1675.28 ms² | 最大（ただし測定順序が最後） |
| 5.0 bpm   | 1268.40 ms² | 2番目 |
| 5.5 bpm   | 756.44 ms² | 中程度 |
| 6.0 bpm   | 625.54 ms² | 低い |
| 6.5 bpm   | 666.09 ms²（平均） | 低い、再現性に課題（CV 19.16%） |

**暫定的な共鳴周波数**: 4.5-5.0 bpm付近

**問題点**:
1. 測定順序とLF Powerに正の相関（後半ほど高い）
2. 同じ6.5 bpmで31.3%の差（575.83 vs 756.35 ms²）
3. 4.5 bpm未満の測定がなく、ピークが確定していない

## ファイル構成

```
014_rf_hrv/
├── README.md                          # このファイル（Issue概要）
├── QUICK_PROTOCOL.md                  # 簡易版プロトコル（25分） ⭐推奨
├── MEASUREMENT_PROTOCOL.md            # 詳細版プロトコル（70分）
├── REPORT.md                          # 分析結果レポート
├── rf_hrv_analysis.py                 # 分析スクリプト
├── generate_measurement_protocol.py   # 測定順序生成スクリプト
├── measurement_protocol.txt           # スクリプト用プロトコルデータ
├── data/                              # Elite HRV測定データ
│   ├── 2026-01-17 17-47-13.txt        # [旧] Trial 1: 6.5 bpm
│   ├── 2026-01-17 17-51-11.txt        # [旧] Trial 2: 6.0 bpm
│   ├── 2026-01-17 17-54-41.txt        # [旧] Trial 3: 5.5 bpm
│   ├── 2026-01-17 17-57-47.txt        # [旧] Trial 4: 5.0 bpm
│   ├── 2026-01-17 18-00-59.txt        # [旧] Trial 5: 4.5 bpm
│   ├── 2026-01-17 18-11-56.txt        # [旧] Trial 6: 6.5 bpm（再測定）
│   └── [再測定データをここに保存]    # 2026-01-XX以降
└── img/                               # 分析結果の可視化
    └── rf_hrv_analysis.png
```

## 使用方法

```bash
# 仮想環境を有効化
source venv/bin/activate

# 分析スクリプトを実行
python issues/014_rf_hrv/rf_hrv_analysis.py

# 出力:
# - issues/014_rf_hrv/REPORT.md（マークダウンレポート）
# - issues/014_rf_hrv/img/rf_hrv_analysis.png（可視化）
```

## 依存ライブラリ

- `scipy`: 信号処理、パワースペクトル密度計算
- `matplotlib`: 可視化
- `pandas`, `numpy`: データ処理
- `lib.loaders.elite_hrv`: Elite HRVデータローダー

## 再測定プロトコル（2026-01-18更新）

### ⚡ 簡易版プロトコル（推奨）

**所要時間**: 約25分（3分×6回 + 休憩1分×5回）

**詳細**: [QUICK_PROTOCOL.md](QUICK_PROTOCOL.md)を参照

**測定順序**（ランダム化済み）:
```
Trial 1: 5.0 bpm → Trial 2: 3.5 bpm → Trial 3: 6.0 bpm
Trial 4: 4.5 bpm → Trial 5: 5.5 bpm → Trial 6: 4.0 bpm
```

**手順**:
1. 静かな環境、測定前5分安静
2. 各トライアル3分間測定
3. データ保存後1分休憩
4. 分析: `python issues/014_rf_hrv/rf_hrv_analysis.py`

### 📋 詳細版プロトコル（オプション）

**所要時間**: 約70分（5分×9回 + 休憩2-3分×8回）

**詳細**: [MEASUREMENT_PROTOCOL.md](MEASUREMENT_PROTOCOL.md)を参照

より厳密な測定が必要な場合に使用

## 主要参考文献

### 共鳴周波数とHRVバイオフィードバック

1. **Lehrer, P., & Gevirtz, R. (2014).**
   Heart rate variability biofeedback: how and why does it work?
   *Frontiers in psychology*, 5, 756.
   https://doi.org/10.3389/fpsyg.2014.00756
   - HRVバイオフィードバックのメカニズム

2. **Lehrer, P. M., Vaschillo, E., & Vaschillo, B. (2000).**
   Resonant frequency biofeedback training to increase cardiac variability:
   Rationale and manual for training.
   *Applied psychophysiology and biofeedback*, 25(3), 177-191.
   - 共鳴周波数訓練の原理と手順

3. **Steffen, P. R., Austin, T., DeBarros, A., & Brown, T. (2017).**
   The Impact of Resonance Frequency Breathing on Measures of Heart Rate Variability,
   Blood Pressure, and Mood.
   *Frontiers in Public Health*, 5, 222.
   https://doi.org/10.3389/fpubh.2017.00222
   - 共鳴周波数呼吸の効果検証

4. **Steffen, P. R., et al. (2020).**
   A Practical Guide to Resonance Frequency Assessment for Heart Rate Variability Biofeedback.
   *Frontiers in Neuroscience*, 14, 570400.
   https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.570400/full
   - 共鳴周波数評価の実践ガイド

### 圧受容器反射と心臓血管系

5. **Vaschillo, E., Vaschillo, B., & Lehrer, P. (2006).**
   Characteristics of resonance in heart rate variability stimulated by biofeedback.
   *Applied psychophysiology and biofeedback*, 31(2), 129-142.
   - 共鳴現象の生理学的メカニズム

## 今後の計画

### Phase 1: 測定プロトコルの改善（優先度: 最優先）

- [x] Elite HRVデータローダーの作成
- [x] 分析スクリプトの作成
- [x] ランダム化された測定プロトコルの作成（2026-01-18）
- [x] 簡易版プロトコルの作成（QUICK_PROTOCOL.md、25分）
- [ ] 🎯 **次**: 再測定の実施（6回 × 3分 = 約25分）
- [ ] データ分析と結果検証

### Phase 2: 共鳴周波数の精密特定（優先度: 高）

- [ ] 暫定的な共鳴周波数周辺の詳細測定（0.25 bpm刻み）
- [ ] 最適レートでの長期測定（10分以上）
- [ ] 異なる時間帯での測定（日内変動の確認）
- [ ] 異なる状態での測定（リラックス vs 集中）

### Phase 3: 訓練プロトコルの開発（優先度: 中）

- [ ] 共鳴周波数での呼吸訓練プログラム作成
- [ ] 訓練効果の経時的追跡
- [ ] HRV指標の改善度合いを評価
- [ ] 主観的ウェルビーイング指標との相関

### Phase 4: 応用展開（優先度: 低）

- [ ] 瞑想実践との統合
- [ ] ストレス管理プロトコルへの組み込み
- [ ] 他者とのデータ比較（可能であれば）
- [ ] 長期的な健康効果の追跡

---

**Issue作成**: 2026-01-18
**最終更新**: 2026-01-18
**ステータス**: 簡易版プロトコル完成、実測定待ち（QUICK_PROTOCOL.mdを参照、約25分）
