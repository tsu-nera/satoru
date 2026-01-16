# Issue #012: 瞑想の指標となるHRV論文調査と検証

## クイックナビゲーション

- 📚 **[PAPER_INVESTIGATION.md](PAPER_INVESTIGATION.md)** - 文献調査の詳細レポート
- 📊 **[RSA_REPORT.md](RSA_REPORT.md)** - 2026-01-12セッションの分析結果
- 💻 **[rsa_analysis.py](rsa_analysis.py)** - 分析スクリプト

---

## 目的

禅瞑想の習熟度を客観的に評価するための心拍変動（HRV）指標を特定し、
自身の瞑想実践データで検証することを目的としています。

本issueでは以下の3つのアプローチを統合します：

1. **文献調査**: 信頼性の高いHRV指標の特定
2. **分析手法の実装**: DFA、RSA、ウェーブレット解析
3. **実データ検証**: 自己の瞑想セッションデータでの評価

## 背景

### 主要研究グループの知見

#### 日本グループ（Hoshiyama et al., 2008）
- **手法**: DFA（Detrended Fluctuation Analysis）
- **知見**:
  - 経験豊富な瞑想者: DFA ≈ 0.5（ランダムウォーク、深い瞑想状態）
  - 初心者: DFA ≈ 0.78（より規則的なパターン）

#### スペイングループ（Peressutti, Mesa et al., 2010-2014）
- **手法**: ウェーブレット解析、RSA定量化、EEG
- **知見**:
  - 経験者はRSA振幅が高く、副交感神経活動が強化
  - 呼吸と心拍の高度な同期
  - 前頭部シータ波の低下（不必要な情報処理の抑制）

## 分析内容

1. **RSA振幅の定量化**: HF Power（0.15-0.4 Hz）による評価
2. **DFA（ゆらぎ解析）**: 心拍変動の長期相関を評価
3. **呼吸-心拍コヒーレンス**: 呼吸と心拍の同期性を定量化
4. **自律神経バランス**: LF/HF比による評価
5. **時系列変化**: 3分ごとのウィンドウで各指標を追跡

## 主要な発見

### データ: selfloops_2026-01-12--06-21-05.csv (32.89分)

| 指標 | 値 | 評価 |
|:-----|---:|:-----|
| **平均呼吸数** | 4.2 ± 1.2 bpm | 非常に深い瞑想呼吸 |
| **RSA振幅（HF Power）** | 488.8 ± 476.4 ms² | 中程度の副交感神経活動 |
| **DFA α1** | 1.478 ± 0.104 | 規則的なパターン（初心者レベル以上） |
| **LF/HF比** | 7.11 ± 4.11 | 交感神経優位 |
| **呼吸-心拍コヒーレンス** | 1.000 @ 0.219 Hz | 完璧な同期 |

### 解釈

1. **DFA α1 = 1.478**: 文献の初心者レベル（0.78）よりも高く、より規則的な心拍パターンを示す
   - これは「1/fノイズ」に近い値で、生理学的に正常だが、深い瞑想状態の特徴（DFA ≈ 0.5）とは異なる

2. **呼吸数 4.2 bpm**: 1呼吸あたり約14秒と非常に深い呼吸
   - これは経験豊富な瞑想者の特徴

3. **高いLF/HF比（7.11）**: 交感神経優位で、完全なリラックス状態ではない
   - 瞑想中に何らかの緊張や集中が存在した可能性

4. **完璧なコヒーレンス（1.000）**: 呼吸と心拍が高度に同期
   - これは良好な呼吸調節を示す

### 矛盾点の考察

- **深い呼吸（4.2 bpm）** と **高いLF/HF比（7.11）** は一見矛盾
- 深い呼吸は通常、副交感神経を活性化し、LF/HF比を下げるはず
- この矛盾は以下の可能性を示唆：
  1. 瞑想中の集中による緊張（交感神経活性化）
  2. データの質の問題（アーティファクト）
  3. 個人特性（自律神経の反応パターン）

## ファイル構成

```
012_rsa_analysis/
├── README.md                   # このファイル（Issue概要）
├── PAPER_INVESTIGATION.md      # 文献調査レポート
├── RSA_REPORT.md               # 自己データ分析レポート
├── rsa_analysis.py             # RSA分析スクリプト
└── rsa_analysis.png            # 可視化結果
```

### ファイル説明

| ファイル | 内容 | 状態 |
|:--------|:-----|:-----|
| **PAPER_INVESTIGATION.md** | 日本・スペイン研究グループの文献調査<br>分析手法の比較、今後の調査方向性 | ✓ 完了 |
| **RSA_REPORT.md** | 2026-01-12セッションの分析結果<br>DFA、RSA、コヒーレンス評価 | ✓ 完了<br>⧗ 出典修正済 |
| **rsa_analysis.py** | RSA分析の実装コード<br>HF Power, DFA, コヒーレンス計算 | ✓ 実装済 |

## 使用方法

```bash
# 仮想環境を有効化
source venv/bin/activate

# RSA分析を実行
python issues/012_rsa_analysis/rsa_analysis.py
```

## 依存ライブラリ

- `neurokit2`: HRV分析
- `scipy`: 信号処理、コヒーレンス計算
- `matplotlib`: 可視化
- `pandas`, `numpy`: データ処理

## 既存ライブラリの活用

- `lib/sensors/ecg/respiration.py`:
  - `calculate_breathing_rate()`: ECG-Derived Respiration
  - `analyze_breathing_hrv_correlation()`: 呼吸数とHRV相関

- `lib/loaders/selfloops.py`:
  - `load_selfloops_csv()`: データ読み込み
  - `get_hrv_data()`: HRVデータ抽出

## 主要参考文献

### 禅瞑想とHRV

1. **Hoshiyama, M., & Hoshiyama, A. (2008).**
   Heart rate variability associated with experienced Zen meditation.
   *2008 Computers in Cardiology*.
   https://ieeexplore.ieee.org/document/4749105
   - DFA分析による習熟度評価の基準値

2. **Peressutti, C., Martín-González, J. M., García-Manso, J. M., & Mesa, D. (2010).**
   Heart rate dynamics in different levels of Zen meditation.
   *International Journal of Cardiology*, 145(1), 142-146.
   - ウェーブレット解析とRSA定量化

3. **Peressutti, C., Martín-González, J. M., García-Manso, J. M., & Mesa, D. (2014).**
   Lower trait frontal theta activity in mindfulness meditators.
   *Arquivos de neuro-psiquiatria*, 72(9), 687-93.
   - EEG研究、前頭部シータ波活動

### HRV分析手法

4. **Task Force of ESC/NASPE (1996).**
   Heart rate variability: standards of measurement, physiological interpretation and clinical use.
   *Circulation*, 93(5), 1043-1065.
   - HRV測定の標準ガイドライン

5. **Peng, C. K., et al. (1995).**
   Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series.
   *Chaos*, 5(1), 82-87.
   - DFA手法の原論文

### 関連リソース

6. **sugimotot (2024).**
   HRVによる禅瞑想レベルの分析実践例.
   https://note.com/sugimotot/n/naebb0bf7694f
   - 日本の曹洞宗実践者による実践報告

## 今後の計画

### Phase 1: 文献調査の深堀り（優先度: 高）
- [ ] Hoshiyama et al. (2008) 全文PDFの入手
- [ ] Peressutti et al. (2010) 全文PDFの入手
- [ ] DFA基準値の詳細統計情報の確認
- [ ] 他の禅HRV研究の追加調査

### Phase 2: 分析手法の改善（優先度: 最優先）
- [ ] DFA分析の精度向上と詳細評価
- [ ] ウェーブレット解析の実装（Peressuttiモデル）
- [ ] 両手法の比較と相関分析
- [ ] 自動レポート生成の改善

### Phase 3: 縦断的データ収集（優先度: 高）
- [ ] 複数セッションの測定と比較
- [ ] 瞑想習熟度の経時的変化追跡
- [ ] 異なる条件（朝/夜、時間、環境）での測定
- [ ] 個人ベースラインの確立

### Phase 4: コミュニティ連携（優先度: 中-低）
- [ ] sugimotot氏（note著者）との情報交換
- [ ] 他の実践者とのデータ比較
- [ ] 研究者への問い合わせ（可能であれば）
- [ ] オープンデータ化の検討

---

**Issue作成**: 2026-01-13
**最終更新**: 2026-01-14
**ステータス**: 文献調査完了、分析手法改善フェーズ
