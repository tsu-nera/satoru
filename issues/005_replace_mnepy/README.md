# Issue #005: MNE-Python活用強化

**ステータス**: 📋 計画中
**優先度**: 中
**開始日**: 2025-11-11

---

## 🎯 目的

独自実装からMNE-Pythonの標準機能への移行を進め、科学的妥当性と保守性を向上させる。

---

## 📄 ドキュメント

- **[PLAN.md](./PLAN.md)** - 詳細な移行計画
  - 現状分析
  - Phase別実装計画
  - 参考資料
  - 成功指標
- **[CURRENT_STATUS.md](./CURRENT_STATUS.md)** - 現在のMNE使用状況
  - 既存実装の分析
  - 使用率評価
  - 依存関係マップ
- **[MUSE_LSL_ANALYSIS.md](./MUSE_LSL_ANALYSIS.md)** - muse-lslプロジェクト分析
  - muse-lsl vs MNE-Python比較
  - 活用可能性の評価
  - 参考にできる実装の抽出

---

## 🔍 背景

### 問題意識

独自実装よりもMNE-Pythonの標準機能を使うべきではないか？

**理由**:
1. 科学的妥当性（学術界で検証済み）
2. 標準化（他研究との比較可能性）
3. 保守性（コミュニティによる改善）

### 調査結果

✅ **良いニュース**: 既に主要な信号処理でMNE-Pythonを使用している！

- PSD計算: `raw.compute_psd()` ✓
- スペクトログラム: `tfr_array_morlet()` ✓
- フィルタリング: `raw.filter()` ✓
- RawArray作成: `mne.io.RawArray` ✓

🟡 **独自実装（これは適切）**:
- Mind Monitor固有の処理（CSVパース、HSI品質評価）
- 高度な指標（Fmθ、FAA、Spectral Entropy、PAF）
- カスタムレポート生成

### コミュニティ事例

複数のGitHubリポジトリで同様のアプローチを発見：

- [muse-lsl](https://github.com/alexandrebarachant/muse-lsl) ⭐708 - リアルタイムストリーミング特化
  - **分析結果**: 我々の目的（オフライン分析）には不適合だが、ノッチフィルタ実装は参考になる
  - 詳細: [MUSE_LSL_ANALYSIS.md](./MUSE_LSL_ANALYSIS.md)
- [muse-lsl-python](https://github.com/urish/muse-lsl-python) - Jupyter Notebook形式の実験例
- [Muse-MotorImageryClassification](https://github.com/vinayakr99/Muse-MotorImageryClassification) - Mind Monitor → MNE → ML

**結論**: **MNE-Pythonを使い続けるべき** ✓

---

## 🚀 改善提案（概要）

### Phase 1: 前処理強化 ⭐高優先度

- ノッチフィルタ追加（電源ノイズ除去）
- バンドパスフィルタの明示化

### Phase 2: Epochベース解析 🟡中優先度

- 時間セグメント分析をMNE Epochsに移行
- 統計解析の標準化

### Phase 3: アーティファクト除去 🟡中優先度

- ICAによる眼電図・筋電図除去
- 4チャネルでの有効性を検証

### Phase 4: 可視化強化 🔵低優先度

- MNE標準プロット関数の活用
- インタラクティブな探索機能

---

## 📊 期待される効果

### メリット

1. **科学的信頼性** - 査読済みアルゴリズムの使用
2. **再現性** - 標準的な手法による実装
3. **保守性** - コミュニティによる継続的改善
4. **機能拡張** - ICA、統計解析など高度な機能へのアクセス

### デメリット

1. **学習コスト** - MNE-Python APIの習得が必要
2. **実装コスト** - 既存コードの一部修正が必要
3. **パフォーマンス** - 一部処理が遅くなる可能性

### リスク管理

- ✓ 段階的な移行（破壊的変更を避ける）
- ✓ 既存機能の保持（オプションとして追加）
- ✓ 後方互換性の維持（レポートフォーマット）

---

## 📅 スケジュール

| Phase | 内容 | 期間 | 優先度 |
|-------|------|------|--------|
| Phase 1 | 前処理強化 | 1-2日 | 高 |
| Phase 2 | Epochs導入 | 3-5日 | 中 |
| Phase 3 | ICA実験 | 5-7日 | 中 |
| Phase 4 | 可視化 | 2-3日 | 低 |

**合計見積もり**: 11-17日

---

## ✅ タスク

### Phase 1: 前処理強化

- [ ] ノッチフィルタの実装
- [ ] バンドパスフィルタの明示化
- [ ] 単体テストの追加
- [ ] ドキュメント更新

### Phase 2: Epochベース解析

- [ ] `epochs.py`モジュール作成
- [ ] 時間セグメント分析の移行
- [ ] 互換性確保
- [ ] パフォーマンステスト

### Phase 3: ICAによるアーティファクト除去

- [ ] ICA実装
- [ ] 4チャネルでの有効性評価
- [ ] オプション機能化
- [ ] ドキュメント作成

### Phase 4: 可視化強化

- [ ] MNEプロット統合
- [ ] レポート生成調整
- [ ] ドキュメント更新

---

## 📚 参考資料

### MNE-Python公式

- [公式サイト](https://mne.tools/)
- [チュートリアル](https://mne.tools/stable/auto_tutorials/index.html)
- [APIリファレンス](https://mne.tools/stable/python_reference.html)

### コミュニティ

- [MNE Forum](https://mne.discourse.group/)
- [GitHub Issues](https://github.com/mne-tools/mne-python/issues)

### 論文

- Gramfort et al. (2013). ["MEG and EEG data analysis with MNE-Python"](https://doi.org/10.3389/fnins.2013.00267)

---

## 💬 議論・質問

### Slackチャンネル
TBD

### GitHub Discussions
TBD

---

## 📝 関連イシュー

- なし（初回イシュー）

---

## 👥 担当者

- **リード**: @tsu-nera
- **レビュー**: TBD

---

## 📊 進捗状況

```
Phase 1: [✓] 100% - 完了 (2025-11-11)
Phase 2: [ ] 0%
Phase 3: [ ] 0%
Phase 4: [ ] 0%
```

**全体進捗**: 25% （Phase 1完了）

### Phase 1完了 ✅

- バンドパスフィルタの明示化（1-50Hz）
- ノッチフィルタの追加（50Hz, 60Hz電源ノイズ除去）
- 後方互換性の維持
- 詳細: [PHASE1_COMPLETE.md](./PHASE1_COMPLETE.md)

---

## 更新履歴

- 2025-11-11: Phase 1実装完了（バンドパスフィルタ明示化、ノッチフィルタ追加）
- 2025-11-11: muse-lsl分析完了（MUSE_LSL_ANALYSIS.md作成）
- 2025-11-11: イシュー作成、PLAN.md作成
