# Refactoring Plan: 脳波解析コードの設計改善

**Issue**: #006
**作成日**: 2025-11-16
**最終更新**: 2025-11-16
**ステータス**: In Progress（Phase 1-2完了）

---

## 背景と目的

### 現状の課題
脳波解析コードは段階的に機能追加されてきた結果、以下の問題が顕在化している：
- バンドパワー計算が複数箇所で異なる方法で実装（Mind Monitor列、`frequency.py`、`statistical_dataframe.py`）
- 可視化コードが計算ロジックと混在し、複数ファイルに分散
- データ層・計算層・可視化層の責務が不明瞭
- IAF計算が未統合、スケーリング単位が混在

### リファクタリングの目的
1. **計算の一貫性確保**: 全解析で統一的なバンドパワー計算を使用
2. **関心の分離**: データ生成・計算・可視化を明確に分離
3. **保守性向上**: 各層で独立したテスト・修正が可能な構造
4. **再利用性向上**: 計算ロジックを可視化から分離し、他用途での再利用を容易に

---

## 現状の問題点分析

### Critical Issue 1: バンドパワー計算の重複

| 計算方式 | 使用場所 | 問題点 |
|---------|----------|--------|
| **Mind Monitor CSV列** | 古い解析コード | 不正確（対数変換なし、統計処理なし） |
| **`frequency.calculate_psd()`** | 個別指標計算 | Welch法でPSD計算、統計処理なし |
| **`statistical_dataframe.create_statistical_dataframe()`** | セグメント分析 | MNE Epochs + Welch + Bels変換、Z-score外れ値除去 |

**影響**: どの計算を信頼すべきか不明瞭、解析結果の再現性が低い

**推奨方針**: `statistical_dataframe.py` の方式に統一（MNE Epochsベース、Bels単位、外れ値除去）

---

### Critical Issue 2: 可視化の無秩序な分散

| ファイル | 責務 | 問題 |
|---------|------|------|
| `lib/sensors/eeg/visualization.py` | EEG汎用プロット (31.7KB) | 肥大化 |
| `lib/visualization.py` | センサー横断可視化 | 境界不明瞭 |
| `frontal_theta.py` | Fmθ計算 + `plot_frontal_theta()` | 計算と可視化が同居 |
| `frontal_asymmetry.py` | FAA計算 + `plot_frontal_asymmetry()` | 計算と可視化が同居 |
| `segment_analysis.py` | セグメント分析 + `plot_segment_comparison()` | 計算と可視化が同居 |

**影響**:
- 単一責任原則（SRP）違反
- テスト時に matplotlib 依存が必須
- 同じようなグラフ生成コードが重複

**推奨方針**: 可視化を `lib/visualization/` ディレクトリに完全分離

---

### Warning Issue 3: スケーリング単位の混在

| 指標 | 単位 | 変換式 |
|------|------|--------|
| バンドパワー | **Bels** | `10*log10(μV²)` |
| FAA | **ln(μV²)** | 自然対数 |
| Fmθ | **μV²** | 実数値（対数化なし） |
| Spectral Entropy | **0-1正規化** | Shannon Entropy |

**影響**: 総合スコア計算で `_normalize_indicator()` による個別正規化が必要、解釈が複雑

**推奨方針**: 全指標を Bels（10*log10）に統一し、物理的意味を明確化

---

### Warning Issue 4: IAF計算の未統合

- `paf.py`（Peak Alpha Frequency）が存在するが、`statistical_dataframe.py` に統合されていない
- `segment_analysis.py` では `iaf_series` を外部から受け取る設計
- IAF変動係数（安定性指標）が総合スコアで **12.5%** の重みを持つにも関わらず、計算が非統一

**推奨方針**: IAF計算を `statistical_dataframe.py` に統合し、自動生成

---

## 改善の方向性

### 理想的なレイヤー構造

```
┌─────────────────────────────────────────────────────────────┐
│ Application Layer (応用層)                                   │
│ - セグメント分析、セッション総合評価、レポート生成            │
│ - ファイル: segment_analysis.py, session_summary.py          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│ Visualization Layer (可視化層) ★分離対象                     │
│ - グラフ生成、プロット設定                                   │
│ - ディレクトリ: lib/visualization/                           │
│   ├── eeg_plots.py      (Raw波形、PSD、スペクトログラム)      │
│   ├── metric_plots.py   (Fmθ、FAA、SE)                      │
│   └── segment_plots.py  (セグメント分析)                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│ Analysis Layer (計算層) ★計算ロジックのみ                    │
│ - 指標計算、統計処理                                         │
│ - ファイル:                                                  │
│   ├── frontal_theta.py         (Fmθ計算)                    │
│   ├── frontal_asymmetry.py     (FAA計算)                    │
│   ├── spectral_entropy.py      (SE計算)                     │
│   └── individual_alpha_freq.py (IAF計算) ★新規              │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│ Data Layer (データ層) ★統一的データ生成                      │
│ - MNE RawArray、DataFrame、統計データ生成                    │
│ - ファイル:                                                  │
│   ├── statistical_dataframe.py (統一バンドパワー計算)        │
│   └── loaders/mind_monitor.py  (CSV読み込み)                │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│ Foundation Layer (基盤層)                                    │
│ - MNE前処理、PSD計算、定数定義                               │
│ - ファイル:                                                  │
│   ├── preprocessing.py (MNE RawArray準備)                   │
│   ├── frequency.py     (PSD・スペクトログラム)               │
│   └── constants.py     (定数定義)                            │
└─────────────────────────────────────────────────────────────┘
```

### 設計原則

1. **単一責任原則（SRP）**: 各モジュールは1つの責務のみ
2. **依存性逆転原則（DIP）**: 上位層が下位層に依存、逆はNG
3. **開放閉鎖原則（OCP）**: 拡張に開き、修正に閉じる
4. **テスト容易性**: 各層で独立したユニットテストが可能

---

## 具体的なアクションプラン

### Phase 1: 計算と可視化の完全分離 【優先度: 高】✅ **完了**

**期間**: 1日（2025-11-16完了）
**目的**: SRP違反を解消し、テスト容易性を向上

#### 実装した構造

```
lib/
├── sensors/eeg/
│   ├── visualization/              # 🆕 EEG可視化
│   │   ├── __init__.py
│   │   ├── eeg_plots.py           # 基本EEG可視化（旧visualization.py）
│   │   ├── frontal_theta_plot.py  # Fmθ可視化
│   │   ├── frontal_asymmetry_plot.py # FAA可視化
│   │   └── spectral_entropy_plot.py  # SE可視化
│   ├── frontal_theta.py           # 計算のみ
│   ├── frontal_asymmetry.py       # 計算のみ
│   └── spectral_entropy.py        # 計算のみ
│
└── visualization/                  # 🆕 統合可視化（旧visualization.pyをディレクトリ化）
    ├── __init__.py
    ├── segment_plot.py            # セグメント比較
    ├── fnirs.py                   # fNIRS可視化
    ├── respiratory.py             # 呼吸可視化
    └── dashboard.py               # 統合ダッシュボード
```

#### 完了したタスク

- [x] **1.1** `lib/visualization/` ディレクトリ作成
  - `lib/sensors/eeg/visualization/` を作成
  - `lib/visualization/` をディレクトリ化（既存ファイルを分割）

- [x] **1.2** プロット関数の移動
  - `frontal_theta.plot_frontal_theta()` → `visualization/frontal_theta_plot.py`
  - `frontal_asymmetry.plot_frontal_asymmetry()` → `visualization/frontal_asymmetry_plot.py`
  - `spectral_entropy.plot_spectral_entropy()` → `visualization/spectral_entropy_plot.py`
  - `segment_analysis.plot_segment_comparison()` → `visualization/segment_plot.py`
  - `sensors/eeg/visualization.py` の関数群 → `visualization/eeg_plots.py`
  - `lib/visualization.py` → `fnirs.py`, `respiratory.py`, `dashboard.py` に分割

- [x] **1.3** import パスの更新
  - `lib/__init__.py` から可視化関数のエクスポート削除
  - `lib/sensors/eeg/__init__.py` から可視化関数のエクスポート削除
  - `lib/sensors/__init__.py` から可視化関数のエクスポート削除
  - `scripts/generate_report.py` のimport更新
  - **Breaking Change**: 後方互換性なし（即座にBreaking Change方式を採用）

- [x] **1.4** 計算モジュールのクリーンアップ
  - `frontal_theta.py` から `plot_frontal_theta()` 削除
  - `frontal_asymmetry.py` から `plot_frontal_asymmetry()` 削除
  - `spectral_entropy.py` から `plot_spectral_entropy()` 削除
  - `segment_analysis.py` から `plot_segment_comparison()` 削除
  - `lib/sensors/eeg/visualization.py` ファイル削除
  - `lib/visualization.py` ファイル削除

#### 達成された効果
- ✅ matplotlib 依存が可視化層のみに限定された
- ✅ 計算ロジックのユニットテストが容易になった
- ✅ 可視化のカスタマイズが独立して可能になった
- ✅ 単一責任原則（SRP）の遵守
- ✅ 全importテストが成功

#### Breaking Changes

**可視化関数の新しいimportパス:**
```python
# ❌ 旧（動作しない）
from lib import plot_frontal_theta, plot_psd, plot_segment_comparison

# ✅ 新（正しい）
from lib.sensors.eeg.visualization import plot_frontal_theta, plot_psd
from lib.visualization import plot_segment_comparison
```

---

### Phase 2: データ層の統一 【優先度: 高】✅ **完了**

**期間**: 1日（2025-11-16完了）
**目的**: IAF計算を `statistical_dataframe.py` に統合し、データ層の一貫性を完成

#### 完了したタスク

- [x] **2.1** IAF計算を `statistical_dataframe.py` に統合
  - ✅ Epochsごとにアルファ帯域（8-13Hz）のピーク周波数を計算
  - ✅ IAF時系列を`pd.Series`として追加
  - ✅ IAF統計量（平均・中央値・標準偏差・変動係数）を追加
  - ✅ 戻り値に `'iaf': Series` を追加
  - ✅ Z-score外れ値除去を適用

- [x] **2.2** `segment_analysis.py` の簡略化
  - ✅ `calculate_segment_analysis()` の `iaf_series` パラメータを削除
  - ✅ `statistical_df['iaf']` から自動取得するように変更
  - ✅ バリデーションに `'iaf'` キーを追加
  - ✅ docstring更新

- [x] **2.3** `generate_report.py` の簡略化
  - ✅ PAF時間推移からのIAF計算ロジックを削除（約30行削減）
  - ✅ セグメント分析呼び出しから `iaf_series` 引数を削除
  - ✅ IAF統計を `statistical_df` から直接取得
  - ✅ 総合スコア計算のIAF変動係数取得を簡略化

- [x] **2.4** テストファイルの作成
  - ✅ `tests/test_statistical_dataframe_iaf.py` を作成
  - ✅ IAF統合の包括的なテストケースを実装
  - ✅ 動作確認完了（実データでテスト成功）

#### 達成された効果
- ✅ IAF計算箇所: 3箇所 → 1箇所（`statistical_dataframe.py`）
- ✅ `generate_report.py`: 約30行削減
- ✅ `segment_analysis.py`: パラメータ1個削減
- ✅ バンドパワー計算の一貫性が保証される
- ✅ 外れ値除去・統計処理が全解析で適用される
- ✅ IAF計算が自動化され、セグメント分析が簡潔に

#### 新しいデータ生成フロー
```python
# 統一的な使い方
from lib import load_mind_monitor_csv, prepare_mne_raw, create_statistical_dataframe

# 1. CSV読み込み
df_clean = load_mind_monitor_csv(csv_path, warmup_minutes=1.0)

# 2. MNE RawArray準備
mne_result = prepare_mne_raw(df_clean)

# 3. 統一的バンドパワー・比率・IAF生成
statistical_df = create_statistical_dataframe(
    raw=mne_result['raw'],
    segment_minutes=5,
    warmup_minutes=1.0
)
# → statistical_df = {
#     'band_powers': DataFrame,       # Bels単位
#     'band_ratios': DataFrame,       # Bels差分 + 実数値
#     'spectral_entropy': DataFrame,  # 正規化済み
#     'iaf': Series,                  # Hz単位（新規追加）
#     'statistics': DataFrame         # 全統計量
# }
```

---

### Phase 3: スケーリング単位の統一 【優先度: 中】

**期間**: 3-5日
**目的**: 全指標を Bels に統一し、物理的意味を明確化

#### タスク

- [ ] **3.1** Fmθ のBels変換
  - `frontal_theta.py` の計算結果を μV² → Bels に変換
  - `10 * np.log10(fmtheta_power_uv2 + 1e-12)` を適用
  - 既存の統計量（前半/後半比較、増加率）も Bels ベースに

- [ ] **3.2** FAA のBels差分変換
  - 現在: `FAA = ln(右) - ln(左)`
  - 変更後: `FAA = 10*log10(右) - 10*log10(左)` (Bels差分)
  - 解釈基準の更新（0.5 Bels ≈ 12%差）

- [ ] **3.3** 総合スコア正規化範囲の見直し
  ```python
  # segment_analysis.py の _normalize_indicator() を更新

  # Fmθ正規化（Bels単位）
  if fmtheta is not None:
      # 旧: 50-200 μV²
      # 新: 17-23 Bels (10*log10(50) ~ 10*log10(200))
      scores['fmtheta'] = _normalize_indicator(fmtheta, min_val=17.0, max_val=23.0)

  # FAA正規化（Bels差分）
  if faa is not None:
      # 旧: -0.5 ~ 0.5 (ln)
      # 新: -2.0 ~ 2.0 (Bels差分)
      scores['faa'] = _normalize_indicator(faa, min_val=-2.0, max_val=2.0)
  ```

- [ ] **3.4** テストデータでの検証
  - 既存のCSVデータで新旧スケール比較
  - 総合スコアの変化を確認
  - ドキュメント更新（単位の解釈）

#### 期待される効果
- 全指標で物理的意味が統一（3 Bels = 2倍、10 Bels = 10倍）
- 総合スコア計算の透明性向上
- 異なるセッション間の比較が容易

#### 注意事項
- **影響範囲が大きい**: 既存の解析結果との互換性が失われる
- **段階的移行推奨**: 新旧両方の値を出力し、検証期間を設ける

---

### Phase 4: テスト強化 【優先度: 中】

**期間**: 継続的
**目的**: 各層で独立したテストを可能にし、リグレッションを防止

#### タスク

- [ ] **4.1** ユニットテストの作成
  ```
  tests/
  ├── test_statistical_dataframe.py  # データ層
  ├── test_frontal_theta.py          # 計算層
  ├── test_frontal_asymmetry.py      # 計算層
  ├── test_segment_analysis.py       # 応用層
  └── test_visualization.py          # 可視化層（画像生成確認）
  ```

- [ ] **4.2** サンプルデータの準備
  - 小規模なテスト用CSVデータ（1分程度）
  - 既知の結果を持つリファレンスデータ

- [ ] **4.3** CI/CDへの統合（オプション）
  - GitHub Actions などで自動テスト実行
  - 各Phaseのコード変更時にリグレッションテスト

#### 期待される効果
- リファクタリング時の安全性向上
- バグ早期発見
- コード品質の継続的向上

---

### Phase 5: ドキュメント整備 【優先度: 低】

**期間**: 1-2日
**目的**: 新しいアーキテクチャを明文化

#### タスク

- [ ] **5.1** アーキテクチャドキュメント作成
  - `docs/ARCHITECTURE.md`
  - レイヤー構造の説明
  - データフロー図
  - 各モジュールの責務

- [ ] **5.2** API リファレンス更新
  - `docs/API.md`
  - 主要関数のシグネチャと使用例
  - 単位・スケールの説明

- [ ] **5.3** マイグレーションガイド
  - `docs/MIGRATION.md`
  - 旧コードから新コードへの移行手順
  - Breaking Changes のリスト

---

## 期待される効果

### 定量的効果

| 指標 | 現状 | 改善後 |
|------|------|--------|
| バンドパワー計算の重複 | 3箇所 | 1箇所（`statistical_dataframe.py`） |
| 可視化関数の分散 | 5ファイル | 1ディレクトリ（`lib/visualization/`） |
| スケーリング単位の種類 | 4種類 | 2種類（Bels、正規化値） |
| 計算と可視化の混在モジュール | 4ファイル | 0ファイル |

### 定性的効果

1. **保守性向上**
   - 各層で独立した修正・テストが可能
   - バグの影響範囲が限定される

2. **再利用性向上**
   - 計算ロジックを他のプロジェクトで再利用可能
   - 可視化カスタマイズが容易

3. **可読性向上**
   - ファイル構造が明確
   - 責務が明確で、新規開発者のオンボーディングが容易

4. **テスト容易性向上**
   - matplotlib 依存が可視化層のみ
   - 各層でモックやスタブを使いやすい

---

## リスクと対策

### リスク 1: 既存コードへの影響
**影響度**: 高
**対策**:
- Phase 1-2 では後方互換性を維持（deprecation warning）
- 段階的移行期間を設ける（1-2ヶ月）
- リグレッションテストで既存機能の動作確認

### リスク 2: スケーリング変更による解析結果の変化
**影響度**: 中
**対策**:
- Phase 3 を独立したブランチで実施
- 新旧両方の値を出力し、検証期間を設ける
- ドキュメントで変更理由と影響を明記

### リスク 3: 工数の見積もり誤差
**影響度**: 中
**対策**:
- Phase 単位で進捗確認
- Phase 1-2 を優先し、Phase 3-5 は後回しも可

---

## 次のステップ

1. **Phase 1 開始**: 計算と可視化の分離（最優先）
   - `lib/visualization/` ディレクトリ作成
   - プロット関数の移動

2. **Phase 2 開始**: データ層の統一
   - IAF計算の統合
   - バンドパワー計算の統一

3. **Phase 3 検討**: スケーリング単位の統一（慎重に）
   - テストデータでの検証
   - 影響範囲の確認

4. **継続的改善**: テスト・ドキュメント整備

---

## 参考資料

- `/home/tsu-nera/repo/satoru/lib/segment_analysis.py` - セグメント分析実装
- `/home/tsu-nera/repo/satoru/lib/statistical_dataframe.py` - 統一的バンドパワー計算
- `/home/tsu-nera/repo/satoru/lib/sensors/eeg/preprocessing.py` - MNE前処理
- `/home/tsu-nera/repo/satoru/docs/AI-CODING.md` - プロジェクトガイドライン

---

**最終更新**: 2025-11-16
