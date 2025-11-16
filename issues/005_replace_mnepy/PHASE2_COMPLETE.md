# Phase 2 完了レポート: MNE Epochs導入

**Issue**: #005_replace_mnepy
**完了日**: 2025-11-14
**担当**: Claude Code

---

## 📋 概要

Phase 2では、時間セグメント分析のバンドパワー計算にMNE Epochsを導入し、計算の高精度化とコードの標準化を実現しました。

---

## ✅ 完了したタスク

### 1. MNE Epochsによるバンドパワー計算関数の実装

**ファイル**: `lib/segment_analysis.py`

**新規追加関数**:
```python
def _calculate_band_power_from_epochs(
    raw: 'mne.io.RawArray',
    segment_minutes: int,
    session_start: pd.Timestamp,
    warmup_minutes: float = 0.0,
) -> Dict[str, pd.Series]:
    """
    MNE Epochsを使用してセグメントごとのバンドパワーを計算する。

    - 固定長イベント作成（mne.make_fixed_length_events）
    - Epochsオブジェクト作成
    - Welch法によるPSD計算
    - バンド別パワー抽出（Theta: 4-8Hz, Alpha: 8-13Hz, Beta: 13-30Hz）
    - Bels変換（10*log10）
    """
```

**主な改善点**:
- ✅ ベクトル化されたPSD計算（全エポックを一度に処理）
- ✅ Nyquist周波数を考慮した安全なfmax自動調整
- ✅ 低サンプリングレートデータ（52.74Hz）でもエラーなく動作

### 2. `calculate_segment_analysis`関数の統合

**変更内容**:
- 新しいパラメータ追加：
  - `raw: Optional[mne.io.RawArray]`
  - `use_mne_epochs: bool = True`
- 条件分岐による計算パス選択：
  - **MNE Epochsパス**: `use_mne_epochs=True` かつ `raw` が渡された場合
  - **レガシーパス**: 既存のDataFrameベース計算（後方互換性）

**ループ簡素化**:
```python
# 旧: セグメント境界を手動生成してループ内で平均計算
for start in segment_starts:
    window = series.loc[(series.index >= start) & (series.index < end)]
    alpha_mean = window.mean()  # ×N回

# 新: MNE Epochsが事前にセグメント化済み
alpha_mean = band_series['Alpha'].iloc[idx - 1]  # 直接取得
```

### 3. `generate_report.py`の更新

**変更内容**:
```python
segment_result = calculate_segment_analysis(
    df_quality,
    fmtheta_result.time_series,
    segment_minutes=5,
    iaf_series=iaf_series,
    warmup_minutes=1.0,
    raw=raw,  # ← 追加
    use_mne_epochs=True,  # ← 追加
)
```

### 4. テストと動作確認

**テストデータ**: `data/mindMonitor_2025-11-03--06-55-17_993618302911552438.csv`
- サンプリングレート: 52.74 Hz（低レート）
- 記録時間: 20.6分
- セグメント数: 3セグメント（5分×3）

**結果**: ✅ 正常動作
- Nyquist周波数（26.37Hz）を考慮したfmax自動調整が機能
- セグメント分析が正常に完了
- レポート生成成功

---

## 📊 成果

### 改善点

| 項目 | 改善前 | 改善後 |
|------|--------|--------|
| **バンドパワー計算** | DataFrameの列平均 | MNE EpochsのPSD計算（Welch法） |
| **計算精度** | 低（時間ドメインの平均） | 高（周波数ドメインの正確なPSD） |
| **セグメント化** | 手動ループ | MNE Epochs自動化 |
| **コード可読性** | 複雑なループロジック | 簡潔な関数呼び出し |
| **標準化** | 独自実装 | MNE-Python標準手法 |
| **エラーハンドリング** | なし | Nyquist周波数チェック |

### コード変更統計

```
lib/segment_analysis.py    | +169 -27 lines
scripts/generate_report.py | +2 lines
```

**注**: 現時点では新関数追加とレガシーパス保持のためコード行数は増加。将来的にレガシーパスを削除すれば削減効果が顕著になる見込み。

---

## 🎯 技術的ポイント

### 1. MNE Epochsの活用

**固定長イベント作成**:
```python
duration_sec = segment_minutes * 60.0
events = make_fixed_length_events(raw_cropped, duration=duration_sec)
```

**Epochsオブジェクト**:
```python
epochs = Epochs(raw_cropped, events, tmin=0, tmax=duration_sec,
                baseline=None, preload=True, verbose=False)
```

**PSD計算（ベクトル化）**:
```python
spectrum = epochs.compute_psd(method='welch', fmin=1.0, fmax=fmax, verbose=False)
psds, freqs = spectrum.get_data(return_freqs=True)
# psds.shape: (n_epochs, n_channels, n_freqs)
```

### 2. Nyquist周波数の安全処理

```python
sfreq = raw_cropped.info['sfreq']
nyquist = sfreq / 2.0
fmax = min(50.0, nyquist * 0.95)  # 安全マージン5%
```

これにより、低サンプリングレートデータ（Mind Monitorの不均一録音）でもエラーなく動作。

### 3. 後方互換性の維持

- `use_mne_epochs=False` で既存の動作を保持
- `band_means` パラメータも引き続きサポート
- 既存のレポートフォーマットは一切変更なし

---

## 🔍 今後の展望

### Phase 3以降での改善案

1. **レガシーパスの段階的廃止**
   - `use_mne_epochs=False` パスの削除
   - コード行数: 推定 -50行程度

2. **マルチモーダル対応**（別Issue）
   - fNIRS: Pandas Resampleで統合
   - 心拍数: 同上
   - EEGのみMNE Epochsで高精度計算

3. **統計解析の強化**
   - MNEの統計関数活用（`mne.stats.permutation_cluster_test`など）
   - セグメント間比較の統計的有意性検定

---

## ⚠️ 注意事項

### 制限事項

1. **fNIRS/心拍数は未対応**
   - MNE-Pythonは脳波（EEG/MEG）専門
   - マルチモーダル対応は別Issueで検討

2. **現時点でのコード削減効果は限定的**
   - 新関数追加: +100行
   - レガシーパス保持: +70行
   - ループ簡素化: -30行程度
   - **純増**: +140行

   しかし、**メンテナンス性と標準化**の観点では大幅な改善。

### トラブルシューティング

**問題**: `Requested fmax must not exceed ½ the sampling frequency`

**解決策**: すでに実装済み（Nyquist周波数自動調整）
```python
fmax = min(50.0, nyquist * 0.95)
```

---

## 📚 参考リソース

### MNE-Python公式ドキュメント

- [Epoching data](https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html)
- [make_fixed_length_events](https://mne.tools/stable/generated/mne.make_fixed_length_events.html)
- [compute_psd](https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.compute_psd)

### プロジェクト内参照

- [PLAN.md](./PLAN.md) - 全体計画
- [PHASE1_COMPLETE.md](./PHASE1_COMPLETE.md) - Phase 1完了レポート

---

## ✅ 完了承認

- [x] MNE Epochs実装完了
- [x] 後方互換性確認
- [x] テストデータ動作確認
- [x] ドキュメント更新
- [x] PLAN.md更新

**Phase 2 正式完了** 🎉
