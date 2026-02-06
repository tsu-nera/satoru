# Alpha Wave - HRV Correlation Analysis Summary

**Analysis Date:** 2026-01-24
**Datasets:** 3 sessions (2026-01-23, 2026-01-22, 2026-01-17)
**Window:** 1-minute intervals

---

## Results Overview

| Session    | Metric | Correlation | P-value | Significant |
|------------|--------|-------------|---------|-------------|
| 2026-01-23 | RMSSD  | +0.206      | 0.283   | No          |
| 2026-01-23 | SDNN   | +0.048      | 0.805   | No          |
| 2026-01-23 | pNN50  | +0.130      | 0.502   | No          |
| 2026-01-22 | RMSSD  | -0.044      | 0.824   | No          |
| 2026-01-22 | SDNN   | +0.014      | 0.943   | No          |
| 2026-01-22 | pNN50  | +0.033      | 0.870   | No          |
| 2026-01-17 | RMSSD  | -0.171      | 0.358   | No          |
| 2026-01-17 | SDNN   | +0.011      | 0.952   | No          |
| 2026-01-17 | pNN50  | -0.223      | 0.228   | No          |

---

## Key Findings

### 1. No Statistically Significant Correlations
- All p-values > 0.05 (significance threshold)
- No HRV metric showed consistent correlation with alpha wave activity across sessions

### 2. Correlation Direction Inconsistency
- **RMSSD**: Mixed results (+0.206, -0.044, -0.171)
- **SDNN**: Consistently near zero (+0.048, +0.014, +0.011)
- **pNN50**: Mixed results (+0.130, +0.033, -0.223)

### 3. Weakest Relationship
- **SDNN** showed the most consistent lack of relationship (r ≈ 0)
- Suggests no systematic linear relationship between alpha waves and overall HRV

---

## Interpretation

アルファ波とHRVの間に**明確な線形相関は認められない**という結果が、3つのセッション全てで一貫して得られました。

これは以下のいずれかを示唆している可能性があります：

1. **独立性**: アルファ波活動とHRVは独立した生理現象である
2. **非線形関係**: 線形相関では捉えられない複雑な関係性が存在する
3. **時間遅延**: 即時的な相関ではなく、時間遅延を伴う関係性がある可能性
4. **個人差**: セッション間でパターンが異なり、個人内でも一貫性がない

---

## Methodology

- **Alpha Wave**: 4チャンネル平均 (TP9, AF7, AF8, TP10)
- **HRV Metrics**:
  - RMSSD: Root Mean Square of Successive Differences
  - SDNN: Standard Deviation of NN intervals
  - pNN50: Percentage of successive RR intervals differing by >50ms
- **Time Window**: 1分間隔でリサンプリング
- **Statistical Test**: Pearson correlation coefficient

---

## Files Generated

- `correlation_summary.csv`: 詳細な統計データ
- `alpha_hrv_correlations.png`: 散布図と回帰直線の可視化
