# Multi-Session Theta Peak Analysis

**Analysis Date**: 2026-02-06 08:41:56
**Number of Sessions**: 10
**Date Range**: 2026-01-11--07-16-50 ~ 2026-02-06--07-42-47
**Theta Range**: 5.0 - 7.0 Hz

---

## Overall Statistics
IA
### Individual Theta Frequency (ITF)

- **Mean ITF**: 5.92 ± 0.30 Hz
- **Coefficient of Variation (CV)**: 5.1%
- **Range**: 5.38 - 6.31 Hz
- **Median**: 5.96 Hz

**安定性評価**: **安定** (CV < 10%)

### Peak Detection Statistics

- **Frontal (AF7/AF8) Detection Rate**: 70.0%
- **TP9 Detection Rate**: 60.0%
- **AF7 Detection Rate**: 60.0%
- **AF8 Detection Rate**: 40.0%
- **TP10 Detection Rate**: 40.0%

### Channel-wise ITF Statistics

| Channel   |   Mean (Hz) |   Std (Hz) |   CV (%) |   N Sessions |
|:----------|------------:|-----------:|---------:|-------------:|
| TP9       |        6.00 |       0.22 |     3.73 |            6 |
| AF7       |        5.88 |       0.34 |     5.87 |            6 |
| AF8       |        5.94 |       0.47 |     7.97 |            4 |
| TP10      |        5.94 |       0.24 |     4.03 |            4 |

## Connection Quality Analysis

### Correlation: Peak Frequency vs Quality

- **TP9**: r = 0.048, p = 0.928 (Not significant)
- **AF7**: r = -0.264, p = 0.613 (Not significant)
- **AF8**: r = -0.343, p = 0.657 (Not significant)
- **TP10**: r = 0.171, p = 0.829 (Not significant)

### Correlation: Peak Power vs Quality

- **TP9**: r = -0.494, p = 0.319 (Not significant)
- **AF7**: r = -0.185, p = 0.725 (Not significant)
- **AF8**: r = 0.561, p = 0.439 (Not significant)
- **TP10**: r = -0.111, p = 0.889 (Not significant)


## Session Details

| Date                 |   Duration (min) |   Avg Quality |   Channels w/ Peak |   Mean Freq (Hz) |   Std Freq (Hz) | Frontal Peak   |
|:---------------------|-----------------:|--------------:|-------------------:|-----------------:|----------------:|:---------------|
| 2026-01-11--07-16-50 |            29.96 |          1.01 |                  0 |           nan    |          nan    | False          |
| 2026-01-12--06-21-12 |            30.45 |          1.01 |                  2 |             5.38 |            0.12 | True           |
| 2026-01-14--07-29-14 |            29.67 |          1.00 |                  2 |             5.75 |            0.00 | False          |
| 2026-01-15--07-28-00 |            29.19 |          1.00 |                  4 |             6.31 |            0.11 | True           |
| 2026-01-16--07-18-47 |            29.90 |          1.05 |                  3 |             5.92 |            0.12 | True           |
| 2026-01-17--07-04-54 |            29.49 |          1.20 |                  0 |           nan    |          nan    | False          |
| 2026-01-22--07-42-10 |            26.94 |          1.01 |                  3 |             6.00 |            0.00 | True           |
| 2026-01-23--07-40-44 |            28.17 |          1.00 |                  3 |             5.75 |            0.00 | True           |
| 2026-01-29--06-21-09 |            30.16 |          1.01 |                  1 |             6.25 |          nan    | True           |
| 2026-02-06--07-42-47 |            32.00 |          1.01 |                  2 |             6.00 |            0.25 | True           |

## Interpretation

### ⚠️ Moderately Reproducible

前頭部シータピークが70%のセッションで検出されました。ある程度の再現性はありますが、接続品質やセッション状態に依存する可能性があります。

### ✅ Stable Individual Theta Frequency

あなたの個人シータ周波数(ITF)は **5.92 ± 0.30 Hz** で、変動係数5.1%と非常に安定しています。これは**個人特性として信頼できる指標**です。

### Quality Independence Assessment

複数セッションの分析から、シータピーク周波数は接続品質に大きく依存していないことが確認できれば、アーチファクトではないと結論できます。上記の相関分析を参照してください。

---

## Figures

1. [Session Comparison](session_comparison.png)
2. [ITF Stability Analysis](itf_stability.png)
