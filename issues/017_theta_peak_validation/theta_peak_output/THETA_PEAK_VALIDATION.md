# Theta Peak Validation Analysis

**Data File**: `mindMonitor_2026-02-06--07-42-47_4574839398217372417.csv`
**Analysis Date**: 2026-02-06 08:33:41
**Theta Range**: 5.0 - 7.0 Hz

---

## Summary Statistics

### Overall Peak Detection Rate

- **Total Windows**: 63
- **Windows with Peak**: 63
- **Detection Rate**: 100.0%

### Detection Rate by Connection Quality

| quality_category   |   Peaks Detected |   Total |   Detection Rate |
|:-------------------|-----------------:|--------:|-----------------:|
| Good               |              232 |     252 |             92.1 |

### Channel-wise Statistics

| channel   |   theta_peak_freq_mean |   theta_peak_freq_std |   theta_peak_freq_min |   theta_peak_freq_max |   theta_peak_power_mean |   theta_peak_power_std |   theta_peak_prominence_mean |   theta_peak_prominence_std |
|:----------|-----------------------:|----------------------:|----------------------:|----------------------:|------------------------:|-----------------------:|-----------------------------:|----------------------------:|
| AF7       |                   5.94 |                  0.5  |                     5 |                  7    |                    3.2  |                   6.11 |                         4.41 |                        3.09 |
| AF8       |                   5.78 |                  0.6  |                     5 |                  7    |                   -3.24 |                   5.24 |                         2.68 |                        1.25 |
| TP10      |                   5.87 |                  0.55 |                     5 |                  6.75 |                    4.24 |                   3.59 |                         3.15 |                        2.26 |
| TP9       |                   5.93 |                  0.5  |                     5 |                  7    |                    9.24 |                   7.1  |                         6.66 |                        5.15 |

## Correlation Analysis

### Peak Frequency vs Connection Quality

- **TP9**: r = -0.102, p = 0.439 (Not significant)
- **AF7**: r = -0.038, p = 0.778 (Not significant)
- **AF8**: r = 0.252, p = 0.056 (Not significant)
- **TP10**: r = 0.048, p = 0.726 (Not significant)

### Peak Power vs Connection Quality

- **TP9**: r = 0.239, p = 0.066 (Not significant)
- **AF7**: r = 0.624, p = 0.000 (**Significant**)
- **AF8**: r = 0.673, p = 0.000 (**Significant**)
- **TP10**: r = 0.476, p = 0.000 (**Significant**)


## Spatial Distribution (Full Session)

| Channel | Peak Freq (Hz) | Peak Power (dB) | Prominence (dB) |
|---------|----------------|-----------------|------------------|
| TP9 | 6.25 | 13.17 | 4.80 |
| AF7 | 5.75 | 8.59 | 1.57 |
| AF8 | No peak | - | - |
| TP10 | No peak | - | - |


## Interpretation

### ✅ Frontal Theta Peak Detected

前頭部(AF7/AF8)でシータピークが検出されました。これは**Frontal Midline Theta (FMT)**の特徴と一致し、瞑想状態における生理学的な現象を示唆します。

### ✅ Quality-Independent Detection

接続品質に関わらず比較的安定してピークが検出されています。これは生理学的な現象である可能性を支持します。

### ✅ Physiologically Valid Frequency Range

平均ピーク周波数 5.88 ± 0.54 Hz は、瞑想研究で報告されているシータ帯域範囲内です。

---

## Figures

1. [Connection Quality vs Theta Peak](quality_vs_theta_peak.png)
2. [Spatial Distribution](spatial_distribution.png)
3. [Temporal Stability](temporal_stability.png)
