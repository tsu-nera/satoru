# Multimodal 1/f Noise Analysis Report

## Overview

This report analyzes the 1/f^β power spectral characteristics across all Muse sensors:

- **EEG** (4ch): Brain electrical activity
- **IMU** (6ch): Body sway (Accelerometer, Gyroscope)
- **Optics (fNIRS)** (4ch): Cerebral blood flow (HbO/HbR)
- **ECG (Selfloops)** (1ch): Heart rate variability (RR intervals)

## All Sensors Summary

| Sensor          | Channel      |     β |    R² | Freq Range   |
|:----------------|:-------------|------:|------:|:-------------|
| EEG             | RAW_TP9      | 0.886 | 0.517 | 1-40 Hz      |
| EEG             | RAW_AF7      | 1.508 | 0.844 | 1-40 Hz      |
| EEG             | RAW_AF8      | 1.622 | 0.842 | 1-40 Hz      |
| EEG             | RAW_TP10     | 0.938 | 0.694 | 1-40 Hz      |
| IMU (Accel)     | Accel_X      | 1.960 | 0.862 | 0.1-10 Hz    |
| IMU (Accel)     | Accel_Y      | 1.968 | 0.826 | 0.1-10 Hz    |
| IMU (Accel)     | Accel_Z      | 1.946 | 0.790 | 0.1-10 Hz    |
| IMU (Gyro)      | Gyro_X       | 1.956 | 0.831 | 0.1-10 Hz    |
| IMU (Gyro)      | Gyro_Y       | 1.978 | 0.788 | 0.1-10 Hz    |
| IMU (Gyro)      | Gyro_Z       | 1.999 | 0.773 | 0.1-10 Hz    |
| Optics (fNIRS)  | HbO_Left     | 1.739 | 0.462 | 0.01-1 Hz    |
| Optics (fNIRS)  | HbO_Right    | 1.844 | 0.522 | 0.01-1 Hz    |
| Optics (fNIRS)  | HbR_Left     | 1.509 | 0.398 | 0.01-1 Hz    |
| Optics (fNIRS)  | HbR_Right    | 1.968 | 0.648 | 0.01-1 Hz    |
| ECG (Selfloops) | RR Intervals | 1.298 | 0.799 | 0.01-0.4 Hz  |

## Sensor-wise Statistics

### EEG

- **Mean β**: 1.239 ± 0.380
- **Channels**: 4
- **Interpretation**: 標準的な1/f^β範囲から外れています。

### IMU (Accel)

- **Mean β**: 1.958 ± 0.011
- **Channels**: 3
- **Interpretation**: ブラウンノイズ（1/f²）に近く、より強い自己相関を持ちます。

### IMU (Gyro)

- **Mean β**: 1.977 ± 0.022
- **Channels**: 3
- **Interpretation**: ブラウンノイズ（1/f²）に近く、より強い自己相関を持ちます。

### Optics (fNIRS)

- **Mean β**: 1.765 ± 0.195
- **Channels**: 4
- **Interpretation**: ブラウンノイズ（1/f²）に近く、より強い自己相関を持ちます。

### ECG (Selfloops)

- **Mean β**: 1.298 ± nan
- **Channels**: 1
- **Interpretation**: 標準的な1/f^β範囲から外れています。

## Visualization

![Multimodal 1/f Analysis](img/multimodal_1f.png)

**Color coding**: Green (R² > 0.8), Orange (R² > 0.6), Red (R² ≤ 0.6)

## Overall Interpretation

### β値の比較（センサー別平均）

- **EEG**: β = 1.239
- **ECG (Selfloops)**: β = 1.298
- **Optics (fNIRS)**: β = 1.765
- **IMU (Accel)**: β = 1.958
- **IMU (Gyro)**: β = 1.977

### 考察

1. **EEG（脳波）**: 高周波領域（1-40Hz）での1/f特性。覚醒時はβ≈2.0が標準。
2. **IMU（体軸）**: 身体動揺の周波数特性。姿勢制御の安定性を反映。
3. **Optics (fNIRS)（脳血流）**: 血流変動の周波数特性。心拍成分も含む生理的ゆらぎ。
4. **ECG（心拍変動）**: 自律神経系のバランス。ピンクノイズ（β≈1）が健康的。

