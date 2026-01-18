# Resonance Frequency HRV Analysis Report

**分析日時**: 2026-01-18 10:04:09

## 1. 最適な共鳴周波数（Optimal Resonance Frequency）

| 指標 | 値 |
|:-----|---:|
| **呼吸レート (Breathing Rate)** | 4.5 breaths/min |
| **吸気時間 (Inhale)** | 4.4 sec |
| **呼気時間 (Exhale)** | 8.9 sec |
| **LF Power** | 1675.28 ms² |
| **RMSSD** | 20.60 ms |
| **HR Mean** | 67.92 bpm |
| **HR Max-Min** | 16.77 bpm |

## 2. 全トライアル結果

| Trial | Rate | Inhale | Exhale | RMSSD | LF Power | HR Mean | HR Max-Min | Samples |
|------:|-----:|-------:|-------:|------:|---------:|--------:|-----------:|--------:|
| 1 | 6.5 | 3.1 | 6.2 | 28.23 | 575.83 | 73.90 | 23.12 | 148 |
| 2 | 6.0 | 3.3 | 6.7 | 23.82 | 625.54 | 71.19 | 19.79 | 142 |
| 3 | 5.5 | 3.6 | 7.3 | 24.29 | 756.44 | 69.81 | 22.40 | 138 |
| 4 | 5.0 | 4.0 | 8.0 | 20.70 | 1268.40 | 67.60 | 14.91 | 134 |
| 5 | 4.5 | 4.4 | 8.9 | 20.60 | 1675.28 | 67.92 | 16.77 | 135 |
| 6 | 6.5 | 3.1 | 6.2 | 20.03 | 756.35 | 71.70 | 25.09 | 142 |

## 3. 測定順序効果の分析

- **測定順序とLF Powerの相関係数**: 0.560

⚠️ **警告**: 測定順序とLF Powerに中程度以上の相関があります

測定条件が時間経過とともに変化している可能性があります:
- ウォーミングアップ効果
- 疲労
- 環境要因の変化

**推奨事項**:
1. 測定順序をランダム化して再測定
2. 各測定間に十分な休憩（2-3分）を挿入
3. 同じ時間帯・環境で測定

## 4. 再現性の評価

同じ呼吸レートでの複数測定:

### 6.5 bpm

| Trial | LF Power | RMSSD | HR Mean |
|------:|---------:|------:|--------:|
| 1 | 575.83 | 28.23 | 73.90 |
| 6 | 756.35 | 20.03 | 71.70 |

- **LF Power平均**: 666.09 ms²
- **標準偏差**: 90.26 ms²
- **変動係数 (CV)**: 13.55%


## 5. 推奨事項

### 次のステップ

1. **さらに低い呼吸レートを測定**: 現在の最低レート（4.5 bpm）でLF Powerが最大ですが、
   さらに低いレート（例: 4.0 bpm, 3.5 bpm）でピークがある可能性があります。

2. **測定の再現性確認**: 同じ呼吸レートで複数回測定し、結果の安定性を確認してください。

3. **測定プロトコルの改善**:
   - 測定順序をランダム化
   - 各測定間に2-3分の休憩
   - 測定前に10分間の安静時間
   - 同じ時間帯・環境で測定

## 参考文献

1. **Lehrer, P., & Gevirtz, R. (2014).**
   Heart rate variability biofeedback: how and why does it work?
   *Frontiers in psychology*, 5, 756.

2. **Steffen, P. R., et al. (2017).**
   A Practical Guide to Resonance Frequency Assessment for Heart Rate Variability Biofeedback.
   *Frontiers in Neuroscience*, 14, 570400.
   https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.570400/full
