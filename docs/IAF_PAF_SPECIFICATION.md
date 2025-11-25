# IAF / PAF 仕様メモ

## 用語

- **PAF (Peak Alpha Frequency)**: Alpha帯域内のピーク周波数
- **CoG (Center of Gravity)**: Alpha帯域のパワー加重平均周波数
- **IAF (Individual Alpha Frequency)**: 個人のAlpha周波数（本実装ではPAFを採用）

## 実装パラメータ

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| Alpha帯域 | 8-12 Hz | Muse標準。Mind Monitorは7.5-13Hz |
| n_fft | 2048 | 周波数解像度 約0.125Hz (@256Hz sampling) |
| 半球平均 | Left=(TP9+AF7)/2, Right=(AF8+TP10)/2 | ノイズ低減 |
| デフォルトIAF | Peak方式 | Muse app互換 |

## 計算方法

### Peak方式 (PAF)
```
PAF = argmax(PSD[alpha_band])
```
Alpha帯域内で最大パワーを持つ周波数。

### Center of Gravity方式 (CoG)
```
CoG = Σ(f × P(f)) / Σ(P(f))
```
周波数をパワーで加重平均。より安定だが、ノイズの影響で高めにシフトする傾向。

### 比較結果（実データ）

| 方式 | IAF | Muse appとの差 |
|------|-----|---------------|
| Peak | 8.75 Hz | 0.15-0.35 Hz |
| CoG | 9.44 Hz | 0.84-1.04 Hz |
| Muse app | 8.4-8.6 Hz | - |

→ **Peak方式がMuse appに近い**

## 周波数解像度

```
周波数解像度 = サンプリングレート / n_fft
```

| n_fft | 解像度 (256Hz時) |
|-------|-----------------|
| 512 | 0.50 Hz |
| 1024 | 0.25 Hz |
| 2048 | 0.125 Hz |

## 使用例

```python
from lib.sensors.eeg.paf import calculate_paf

paf_result = calculate_paf(psd_dict)

# 両方の値にアクセス可能
print(f"IAF (Peak): {paf_result['iaf_peak']:.2f} Hz")
print(f"IAF (CoG):  {paf_result['iaf_cog']:.2f} Hz")
print(f"IAF (推奨): {paf_result['iaf']:.2f} Hz")  # Peak方式
```

## 参考情報

### Mind Monitor フォーラム (James, 作者)

> Alpha is the sum of all power within the frequency range of 7.5Hz to 13Hz.
> If you wanted to calculate where the high point is over a large data set,
> you would need to set the recording interval to Constant to get all the RAW EEG data at 256Hz,
> then perform an FFT on the entire data set.

## 参考リンク

- [Mind Monitor Technical Manual](https://mind-monitor.com/Technical_Manual.php)
- [Bandpower of an EEG signal (Raphael Vallat)](https://raphaelvallat.com/bandpower.html)
- [Toward a reliable, automated method of IAF quantification (PubMed)](https://pubmed.ncbi.nlm.nih.gov/29357113/)
- [Validating the wearable MUSE headset (bioRxiv)](https://www.biorxiv.org/content/10.1101/2021.11.02.466989v1.full)
