# MuseIO OSC 仕様書

Interaxon公式SDK（MuseIO）のOSC仕様。2018年11月5日時点のアーカイブに基づく。

**出典**: [MuseIO Available Data (Wayback Machine)](https://web.archive.org/web/20181105231756/http://developer.choosemuse.com/tools/available-data)

## 概要

MuseIOはInteraxonが提供していた公式SDKツールで、Museヘッドバンドからのデータをネットワーク越しにOSCプロトコルで配信する。このフォーマットはMuse Lab、Muse Direct、Mind Monitorなど多くのツールで採用されている。

## EEG データ

### Raw EEG

生のEEGデータ（マイクロボルト単位）。

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/eeg` |
| 単位 | μV |
| データ型 | floats |
| 範囲 | 0.0 - 1682.815 μV |

**プリセット別設定**:

| プリセットID | 分解能 | サンプリングレート | チャンネル |
|-------------|--------|-------------------|-----------|
| 10, 12, 14 | 10 bit | 220 Hz | TP9, Fp1, Fp2, TP10 |
| AD | 16 bit | 500 Hz | TP9, Fp1, Fp2, TP10 |
| AE | 16 bit | 500 Hz | TP9, Fp1, Fp2, TP10, DRL, REF |

### EEG Quantization Level

圧縮時の量子化レベル。値が大きいほど精度が低下。

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/eeg/quantization` |
| データ型 | int |
| 範囲 | 1, 2, 4, 8, 16, 32, 64, 128 |

### Dropped EEG Samples

Bluetooth接続の問題で欠落したサンプル数。

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/eeg/dropped_samples` |
| データ型 | int |
| 範囲 | 0 - 65535 |

## 加速度計データ

### Raw Accelerometer

3軸加速度データ（milli-g単位）。

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/acc` |
| 単位 | milli-g |
| データ型 | 3 floats |
| 範囲 | -2000.0 - 1996.1 milli-g |
| サンプリングレート | 50 Hz |
| チャンネル | ACC_X, ACC_Y, ACC_Z |

### Dropped Accelerometer Samples

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/acc/dropped_samples` |
| データ型 | int |
| 範囲 | 0 - 65535 |

## Muse Elements（信号処理済みデータ）

生のEEGから計算された高レベルな特徴量。

### Raw FFT

各チャンネルのパワースペクトル密度。0-110Hzを129ビンに分割。

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/elements/raw_fft0` 〜 `/muse/elements/raw_fft3` |
| データ型 | 129 floats |
| 単位 | dB |
| 送信レート | 10 Hz |
| 範囲 | 約 -40.0 〜 20.0 |

**FFT設定**:
- ウィンドウサイズ: 256サンプル（Hammingウィンドウ）
- オーバーラップ: 90%（22サンプルずつスライド）
- 周波数分解能: 220/256 ≈ 0.86 Hz/bin

### Absolute Band Powers

周波数帯域ごとの絶対パワー（対数スケール、Bels単位）。

| 項目 | 値 |
|------|-----|
| 送信レート | 10 Hz |
| データ型 | 4 floats（各チャンネル） |
| 単位 | Bels (B) |

**OSCパスと周波数帯域**:

| OSCパス | 名称 | 周波数範囲 |
|--------|------|-----------|
| `/muse/elements/low_freqs_absolute` | Low Freqs | 2.5-6.1 Hz |
| `/muse/elements/delta_absolute` | Delta | 1-4 Hz |
| `/muse/elements/theta_absolute` | Theta | 4-8 Hz |
| `/muse/elements/alpha_absolute` | Alpha | 7.5-13 Hz |
| `/muse/elements/beta_absolute` | Beta | 13-30 Hz |
| `/muse/elements/gamma_absolute` | Gamma | 30-44 Hz |

### Relative Band Powers

全帯域に対する相対パワー（0-1の範囲）。

計算式:
```
alpha_relative = 10^alpha_absolute / (10^alpha_absolute + 10^beta_absolute + 10^delta_absolute + 10^gamma_absolute + 10^theta_absolute)
```

| OSCパス | 周波数範囲 |
|--------|-----------|
| `/muse/elements/delta_relative` | 1-4 Hz |
| `/muse/elements/theta_relative` | 4-8 Hz |
| `/muse/elements/alpha_relative` | 7.5-13 Hz |
| `/muse/elements/beta_relative` | 13-30 Hz |
| `/muse/elements/gamma_relative` | 30-44 Hz |

### Band Power Session Scores

セッション中のパワー分布に基づくスコア（0-1）。10-90パーセンタイルを基準に線形正規化。

| OSCパス | 周波数範囲 |
|--------|-----------|
| `/muse/elements/delta_session_score` | 1-4 Hz |
| `/muse/elements/theta_session_score` | 4-8 Hz |
| `/muse/elements/alpha_session_score` | 7.5-13 Hz |
| `/muse/elements/beta_session_score` | 13-30 Hz |
| `/muse/elements/gamma_session_score` | 30-44 Hz |

## ヘッドバンド装着状態

### Touching Forehead

額に接触しているかどうか。

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/elements/touching_forehead` |
| データ型 | int (boolean) |
| 送信レート | 10 Hz |

### Horseshoe（装着品質インジケータ）

各チャンネルの装着品質。

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/elements/horseshoe` |
| データ型 | 4 floats |
| 送信レート | 10 Hz |
| 値 | 1=Good, 2=OK, ≥3=Bad |

### Strict Indicator

厳密な装着品質インジケータ。

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/elements/is_good` |
| データ型 | 4 ints |
| 送信レート | 10 Hz |
| 値 | 0=Bad, 1=Good |

## 筋電アーチファクト検出

### Blink（瞬き検出）

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/elements/blink` |
| データ型 | int (boolean) |
| 送信レート | 10 Hz |

### Jaw Clench（歯の食いしばり検出）

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/elements/jaw_clench` |
| データ型 | int (boolean) |
| 送信レート | 10 Hz |

## 実験的指標

**警告**: これらの値は実験的であり、将来のリリースで変更または削除される可能性がある。

### Concentration（集中度）

ガンマ波を主成分とする集中度スコア。

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/elements/experimental/concentration` |
| データ型 | float |
| 範囲 | 0.0 - 1.0 |
| 送信レート | 10 Hz |

注意: 筋電アーチファクトで偽陽性が発生しやすい。

### Mellow（リラックス度）

アルファ波を主成分とするリラクゼーションスコア。

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/elements/experimental/mellow` |
| データ型 | float |
| 範囲 | 0.0 - 1.0 |
| 送信レート | 10 Hz |

## バッテリーデータ

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/batt` |
| データ型 | 4 ints |
| 送信レート | 0.1 Hz（10秒ごと） |

**データ形式**: `[残量, Fuel Gauge電圧, ADC電圧, 温度]`

| フィールド | 単位 | 範囲 |
|-----------|------|------|
| 残量 | %/100 | 0 - 10000 |
| Fuel Gauge電圧 | mV | 3000 - 4200 |
| ADC電圧 | mV | 3200 - 4200 |
| 温度 | ℃ | -40 - +125 |

## DRL/REF データ

Driven-Right-Leg回路によるコモンモードノイズ除去用の基準電極データ。

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/drlref` |
| データ型 | 2 floats |
| データ形式 | [DRL電圧, REF電圧] |
| 単位 | μV |
| 範囲 | 0 - 3,300,000 |
| サンプリングレート | EEGと同一 |

## バージョン・アノテーション

### Version

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/version` |
| 送信レート | 0.1 Hz（10秒ごと） |
| データ形式 | JSON文字列 |

### Annotation

| 項目 | 値 |
|------|-----|
| OSCパス | `/muse/annotation` |
| データ形式 | 5つの文字列フィールド |

フィールド: `[イベントデータ, フォーマット, イベントタイプ, イベントID, 親ID]`

## 設定情報

MuseIOは接続時に設定情報をJSON形式で出力する。

**主要な設定項目**:

| カテゴリ | 設定項目 |
|---------|---------|
| グローバル | `mac_addr`, `serial_number`, `preset` |
| EEG | `eeg_sample_frequency_hz`, `eeg_output_frequency_hz`, `eeg_channel_layout`, `filters_enabled`, `notch_frequency_hz` |
| 加速度計 | `acc_data_enabled`, `acc_sample_frequency_hz`, `acc_units` |
| DRL/REF | `drlref_data_enabled`, `drlref_sample_frequency_hz` |
| バッテリー | `battery_data_enabled`, `battery_percent_remaining` |

## 互換性

このOSC仕様に準拠するツール:
- **Muse Lab** - Interaxon公式ビューア
- **Muse Direct** - Interaxon公式ストリーミングアプリ（iOS）
- **Mind Monitor** - サードパーティアプリ（iOS/Android）

**注意**: Muse公式瞑想アプリ（Muse: Meditation & Sleep）は異なるOSC形式を使用。詳細は [MUSE_APP_OSC_OUTPUT.md](MUSE_APP_OSC_OUTPUT.md) を参照。
