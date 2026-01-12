# ECG呼吸分析

ECGのR-R間隔データから呼吸系指標を計算する分析ディレクトリ

## 概要

NeuroKit2のECG-Derived Respiration (EDR)機能を使用して、心拍変動データから呼吸数や呼吸パターンを推定します。

## ファイル構成

```
011_br_analysis/
├── analyze_breathing.py          # 呼吸分析メインスクリプト（NeuroKit2使用）
├── BREATHING_REPORT.md            # 分析結果レポート
├── breathing_analysis.png         # 可視化プロット
└── README.md                      # このファイル
```

## 使用方法

### 基本的な使い方

```bash
# プロジェクトルートから実行
cd /home/tsu-nera/repo/satoru
source venv/bin/activate
python issues/011_br_analysis/analyze_breathing.py
```

デフォルトでは以下のデータを使用：
- 入力: `data/selfloops/selfloops_2026-01-12--06-21-05.csv`
- 出力: `issues/011_br_analysis/`

### カスタマイズ

スクリプト内の以下の行を編集してファイルパスを変更できます：

```python
# analyze_breathing.py の最終行付近
if __name__ == '__main__':
    csv_path = 'data/selfloops/your_file.csv'  # 変更
    output_dir = Path(__file__).parent

    analyze_breathing_neurokit2(csv_path, output_dir)
```

## 分析内容

### 1. R-R間隔の前処理
- SelfLoops HRVデータの読み込み
- R-R間隔のクリーニング
- 等間隔リサンプリング（8 Hz）

### 2. ECG-Derived Respiration (EDR)
- `nk.ecg_rsp()`: 心拍数信号から呼吸成分を抽出
- バンドパスフィルタ（0.1-0.4 Hz = 6-24 bpm）

### 3. 呼吸ピーク検出
- `nk.rsp_clean()`: 信号のクリーニング
- `nk.rsp_findpeaks()`: ピークとトラフの自動検出

### 4. 呼吸数計算
- `nk.rsp_rate()`: 瞬時呼吸数の計算
- スペクトル解析: Welch法によるパワースペクトル密度

### 5. 出力
- **BREATHING_REPORT.md**: 呼吸数、検出ピーク数などの統計
- **breathing_analysis.png**: 4つのプロット
  1. 心拍数信号
  2. EDR信号とピーク検出
  3. 瞬時呼吸数の時系列
  4. パワースペクトル

## 出力例

### 呼吸数
- スペクトル法: 9.4 bpm
- 平均（ピーク検出）: 11.1 ± 4.1 bpm
- 検出ピーク数: 351

### 評価
- 6-10 bpm: 深呼吸・瞑想状態
- 12-20 bpm: 正常範囲（安静時）
- 20+ bpm: 運動・緊張状態

## 技術スタック

### 主要ライブラリ
- **NeuroKit2**: ECG-Derived Respiration、呼吸ピーク検出
- **SciPy**: 信号補間、スペクトル解析
- **Matplotlib**: 可視化

### 使用する既存ライブラリ
- `lib.loaders.selfloops`: SelfLoops HRVデータローダー
- `lib.sensors.ecg`: (将来的にこの機能を統合予定)

## 参考資料

### NeuroKit2の参照論文
- van Gent et al. (2019): HeartPy アルゴリズム
- Charlton et al. (2016): EDR評価研究
- Sarkar et al. (2015): 呼吸抽出手法

## ライブラリへの統合予定

このスクリプトは将来的に以下のように統合予定：

```python
# lib/sensors/ecg/breathing.py として
from lib.sensors.ecg.breathing import analyze_breathing

result = analyze_breathing(
    hrv_data,
    method='neurokit2',  # 標準
    show=True
)
```

## 制約事項

### EDR法の限界
1. **精度**: 実際の呼吸センサーより精度は劣る
2. **個人差**: RSA（呼吸性洞性不整脈）の強度に個人差
3. **体動**: アーティファクトの影響を受ける可能性
4. **測定時間**: 正確な推定には最低2-3分必要

### 推奨事項
- 簡易的な呼吸パターンモニタリングに適している
- 正確な呼吸測定には専用センサー（胸部ベルト等）を推奨
- ECGと呼吸センサーの併用でRSA詳細分析が可能

---

作成日: 2026-01-12
