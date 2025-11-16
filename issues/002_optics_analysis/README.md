# Muse光学データ統合分析

このディレクトリは、Muse Sヘッドバンドの光学センサー（fNIRS/PPG）データを解析し、マークダウンレポートと可視化画像を自動生成するためのツール群です。  
deprecatedなノートブック（`@_deprecated/notes/integrated_physiological_analysis.ipynb`）で行っていた処理をスクリプト化し、再現性のあるワークフローとして整理しました。

## 概要

`generate_report.py` は Mind Monitor CSV を入力として以下の分析を実施します。

1. **データ読み込みとサマリー**: 計測時間・サンプリングレート・チャネル数などを算出
2. **fNIRS解析** (`lib/sensors/fnirs.py`): HbO/HbRの推定、左右半球の統計値算出
3. **PPG解析** (`lib/sensors/ppg.py`): RRインターバル、呼吸数（Welch/FFT）の推定
4. **可視化** (`lib/visualization.py`): Museアプリ風グラフ、HRVスペクトル、統合ダッシュボードの作成
5. **Markdownレポート生成**: グラフと統計テーブルをまとめた `REPORT.md` を出力

## ディレクトリ構成

```
002_optics_analysis/
├── generate_report.py   # メイン分析スクリプト
├── REPORT.md            # 生成されたレポート（サンプル）
├── README.md            # このファイル
└── img/                 # 生成された画像
    ├── fnirs_panel.png
    ├── fnirs_muse_style.png
    ├── respiratory_overview.png
    ├── respiratory_psd.png
    └── integrated_dashboard.png
```

## 使い方

### 1. 仮想環境を有効化

```bash
source venv/bin/activate
```

### 2. レポート生成

```bash
python generate_report.py --data ../../data/<MindMonitor CSV>
```

オプション:
- `--output <DIR>`: レポートと画像の出力先を指定（デフォルトはスクリプトディレクトリ）
- `--no-font`: matplotlib の日本語フォント設定をスキップ

### 3. 実行結果

- `REPORT.md`: Markdown形式の分析レポート
- `img/*.png`: レポートで使用する可視化グラフ

## 分析項目

- **fNIRS統計**: 左右半球HbO/HbRの平均・標準偏差・範囲
- **Museアプリ風グラフ**: HbO/HbRの時系列を単一図にまとめて表示
- **心拍 & 呼吸**: RRインターバル推定、呼吸数推定（Welch/FFT）
- **統合ダッシュボード**: fNIRSと心拍指標を同一キャンバスで確認可能

## 参考

- リファクタ済みライブラリ: `lib/sensors/fnirs.py`, `lib/sensors/ppg.py`, `lib/visualization.py`
- 旧ノートブック: `@_deprecated/notes/integrated_physiological_analysis.ipynb`

---

**最終更新**: 2025-10-30
