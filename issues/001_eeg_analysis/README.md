# Muse脳波データ基本分析

このディレクトリには、Museヘッドバンドで取得した脳波データの基本分析を実行し、マークダウンレポートを自動生成するスクリプトが含まれています。

## 概要

`generate_report.py` は、Jupyter notebookで行っていた分析をスクリプト化し、以下を実行します：

1. **データ読み込みと前処理**: Mind Monitor CSVデータの読み込み
2. **周波数バンド統計**: Delta, Theta, Alpha, Beta, Gammaの基本統計
3. **バンドパワー時系列**: 各周波数バンドの時間推移
4. **パワースペクトル密度（PSD）**: 周波数成分の詳細分析
5. **スペクトログラム**: 時間-周波数解析
6. **バンド比率分析**: リラックス度、集中度、瞑想深度の算出
7. **Peak Alpha Frequency（PAF）分析**: Alpha波のピーク周波数と時間変化

## ディレクトリ構成

```
001_basic_analysis/
├── generate_report.py     # 分析スクリプト（メイン）
├── REPORT.md              # 生成されたマークダウンレポート
├── README.md              # このファイル
├── img/                   # 生成された画像ディレクトリ
│   ├── band_power_time_series.png
│   ├── band_ratios.png
│   ├── paf.png
│   ├── paf_time_evolution.png
│   ├── psd.png
│   └── spectrogram.png
└── notes/                 # 過去の分析ノート（参考用）
    └── mind_monitor_basic_analysis.ipynb

../../data/                # 入力データディレクトリ（プロジェクトルート）
└── *.csv                 # Mind Monitor CSVファイル
```

## 使い方

### 1. 仮想環境の準備

```bash
# プロジェクトルートから
source venv/bin/activate
```

### 2. 必要なパッケージのインストール

スクリプトは以下のパッケージを使用します（通常はすでにインストール済み）：

```bash
pip install pandas numpy matplotlib mne scipy tabulate
```

### 3. スクリプトの実行

```bash
# 基本的な使用方法
python generate_report.py --data <CSVファイルパス>

# 例：プロジェクトルートのdataディレクトリから読み込み
python generate_report.py --data ../../data/mindMonitor_2025-10-15--17-08-15_2188947289907039573.csv

# 出力先を指定する場合
python generate_report.py --data <CSVファイルパス> --output <出力ディレクトリ>
```

### 4. 生成される成果物

- **REPORT.md**: 分析結果をまとめたマークダウンレポート
- **img/*.png**: レポートで参照される各種グラフ画像

## コマンドライン引数

- `--data`: **必須** - 入力CSVファイルのパス
- `--output`: オプション - 出力ディレクトリ（デフォルト: スクリプトと同じディレクトリ）

## レポートの内容

生成される `REPORT.md` には以下のセクションが含まれます：

1. **データ概要**: ファイル情報、記録時間、計測時間
2. **周波数バンド統計**: 各バンドの平均値、標準偏差、有効データ率
3. **バンドパワーの時間推移**: 全バンドの時系列プロット
4. **パワースペクトル密度（PSD）**: 周波数成分の詳細、各バンドのピーク周波数
5. **スペクトログラム**: 時間-周波数のヒートマップ
6. **脳波指標（バンド比率）**:
   - リラックス度（α/β比）
   - 集中度（β/θ比）
   - 瞑想深度（θ/α比）
   - セッション評価
   - データ品質（スパイク分析）
7. **Peak Alpha Frequency（PAF）分析**:
   - チャネル別PAF
   - Individual Alpha Frequency（IAF）
   - PAFの時間推移と安定性評価
8. **まとめ**: 分析内容のサマリーと次のステップ

## 技術スタック

- **Python 3.12**
- **pandas**: データ処理
- **numpy**: 数値計算
- **matplotlib**: グラフ描画
- **MNE-Python**: 脳波信号処理（PSD、スペクトログラム）
- **scipy**: 統計解析

## データ品質に関する注意

スクリプトは以下のデータクリーニングを自動実行します：

- タイムスタンプの日時型変換
- 欠損値の除外とソート
- ゼロ値の除外（脳波パワー解析時）
- 重複タイムスタンプの平均化
- 時間補間による欠損値補完
- DCドリフト軽減のためのハイパスフィルタ（1 Hz以上）

スパイク分析により、外れ値や変動係数を評価し、データ品質を可視化します。

## トラブルシューティング

### 日本語フォントが表示されない

グラフの日本語が正しく表示されない場合：

```bash
# Noto Sans CJK フォントをインストール（Ubuntu/WSL）
sudo apt install fonts-noto-cjk
```

### メモリ不足エラー

大きなCSVファイルを処理する際にメモリ不足になる場合：

1. `low_memory=False` オプションはすでに設定済み
2. 必要に応じてデータを分割して処理
3. スペクトログラム計算の周波数分解能を下げる（`freqs_tfr` の間隔を調整）

## 参考資料

- **Mind Monitor CSV仕様**: [../../docs/MIND_MONITOR_CSV_SPECIFICATION.md](../../docs/MIND_MONITOR_CSV_SPECIFICATION.md)
- **プロジェクトガイドライン**: [../../docs/AI-CODING.md](../../docs/AI-CODING.md)
- **過去の分析ノート**: [notes/mind_monitor_basic_analysis.ipynb](notes/mind_monitor_basic_analysis.ipynb)

## 今後の拡張

このスクリプトをベースに、以下の専門分析スクリプトを追加予定：

- **左右半球差分析**: 左右半球の非対称性指数（AI）
- **サマタ瞑想のfmシータ波分析**: 前頭部シータ波の詳細解析
- **統合生理指標分析**: 心拍変動（HRV）との統合解析

## ライセンス

このプロジェクトはTitan2プロジェクトの一部です。

---

**最終更新**: 2025-10-30
**バージョン**: 1.0
