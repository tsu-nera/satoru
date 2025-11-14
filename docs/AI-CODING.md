# AI Coding Guidelines

このドキュメントは、Codex CLI、Claude Code、およびその他のAIコーディングツールが共通で参照するプロジェクトガイドラインです。

---

## プロジェクト概要

Muse Headband（脳波測定デバイス）から取得したデータの解析プロジェクトです。

---

## Python 仮想環境

### 環境名
- `venv`

### セットアップ手順
```bash
# 環境の作成
python3 -m venv venv

# 有効化 (WSL/Linux)
source venv/bin/activate

# パッケージインストール
pip install -r requirements.txt
```

### インストール済みパッケージ
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `jupyterlab`
- `scipy`

---

## 開発環境

### Jupyter Lab
分析用ノートブックは以下で起動します：
```bash
jupyter lab
```

### WSL2 環境
Google Drive (Windows側 `G:`) をWSL2から扱う場合：
```bash
sudo mount -t drvfs G: /mnt/g
```
マウント後は `/mnt/g/マイドライブ/` 以下のデータを参照できます。

---

## データ仕様

### Mind Monitor CSV Data

Mind Monitor (Muse Headband用アプリ) のCSVデータを扱う場合は、以下のドキュメントを参照してください：

**仕様書**: [docs/MIND_MONITOR_CSV_SPECIFICATION.md](MIND_MONITOR_CSV_SPECIFICATION.md)

#### 内部ドキュメント
- [docs/MIND_MONITOR_CSV_SPECIFICATION.md](MIND_MONITOR_CSV_SPECIFICATION.md)
- [docs/MUSE_EEG_SPEC.md](MUSE_EEG_SPEC.md)

#### 含まれる情報
- 全カラムの詳細仕様（単位、範囲、説明）
- センサー位置と脳波周波数帯域の説明
- データ品質管理の方法
- サンプリングレートと欠損値の扱い

---

## コーディング規約

### Python コードスタイル
- PEP 8に準拠
- 関数・クラスには必ずdocstringを記述
- 型ヒントを推奨

### ファイル構成
```
satoru/
├── data/              # データファイル
│   └── samples/       # サンプルCSVデータ
├── docs/              # ドキュメント
├── notebooks/         # Jupyter notebooks
├── scripts/           # Python スクリプト
└── venv/              # Python 仮想環境
```

### 命名規則
- スクリプト名: `snake_case.py`
- クラス名: `PascalCase`
- 関数・変数名: `snake_case`

### DataFrameのカラム名規約

**すべての統計情報DataFrameは英語カラム名を使用すること**

#### 基本構造
統計情報を格納するDataFrameは以下の標準形式を使用：
```python
pd.DataFrame([
    {'Metric': 'Mean', 'Value': float, 'Unit': 'μV²'},
    {'Metric': 'Median', 'Value': float, 'Unit': 'μV²'},
    {'Metric': 'Std Dev', 'Value': float, 'Unit': 'μV²'},
])
```

#### カラム名
- **禁止**: `'指標'`, `'値'`, `'単位'`, `'平均値'`, `'中央値'` などの日本語カラム名
- **推奨**: `'Metric'`, `'Value'`, `'Unit'`, `'Mean'`, `'Median'` などの英語カラム名

#### メトリック名
| 日本語（旧） | 英語（新） |
|------------|-----------|
| '平均' | 'Mean' |
| '中央値' | 'Median' |
| '標準偏差' | 'Std Dev' |
| '前半平均' | 'First Half Mean' |
| '後半平均' | 'Second Half Mean' |
| '変化率 (後半/前半)' | 'Change Rate (2nd/1st)' |

#### 理由
- コードの一貫性向上
- 国際化対応の容易化
- Pandas/scikit-learnエコシステムとの親和性
- 将来的なAPI公開やデータ共有を考慮

---

## コミットメッセージ

### フォーマット
```
<type>: <subject>

<body>
```

### Type
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメントのみの変更
- `refactor`: リファクタリング
- `test`: テスト追加・修正
- `chore`: ビルド、設定変更

---

## 実装方針メモ

- EEG処理や信号解析でMNE-Pythonに相当機能が存在する場合は、自前実装よりもMNEの公式APIを優先して利用する。
- 既存コードに独自アルゴリズムが残っている場合も、検証を行いながら段階的にMNE-Pythonの機能へ置き換えることを検討する。
- レポート用の図版・画像出力ではラベルや注記に英語を用い、国際的な再利用性を確保する（必要に応じて本文で日本語解説を補足する）。

### 可視化・画像出力のガイドライン

- **すべての画像出力（グラフ、チャート等）では英語を使用する**
  - タイトル、軸ラベル、凡例、注釈などすべて英語で記述
  - 例: "周波数 (Hz)" → "Frequency (Hz)"、"パワースペクトル密度" → "Power Spectral Density"
- **日本語フォント設定は使用しない**
  - GitHub Actionsなどの環境で日本語フォントが利用できないため
  - matplotlibのデフォルトフォント（英語対応）を使用
- **理由**: 国際的な再利用性の確保、CI/CD環境での互換性維持

---

## issues/ ディレクトリの取り扱い

### 基本方針

`issues/` ディレクトリは各 issue の作業領域であり、**明示的な指示がない限りリファクタリングや修正を行わない**こと。

### ルール

1. **読み取り専用として扱う**
   - コード参照、仕様確認、構造理解のための読み取りは可
   - 修正・リファクタリングは明示的に指示された場合のみ実施

2. **修正が許可される場合**
   - ユーザーが明示的に「issues/XXX を修正して」と指示した場合
   - プロジェクト全体に影響する変更（例: 仮想環境名の変更）でユーザーの承認がある場合
   - セキュリティ上の緊急対応が必要な場合

3. **修正が禁止される例**
   - コードスタイルの統一
   - 変数名のリネーム
   - コメントの追加・修正
   - ファイル構造の変更
   - 依存関係の更新

### 理由

- 各 issue は進行中の作業であり、意図しない変更により作業が混乱する可能性がある
- issue 固有のコンテキストや判断基準が存在する場合がある
- 過去の issue は歴史的な記録として保持する価値がある

---

## 参考資料

### 外部ドキュメント
- [Mind Monitor FAQ](https://mind-monitor.com/FAQ.php)
- [Muse Developer Docs](https://choosemuse.com/pages/developers)
