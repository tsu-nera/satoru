# Data Engineering: レポート生成システムの改善

## 目的
データ分析プロジェクトの観点から、Raw Data管理、レポートテンプレート化、設定管理を改善し、保守性と拡張性を向上させる。

---

## 現状の問題点

### 1. Raw Dataの配置と管理

**問題**:
- `data/`ディレクトリにCSVファイルが永続化（97MB、197MB）
- `.gitignore`でCSVは除外されているが、ディスク容量を圧迫
- どのCSVがどのレポートに対応するか追跡困難
- `run_analysis.sh`が最新CSVを自動選択（意図しないファイルを使う可能性）
- 削除タイミングが不明確（手動管理が必要）

**現在のファイル構造**:
```
satoru/
├── data/
│   ├── mindMonitor_2025-10-15--17-08-15_*.csv  (97MB)
│   └── mindMonitor_2025-10-31--16-04-01_*.csv  (197MB)
├── issues/001_eeg_analysis/
│   ├── run_analysis.sh          # data/から最新CSVを自動選択
│   └── REPORT.md
└── .gitignore                   # *.csv は除外済み
```

### 2. レポートテンプレートの扱い

**問題**:
- `generate_report.py`内でf-string連結によるレポート生成（64-215行目）
- レポート構造の変更がPythonコードに直結
- 多言語対応やカスタマイズが困難
- レイアウト変更時にPythonコードを編集する必要
- DRY原則違反（複数issueで同じテンプレートロジックを重複）

**現在のコード**:
```python
# generate_report.py (64-215行目)
def generate_markdown_report(data_path, output_dir, results):
    report = f"""# Muse脳波データ分析レポート

**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**データファイル**: `{data_path.name}`

---

## 1. データ概要
...
"""
    # 150行以上のf-string連結
```

### 3. 設定とパラメータの管理

**問題**:
- 閾値やパラメータがコード内にハードコード
- ユーザーごとのカスタマイズが困難
- 実験的なパラメータ変更のたびにコード修正が必要

**ハードコードされている値の例**:
```python
# generate_report.py (132-141行目)
if 'リラックス' in ratio_name:
    level = 'とても高い' if avg_value > 2.0 else '高い' if avg_value > 1.0 else '普通'
elif '集中' in ratio_name:
    level = 'とても高い' if avg_value > 2.0 else '高い' if avg_value > 1.0 else '普通'
elif '瞑想' in ratio_name:
    level = '深い' if avg_value > 1.5 else '中程度' if avg_value > 0.8 else '浅い'

# generate_report.py (176-183行目)
cv = stats['変動係数 (%)']
if cv < 2:
    stability = '非常に安定'
elif cv < 5:
    stability = '安定'
```

### 4. メタデータとトレーサビリティの欠如

**問題**:
- どのRaw CSVからどのレポートが生成されたか記録がない
- 過去セッションとの比較に必要な情報が散逸
- データ品質の履歴管理ができない
- 再現性の確保が困難

### 5. ディスク容量の無駄

**問題**:
- 約300MBのCSVファイルが累積（毎日瞑想すると月9GB）
- 実際に必要なのは統計サマリーのみ（数KB）
- バックアップやクローン時に不要なデータが含まれる

---

## 改善提案

### 提案1: Raw Data管理の改善（優先度：最高）

#### ディレクトリ構造の変更

```
titan2/
├── data/
│   ├── raw/                           # 一時的なRawデータ（分析後削除）
│   │   └── session_20251031_160401/
│   │       └── input.csv              # 分析中のみ存在
│   │
│   ├── processed/                     # 処理済み軽量データ（永続化）
│   │   └── session_20251031_160401/
│   │       ├── metadata.json          # ファイル情報、ハッシュ値
│   │       ├── summary_stats.json     # 統計サマリー（数KB）
│   │       └── key_metrics.parquet    # 時系列の重要指標（軽量）
│   │
│   └── archive/                       # オプション：長期保存用
│       └── session_20251031.csv.gz    # 圧縮版
```

#### DataManagerクラスの実装

**ファイル**: `lib/data_manager.py` (新規作成)

```python
from pathlib import Path
import shutil
import hashlib
import json
from datetime import datetime
import pandas as pd

class DataManager:
    """Raw CSVの一時管理と処理済みデータの永続化を担当"""

    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.raw_dir = self.project_root / 'data' / 'raw'
        self.processed_dir = self.project_root / 'data' / 'processed'
        self.archive_dir = self.project_root / 'data' / 'archive'

        # ディレクトリ作成
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def register_raw_data(self, csv_path):
        """
        Raw CSVを一時ディレクトリに登録

        Returns:
            tuple: (session_id, temp_csv_path)
        """
        session_id = self._generate_session_id(csv_path)
        session_dir = self.raw_dir / session_id
        session_dir.mkdir(exist_ok=True)

        dest_path = session_dir / 'input.csv'
        shutil.copy(csv_path, dest_path)

        print(f'✓ Raw data registered: {session_id}')
        return session_id, dest_path

    def save_processed_data(self, session_id, csv_path, df, analysis_results):
        """
        処理済みデータとメタデータを保存

        Parameters:
            session_id: セッションID
            csv_path: 元のCSVパス
            df: pandas DataFrame
            analysis_results: 分析結果の辞書
        """
        processed_session = self.processed_dir / session_id
        processed_session.mkdir(exist_ok=True)

        # メタデータ
        metadata = {
            'session_id': session_id,
            'original_file': csv_path.name,
            'file_size_mb': round(csv_path.stat().st_size / (1024*1024), 2),
            'file_hash': self._calculate_hash(csv_path),
            'processed_at': datetime.now().isoformat(),
            'data_info': {
                'shape': str(df.shape),
                'start_time': str(df['TimeStamp'].min()),
                'end_time': str(df['TimeStamp'].max()),
                'duration_sec': (df['TimeStamp'].max() - df['TimeStamp'].min()).total_seconds()
            }
        }

        with open(processed_session / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # 統計サマリー（軽量）
        summary = {
            'total_score': analysis_results.get('total_score'),
            'band_statistics': analysis_results.get('band_statistics', {}).to_dict()
                if 'band_statistics' in analysis_results else {},
            'band_ratios_stats': analysis_results.get('band_ratios_stats', {}).to_dict()
                if 'band_ratios_stats' in analysis_results else {},
            'paf_summary': analysis_results.get('paf_summary', {}).to_dict()
                if 'paf_summary' in analysis_results else {},
        }

        with open(processed_session / 'summary_stats.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # 重要な時系列データのみ保存（Parquet形式で圧縮）
        key_columns = ['TimeStamp', 'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
                      'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
                      'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']

        if all(col in df.columns for col in key_columns):
            df[key_columns].to_parquet(
                processed_session / 'key_metrics.parquet',
                compression='gzip',
                index=False
            )

        print(f'✓ Processed data saved: {processed_session}')

    def cleanup_raw_data(self, session_id, confirm=True):
        """
        Raw CSVを削除（分析完了後）

        Parameters:
            session_id: セッションID
            confirm: 確認メッセージを表示するか
        """
        session_dir = self.raw_dir / session_id

        if not session_dir.exists():
            print(f'⚠ Raw data not found: {session_id}')
            return

        if confirm:
            print(f'🗑 Deleting raw data: {session_dir}')

        shutil.rmtree(session_dir)
        print(f'✓ Raw data deleted: {session_id}')

    def archive_raw_data(self, session_id):
        """
        Raw CSVを圧縮してアーカイブ（オプション）
        """
        self.archive_dir.mkdir(exist_ok=True)
        session_dir = self.raw_dir / session_id

        if not session_dir.exists():
            print(f'⚠ Raw data not found: {session_id}')
            return

        import gzip
        raw_csv = session_dir / 'input.csv'
        archive_path = self.archive_dir / f'{session_id}.csv.gz'

        with open(raw_csv, 'rb') as f_in:
            with gzip.open(archive_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f'✓ Archived: {archive_path}')

        # アーカイブ後にrawを削除
        self.cleanup_raw_data(session_id, confirm=False)

    def load_historical_data(self, limit=30):
        """
        過去の処理済みデータを読み込み（トレンド分析用）

        Returns:
            list: 過去セッションのメタデータとサマリーのリスト
        """
        sessions = []

        for session_dir in sorted(self.processed_dir.iterdir(), reverse=True)[:limit]:
            if not session_dir.is_dir():
                continue

            metadata_path = session_dir / 'metadata.json'
            summary_path = session_dir / 'summary_stats.json'

            if metadata_path.exists() and summary_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                with open(summary_path) as f:
                    summary = json.load(f)

                sessions.append({
                    'session_id': session_dir.name,
                    'metadata': metadata,
                    'summary': summary
                })

        return sessions

    @staticmethod
    def _generate_session_id(csv_path):
        """CSVファイル名からセッションIDを生成"""
        # mindMonitor_2025-10-31--16-04-01_*.csv -> session_20251031_160401
        name = csv_path.stem
        if 'mindMonitor_' in name:
            date_part = name.split('mindMonitor_')[1].split('_')[0]
            # 2025-10-31--16-04-01 -> 20251031_160401
            date_clean = date_part.replace('-', '').replace('--', '_').replace('-', '')[:15]
            return f'session_{date_clean}'
        else:
            return f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    @staticmethod
    def _calculate_hash(file_path):
        """ファイルのSHA256ハッシュを計算（先頭16文字）"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()[:16]


def cleanup_all_raw_data(project_root, keep_latest=1):
    """
    すべてのRaw dataをクリーンアップ（手動実行用）

    Parameters:
        project_root: プロジェクトルート
        keep_latest: 最新N件は保持
    """
    dm = DataManager(project_root)
    sessions = sorted(dm.raw_dir.iterdir(), reverse=True)

    for i, session_dir in enumerate(sessions):
        if i < keep_latest:
            print(f'⏩ Keeping: {session_dir.name}')
            continue

        if session_dir.is_dir():
            dm.cleanup_raw_data(session_dir.name, confirm=False)
```

#### generate_report.pyの修正

```python
# generate_report.py の run_full_analysis() に統合

from lib.data_manager import DataManager

def run_full_analysis(data_path, output_dir):
    # DataManager初期化
    dm = DataManager(project_root)

    # Raw dataを一時ディレクトリに登録
    session_id, temp_csv_path = dm.register_raw_data(data_path)

    # データ読み込み（一時ファイルから）
    df = load_mind_monitor_csv(temp_csv_path, quality_filter=False)

    # ... 分析処理 ...

    # 処理済みデータを保存
    dm.save_processed_data(session_id, data_path, df, results)

    # レポート生成
    generate_markdown_report(data_path, output_dir, results)

    # Raw dataを削除
    dm.cleanup_raw_data(session_id)

    print(f'\n✓ Session: {session_id}')
    print(f'✓ Processed data: {dm.processed_dir / session_id}')
```

#### クリーンアップスクリプト

**ファイル**: `scripts/cleanup_raw_data.sh` (新規作成)

```bash
#!/bin/bash
# Raw dataを一括削除

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================="
echo "Raw Data Cleanup"
echo "========================================="
echo ""

RAW_DIR="$PROJECT_ROOT/data/raw"

if [ ! -d "$RAW_DIR" ]; then
    echo "✓ No raw data directory found"
    exit 0
fi

COUNT=$(find "$RAW_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

if [ "$COUNT" -eq 0 ]; then
    echo "✓ No raw data sessions found"
    exit 0
fi

echo "Found $COUNT raw data session(s)"
echo ""

read -p "Delete all raw data? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$RAW_DIR"/*
    echo "✓ All raw data deleted"
else
    echo "✗ Cancelled"
fi
```

---

### 提案2: テンプレートベースのレポート生成（優先度：高）

#### Jinja2テンプレートの導入

**必要な追加パッケージ**:
```txt
# requirements.txt に追加
jinja2>=3.1.0
```

**ディレクトリ構造**:
```
titan2/
├── templates/
│   ├── report_template.md           # 基本テンプレート
│   ├── report_template_en.md        # 英語版（将来）
│   ├── report_summary.md            # サマリー版
│   └── components/                  # 再利用可能なコンポーネント
│       ├── executive_summary.md
│       ├── band_statistics.md
│       └── data_quality.md
```

**テンプレートファイル**: `templates/report_template.md` (新規作成)

```jinja2
# Muse脳波データ分析レポート

**生成日時**: {{ generation_time }}
**データファイル**: `{{ data_file }}`
**セッションID**: {{ session_id }}

---

{% if executive_summary %}
## エグゼクティブサマリー

- **総合スコア**: {{ executive_summary.total_score }}/100 {% if executive_summary.score_change %}({{ executive_summary.score_change }}){% endif %}
- **セッション評価**: {{ executive_summary.evaluation }}
- **主要指標**:
  - Fmθ: {{ executive_summary.frontal_theta_level }}
  - Alpha安定性: {{ executive_summary.alpha_stability_level }}
  - 集中持続力: {{ executive_summary.attention_span_level }}

{% if executive_summary.dashboard_img %}
![サマリーダッシュボード](img/{{ executive_summary.dashboard_img }})
{% endif %}

---
{% endif %}

## 1. データ概要

- **データ形状**: {{ data_info.shape }}
- **記録時間**: {{ data_info.start_time }} ~ {{ data_info.end_time }}
- **計測時間**: {{ data_info.duration_sec|round(1) }} 秒

{% if time_segments %}
## 2. 時間セグメント分析

{{ time_segments.stats_table|safe }}

### 集中度の時間推移

![時間セグメント](img/{{ time_segments.plot_img }})

**ピークパフォーマンス**: {{ time_segments.peak_time_range }}

---
{% endif %}

## {% if time_segments %}3{% else %}2{% endif %}. 周波数バンド統計

{{ band_statistics|safe }}

**周波数バンドの説明**:
- **Delta (0.5-4 Hz)**: 深い睡眠
- **Theta (4-8 Hz)**: 瞑想、創造性
- **Alpha (8-13 Hz)**: リラックス、目を閉じた状態
- **Beta (13-30 Hz)**: 集中、活動
- **Gamma (30-50 Hz)**: 高度な認知処理

{% if band_power_img %}
## {% if time_segments %}4{% else %}3{% endif %}. バンドパワーの時間推移

![バンドパワー時系列](img/{{ band_power_img }})

Alpha波が高いとリラックス状態、Beta波が高いと集中状態を示します。
{% endif %}

{% if psd_img %}
## {% if time_segments %}5{% else %}4{% endif %}. パワースペクトル密度（PSD）

![PSD](img/{{ psd_img }})

{% if psd_peaks %}
### 各バンドのピーク周波数

{{ psd_peaks|safe }}
{% endif %}
{% endif %}

{% if spectrogram_img %}
## {% if time_segments %}6{% else %}5{% endif %}. スペクトログラム

![スペクトログラム](img/{{ spectrogram_img }})

時間とともに周波数分布がどう変化するかを可視化しています。
{% endif %}

{% if band_ratios_img %}
## {% if time_segments %}7{% else %}6{% endif %}. 脳波指標（バンド比率）

![バンド比率](img/{{ band_ratios_img }})

{% if band_ratios_stats %}
### 指標サマリー

{{ band_ratios_stats|safe }}

### セッション評価

{% for evaluation in session_evaluations %}
- **{{ evaluation.name }}**: {{ evaluation.value }} ({{ evaluation.level }})
{% endfor %}
{% endif %}

{% if spike_analysis %}
### データ品質（スパイク分析）

{{ spike_analysis|safe }}
{% endif %}
{% endif %}

{% if paf_img %}
## {% if time_segments %}8{% else %}7{% endif %}. Peak Alpha Frequency (PAF) 分析

![PAF](img/{{ paf_img }})

{% if paf_summary %}
### チャネル別PAF

{{ paf_summary|safe }}
{% endif %}

{% if iaf %}
**Individual Alpha Frequency (IAF)**: {{ iaf.value|round(2) }} ± {{ iaf.std|round(2) }} Hz
{% endif %}

{% if paf_time_img %}
### PAFの時間的変化

![PAF時間推移](img/{{ paf_time_img }})

{% if paf_time_stats %}
#### PAF統計

{% for key, value in paf_time_stats.items() %}
- **{{ key }}**: {{ value|round(2) }}
{% endfor %}

**PAF安定性評価**: {{ paf_stability }}
{% endif %}
{% endif %}
{% endif %}

{% if historical_comparison %}
## {% if time_segments %}9{% else %}8{% endif %}. 過去データとの比較

### トレンド

- **前日比**: {{ historical_comparison.yesterday_change }}
- **週平均**: {{ historical_comparison.week_average }}
- **ベストスコア**: {{ historical_comparison.best_score }} ({{ historical_comparison.best_date }})

![スコアトレンド](img/{{ historical_comparison.trend_img }})
{% endif %}

{% if action_advice %}
---

## アクションアドバイス

### 今回の強み

{% for strength in action_advice.strengths %}
- {{ strength }}
{% endfor %}

### 改善ポイント

{% for improvement in action_advice.improvements %}
- {{ improvement }}
{% endfor %}

### 次回への提案

{% for suggestion in action_advice.suggestions %}
- {{ suggestion }}
{% endfor %}
{% endif %}

---

## まとめ

このレポートでは、Museヘッドバンドで取得した脳波データの基本的な分析を行いました。

### 分析内容

1. **データ読み込み**: CSVデータの読み込みと前処理
2. **基本統計**: 各周波数バンドの特性を把握
3. **バンドパワー**: 時間経過に伴う各バンドの変化を追跡
4. **周波数解析**: PSDとスペクトログラムで周波数特性を詳細分析
5. **指標分析**: リラックス度、集中度、瞑想深度の数値化
6. **PAF分析**: Peak Alpha Frequencyの測定と時間変化の追跡

---

**生成スクリプト**: `generate_report.py`
**分析エンジン**: MNE-Python, pandas, matplotlib
**ライブラリ**: lib/eeg.py
**セッションID**: {{ session_id }}
**データハッシュ**: {{ data_hash }}
```

#### ReportGeneratorクラスの実装

**ファイル**: `lib/report_generator.py` (新規作成)

```python
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import pandas as pd

class ReportGenerator:
    """Jinja2テンプレートベースのレポート生成"""

    def __init__(self, template_dir=None):
        if template_dir is None:
            # プロジェクトルートからtemplatesディレクトリを探す
            template_dir = Path(__file__).parents[1] / 'templates'

        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # カスタムフィルタ登録
        self.env.filters['to_markdown'] = self._df_to_markdown

    def generate(self, template_name, output_path, context):
        """
        テンプレートからレポートを生成

        Parameters:
            template_name: テンプレートファイル名（例: 'report_template.md'）
            output_path: 出力ファイルパス
            context: テンプレートに渡すデータ（辞書）
        """
        template = self.env.get_template(template_name)

        # DataFrameをMarkdownテーブルに変換
        context = self._prepare_context(context)

        content = template.render(**context)

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f'✓ Report generated: {output_path}')

    def _prepare_context(self, context):
        """コンテキストデータを前処理（DataFrameをMarkdownに変換）"""
        prepared = {}

        for key, value in context.items():
            if isinstance(value, pd.DataFrame):
                prepared[key] = self._df_to_markdown(value)
            elif isinstance(value, dict):
                # 辞書の中のDataFrameも変換
                prepared[key] = {
                    k: self._df_to_markdown(v) if isinstance(v, pd.DataFrame) else v
                    for k, v in value.items()
                }
            else:
                prepared[key] = value

        return prepared

    @staticmethod
    def _df_to_markdown(df, **kwargs):
        """DataFrameをMarkdownテーブルに変換"""
        if not isinstance(df, pd.DataFrame):
            return df

        default_kwargs = {'index': False, 'floatfmt': '.4f'}
        default_kwargs.update(kwargs)

        return df.to_markdown(**default_kwargs)
```

#### generate_report.pyの修正

```python
# generate_report.py

from lib.report_generator import ReportGenerator

def generate_markdown_report(data_path, output_dir, results, session_id):
    """
    マークダウンレポートを生成（テンプレートベース）
    """
    report_path = output_dir / 'REPORT.md'
    print(f'生成中: マークダウンレポート -> {report_path}')

    # ReportGenerator初期化
    generator = ReportGenerator()

    # テンプレートに渡すコンテキスト
    context = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': data_path.name,
        'session_id': session_id,
        'data_hash': results.get('data_hash', 'N/A'),
        'data_info': results.get('data_info', {}),
        'band_statistics': results.get('band_statistics'),
        'band_power_img': results.get('band_power_img'),
        'psd_img': results.get('psd_img'),
        'psd_peaks': results.get('psd_peaks'),
        'spectrogram_img': results.get('spectrogram_img'),
        'band_ratios_img': results.get('band_ratios_img'),
        'band_ratios_stats': results.get('band_ratios_stats'),
        'spike_analysis': results.get('spike_analysis'),
        'paf_img': results.get('paf_img'),
        'paf_summary': results.get('paf_summary'),
        'iaf': results.get('iaf'),
        'paf_time_img': results.get('paf_time_img'),
        'paf_time_stats': results.get('paf_time_stats'),
        'paf_stability': results.get('paf_stability', '不明'),

        # 新機能（将来実装）
        'executive_summary': results.get('executive_summary'),
        'time_segments': results.get('time_segments'),
        'historical_comparison': results.get('historical_comparison'),
        'action_advice': results.get('action_advice'),

        # セッション評価（既存ロジックから抽出）
        'session_evaluations': _build_session_evaluations(results)
    }

    # テンプレートからレポート生成
    generator.generate('report_template.md', report_path, context)

    print(f'✓ レポート生成完了: {report_path}')


def _build_session_evaluations(results):
    """セッション評価を構築（既存ロジックをヘルパー関数化）"""
    evaluations = []

    if 'band_ratios_stats' not in results:
        return evaluations

    for _, row in results['band_ratios_stats'].iterrows():
        ratio_name = row['指標']
        avg_value = row['平均値']

        if 'リラックス' in ratio_name:
            level = 'とても高い' if avg_value > 2.0 else '高い' if avg_value > 1.0 else '普通'
        elif '集中' in ratio_name:
            level = 'とても高い' if avg_value > 2.0 else '高い' if avg_value > 1.0 else '普通'
        elif '瞑想' in ratio_name:
            level = '深い' if avg_value > 1.5 else '中程度' if avg_value > 0.8 else '浅い'
        else:
            level = '不明'

        evaluations.append({
            'name': ratio_name,
            'value': f'{avg_value:.3f}',
            'level': level
        })

    return evaluations
```

---

### 提案3: 設定ファイルの外部化（優先度：中）

#### 設定ファイルの導入

**必要な追加パッケージ**:
```txt
# requirements.txt に追加
pyyaml>=6.0
```

**ディレクトリ構造**:
```
titan2/
├── config/
│   ├── analysis_config.yaml      # 分析パラメータ
│   ├── scoring_config.yaml       # スコアリング設定
│   └── visualization_config.yaml # 可視化設定
```

**設定ファイル**: `config/analysis_config.yaml` (新規作成)

```yaml
# 分析パラメータ設定

analysis:
  # 時系列分析
  segment_duration_minutes: 5
  window_size_seconds: 4
  overlap_ratio: 0.5

  # 周波数バンド定義（必要に応じてカスタマイズ可能）
  frequency_bands:
    delta: [0.5, 4]
    theta: [4, 8]
    alpha: [8, 13]
    beta: [13, 30]
    gamma: [30, 50]
    frontal_theta: [6, 7]  # Fmθ専用

  # データ品質
  quality_filter:
    enabled: false
    min_signal_quality: 0.8
    remove_outliers: true
    outlier_std_threshold: 3.0

# PSD計算
psd:
  method: 'welch'
  fmin: 0.5
  fmax: 50
  n_fft: 1024
  n_overlap: 512

# スペクトログラム
spectrogram:
  method: 'morlet'
  freqs_min: 1
  freqs_max: 50
  n_cycles: 7
```

**設定ファイル**: `config/scoring_config.yaml` (新規作成)

```yaml
# スコアリング設定

scoring:
  # 総合スコアの重み配分
  weights:
    frontal_theta: 0.30      # Fmθパワー
    alpha_stability: 0.20    # Alpha安定性
    attention_span: 0.20     # 注意持続力
    beta_alpha_ratio: 0.15   # 緊張度の低さ
    data_quality: 0.15       # データ品質

# 評価閾値
thresholds:
  # リラックス度 (α/β)
  relaxation_ratio:
    very_high: 2.0
    high: 1.0
    normal: 0.5

  # 集中度 (β/θ)
  concentration_ratio:
    very_high: 2.0
    high: 1.0
    normal: 0.5

  # 瞑想深度 (θ/α)
  meditation_depth:
    deep: 1.5
    moderate: 0.8
    shallow: 0.3

  # PAF安定性（変動係数）
  paf_stability:
    very_stable: 2
    stable: 5
    moderate: 10
    unstable: 999

# 総合スコア評価レベル
score_levels:
  excellent: 80
  good: 60
  fair: 40
  poor: 0
```

**設定ファイル**: `config/visualization_config.yaml` (新規作成)

```yaml
# 可視化設定

visualization:
  # 図の基本設定
  figure:
    dpi: 300
    width: 12
    height: 8
    style: 'seaborn-v0_8-darkgrid'

  # カラースキーム
  colors:
    delta: '#1f77b4'
    theta: '#ff7f0e'
    alpha: '#2ca02c'
    beta: '#d62728'
    gamma: '#9467bd'

  # フォント
  font:
    family: 'sans-serif'
    size: 12
    japanese: 'Noto Sans CJK JP'

  # 出力形式
  output:
    format: 'png'
    transparent: false
    bbox_inches: 'tight'
```

#### ConfigLoaderクラスの実装

**ファイル**: `lib/config_loader.py` (新規作成)

```python
import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigLoader:
    """YAML設定ファイルの読み込みと管理"""

    def __init__(self, config_dir=None):
        if config_dir is None:
            config_dir = Path(__file__).parents[1] / 'config'

        self.config_dir = Path(config_dir)
        self._cache = {}

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        設定ファイルを読み込み

        Parameters:
            config_name: 設定ファイル名（.yaml拡張子なし）

        Returns:
            設定辞書
        """
        if config_name in self._cache:
            return self._cache[config_name]

        config_path = self.config_dir / f'{config_name}.yaml'

        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_path}')

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self._cache[config_name] = config
        return config

    def get(self, config_name: str, *keys, default=None):
        """
        ネストされた設定値を取得

        Parameters:
            config_name: 設定ファイル名
            *keys: ネストされたキー
            default: デフォルト値

        Examples:
            >>> config.get('analysis_config', 'analysis', 'segment_duration_minutes')
            5
        """
        config = self.load(config_name)

        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value


# グローバルインスタンス
_config_loader = None

def get_config_loader():
    """ConfigLoaderのシングルトンインスタンスを取得"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(config_name: str) -> Dict[str, Any]:
    """設定ファイルを読み込み（簡易版）"""
    return get_config_loader().load(config_name)


def get_config_value(config_name: str, *keys, default=None):
    """設定値を取得（簡易版）"""
    return get_config_loader().get(config_name, *keys, default=default)
```

#### 使用例

```python
# generate_report.py

from lib.config_loader import load_config, get_config_value

def run_full_analysis(data_path, output_dir):
    # 設定読み込み
    analysis_config = load_config('analysis_config')
    scoring_config = load_config('scoring_config')

    # 設定値の取得
    segment_duration = analysis_config['analysis']['segment_duration_minutes']
    fmtheta_band = analysis_config['analysis']['frequency_bands']['frontal_theta']

    # または直接取得
    weights = get_config_value('scoring_config', 'scoring', 'weights')
    relaxation_thresholds = get_config_value('scoring_config', 'thresholds', 'relaxation_ratio')

    # 評価ロジックで使用
    avg_value = 1.5
    if avg_value > relaxation_thresholds['very_high']:
        level = 'とても高い'
    elif avg_value > relaxation_thresholds['high']:
        level = '高い'
    elif avg_value > relaxation_thresholds['normal']:
        level = '普通'
    else:
        level = '低い'
```

---

### 提案4: run_analysis.shの改善（優先度：中）

#### 改善点

**現状の問題**:
- 最新CSVを自動選択（意図しないファイルを使う可能性）
- 分析後のRaw data削除オプションがない

**改善版**: `issues/003_improve_daily_report/run_analysis.sh`

```bash
#!/bin/bash
#
# Muse脳波データ基本分析 実行スクリプト（改善版）
#
# Usage:
#   ./run_analysis.sh [CSV_FILE_PATH] [OPTIONS]
#
# Options:
#   --keep-raw      Raw CSVを削除しない（デフォルトは削除）
#   --archive       Raw CSVを圧縮してアーカイブ
#

set -e

# プロジェクトルートへの相対パス
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# オプション解析
KEEP_RAW=false
ARCHIVE_RAW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-raw)
            KEEP_RAW=true
            shift
            ;;
        --archive)
            ARCHIVE_RAW=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            CSV_FILE="$1"
            shift
            ;;
    esac
done

# 仮想環境のチェック
if [ ! -d "$PROJECT_ROOT/titan" ]; then
    echo "エラー: 仮想環境 'titan' が見つかりません"
    echo "以下のコマンドでセットアップしてください:"
    echo "  cd $PROJECT_ROOT"
    echo "  python3 -m venv titan"
    echo "  source titan/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 仮想環境の有効化
source "$PROJECT_ROOT/titan/bin/activate"

# CSVファイルパスの取得
if [ -z "$CSV_FILE" ]; then
    # 引数なしの場合、dataディレクトリの最新CSVを表示して確認
    LATEST_CSV=$(find "$PROJECT_ROOT/data" -name "mindMonitor_*.csv" -type f -printf "%T@ %p\n" | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -z "$LATEST_CSV" ]; then
        echo "エラー: data/ ディレクトリにCSVファイルが見つかりません"
        echo ""
        echo "Usage: $0 [CSV_FILE_PATH] [--keep-raw] [--archive]"
        exit 1
    fi

    echo "最新のCSVファイル: $(basename "$LATEST_CSV")"
    read -p "このファイルで分析を実行しますか? [Y/n] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "キャンセルしました"
        exit 0
    fi

    CSV_FILE="$LATEST_CSV"
else
    if [ ! -f "$CSV_FILE" ]; then
        echo "エラー: ファイルが見つかりません: $CSV_FILE"
        exit 1
    fi
fi

# 分析実行
echo "============================================================"
echo "Muse脳波データ基本分析を実行します"
echo "============================================================"
echo ""
echo "データファイル: $(basename "$CSV_FILE")"
echo "出力先: $SCRIPT_DIR"
echo "Raw data削除: $([ "$KEEP_RAW" = true ] && echo "しない" || echo "する")"
echo "アーカイブ: $([ "$ARCHIVE_RAW" = true ] && echo "する" || echo "しない")"
echo ""

cd "$SCRIPT_DIR"

# Pythonスクリプトにオプションを渡す
PYTHON_ARGS="--data \"$CSV_FILE\" --output ."
[ "$KEEP_RAW" = true ] && PYTHON_ARGS="$PYTHON_ARGS --keep-raw"
[ "$ARCHIVE_RAW" = true ] && PYTHON_ARGS="$PYTHON_ARGS --archive"

eval python generate_report.py $PYTHON_ARGS

echo ""
echo "============================================================"
echo "分析完了!"
echo "============================================================"
echo ""
echo "生成されたファイル:"
echo "  - REPORT.md (マークダウンレポート)"
echo "  - img/*.png (グラフ画像)"
echo ""

if [ "$KEEP_RAW" = false ]; then
    echo "✓ Raw CSVは削除されました"
fi

if [ "$ARCHIVE_RAW" = true ]; then
    echo "✓ Raw CSVはアーカイブされました: data/archive/"
fi

echo ""
echo "レポートを確認:"
echo "  cat REPORT.md"
echo ""
```

#### generate_report.pyのコマンドライン引数追加

```python
# generate_report.py

def main():
    parser = argparse.ArgumentParser(
        description='Muse脳波データの基本分析とレポート生成（リファクタリング版）'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='入力CSVファイルパス'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent,
        help='出力ディレクトリ（デフォルト: スクリプトと同じディレクトリ）'
    )
    parser.add_argument(
        '--keep-raw',
        action='store_true',
        help='Raw CSVを削除しない'
    )
    parser.add_argument(
        '--archive',
        action='store_true',
        help='Raw CSVを圧縮してアーカイブ'
    )

    args = parser.parse_args()

    # ... (以下省略)
```

---

## 実装順序

### Phase 1: 基盤整備（Week 1）

1. **DataManagerクラスの実装**
   - [ ] `lib/data_manager.py` 作成
   - [ ] `data/raw/`, `data/processed/` ディレクトリ作成
   - [ ] `generate_report.py` にDataManager統合
   - [ ] テスト実行

2. **クリーンアップスクリプト**
   - [ ] `scripts/cleanup_raw_data.sh` 作成
   - [ ] 実行権限付与: `chmod +x scripts/cleanup_raw_data.sh`

3. **既存データの移行**
   - [ ] 現在の`data/*.csv`を手動削除
   - [ ] 新システムでテスト実行

### Phase 2: テンプレート化（Week 2）

4. **Jinja2テンプレートの作成**
   - [ ] `templates/report_template.md` 作成
   - [ ] `requirements.txt` にjinja2追加
   - [ ] pip install

5. **ReportGeneratorクラスの実装**
   - [ ] `lib/report_generator.py` 作成
   - [ ] `generate_report.py` をテンプレートベースに変更
   - [ ] テスト実行

### Phase 3: 設定外部化（Week 3）

6. **設定ファイルの作成**
   - [ ] `config/analysis_config.yaml` 作成
   - [ ] `config/scoring_config.yaml` 作成
   - [ ] `config/visualization_config.yaml` 作成
   - [ ] `requirements.txt` にpyyaml追加

7. **ConfigLoaderクラスの実装**
   - [ ] `lib/config_loader.py` 作成
   - [ ] `generate_report.py` で設定ファイルを使用
   - [ ] ハードコードされた閾値を設定ファイルに移行

### Phase 4: 統合とテスト（Week 4）

8. **run_analysis.shの改善**
   - [ ] オプション追加（--keep-raw, --archive）
   - [ ] CSVファイル選択の確認機能

9. **ドキュメント更新**
   - [ ] README.md更新
   - [ ] 使用例の追加

10. **最終テスト**
    - [ ] 完全な分析フロー実行
    - [ ] 既存issue (001, 002) の動作確認

---

## 期待される効果

### 1. ディスク容量の削減
- **現状**: 月間約9GB（毎日30分瞑想想定）
- **改善後**: 月間約10MB（メタデータとサマリーのみ）
- **削減率**: 99.9%

### 2. 保守性の向上
- テンプレート分離によりレポート構造の変更が容易
- 設定ファイルによりパラメータ調整が簡単
- コードの可読性向上（ビジネスロジックと表示の分離）

### 3. トレーサビリティの確保
- セッションIDとメタデータによる完全な追跡
- ファイルハッシュによるデータ整合性検証
- 過去データとの比較機能の基盤

### 4. 拡張性の向上
- 多言語対応が容易（テンプレート追加のみ）
- 新指標の追加が簡単（設定ファイル更新）
- レポート形式の追加（HTML, PDFなど）

---

## 参考: ファイルサイズ比較

```
# 現状（Raw CSVのみ保存）
mindMonitor_2025-10-31.csv          197MB

# 改善後（処理済みデータのみ）
session_20251031_160401/
  ├── metadata.json                  2KB
  ├── summary_stats.json             5KB
  └── key_metrics.parquet            50KB
合計:                                 57KB

圧縮率: 99.97%
```

---

## 備考

- 既存の`issues/001_eeg_analysis`と`issues/002_optics_analysis`にも同様の改善を適用可能
- Jupyter Notebookとの統合も検討（`.ipynb`でのインタラクティブ分析）
- CI/CDパイプラインへの組み込み（自動レポート生成）
