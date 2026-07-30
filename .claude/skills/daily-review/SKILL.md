---
name: daily-review
description: 瞑想セッションの日次AIレビュー。Muse脳波＋SelfLoops HRVデータを取得・分析してレポートを生成し、Claudeが脳波・自律神経・前頭前野・時系列の観点で解釈して対話する。「今日の瞑想をレビューして」「瞑想の振り返り」「daily-review」「瞑想セッションを分析して」等で起動。瞑想・EEG・HRV・Muse・SelfLoopsの振り返りや所感を求められたら、明示的に「レポート生成」と言われなくても使う。
user-invocable: true
allowed-tools: Bash, Read, Glob
---

# 瞑想日次レビュースキル

データ取得 → レポート生成 → AIレビュー を1フローで実行し、レビュー後に対話する。
AIレビューの本体は Claude 自身がレポートの数値を読んで解釈すること。ローカルで完結させる（GitHub Actions は使わない）。

## オプション

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--no-fetch` | Step 1（データ取得）をスキップし、`data/muse/` の既存CSVを使う | fetchあり |
| `--date YYYY-MM-DD` | 対象セッションの日付を指定 | latest（最新ファイル） |

例:
- `/daily-review` → 取得→生成→レビュー
- `/daily-review --no-fetch` → 取得スキップ、手元データでレビュー
- `/daily-review --date 2026-01-10` → 日付指定

## Step 0: 起動時刻の確認

```bash
date '+%Y-%m-%d %H:%M %A'
```

## Step 1 + 2: データ取得とレポート生成

`scripts/run_analysis.sh` が「取得 → 最新Muse CSV選択 → 同一タイムスタンプのSelfLoops自動マッチング → レポート生成」を一括で担う。出力は `tmp/REPORT.md` と `tmp/img/*.png`。

```bash
cd /home/tsu-nera/repo/satoru
bash scripts/run_analysis.sh --fetch          # 取得あり（--date指定時は下記）
```

- `--date YYYY-MM-DD` 指定時は fetch を `download_data.sh all <date>` で先に回してから、`data/muse/` の該当CSVパスを渡す:
  ```bash
  bash scripts/download_data.sh all 2026-01-10
  bash scripts/run_analysis.sh data/muse/mindMonitor_2026-01-10--*.csv
  ```
- `--no-fetch` 時は `--fetch` を付けず `bash scripts/run_analysis.sh`（`data/muse/` の最新CSVを自動選択）。

**落とし穴:**
- fetch は `.env` の `GDRIVE_CREDENTIALS`・`GDRIVE_FOLDER_ID_MUSE`・`GDRIVE_FOLDER_ID_SELFLOOPS` が必須。`.env` が無い／認証エラーが出たら、その旨を報告し `--no-fetch` で手元データを使うよう促す（勝手に `.env` を作らない）。
- SelfLoops が見つからない場合は EEG のみで生成される。これは正常動作。レビューでは HRV セクションが欠落する前提で扱う。
- セッションログ（Google Sheets）への保存はしない。本番の記録は GitHub Actions 側が担うため、ローカルのレビュー再実行でログを二重化させない（`SAVE_TO` は既定の `none` のまま）。

エラーが出たら握りつぶさず内容を報告する。

## Step 3: AIレビュー

`tmp/REPORT.md` を読み、下記観点で解釈する。レポートに含まれる図（`tmp/img/*.png`）は必要に応じて参照する。

### レビュー観点

計器が測っている構成概念を意識して読む。数値の羅列ではなく「このセッションで何が起きたか」を言語化する。

- **脳波（EEG帯域）**: α・θ帯のパワー。瞑想が深まると α↑・θ↑ の傾向。FAA（前頭α非対称）は左右バランス — 左優位はポジティブ/接近傾向の目安。IAF/PAF は個人のαピーク周波数で、日による変動も見る。
- **自律神経（HRV）**: HRVトレンドは副交感神経の活性＝リラックス到達度の目安（上昇＝深いリラックス）。心拍推移、呼吸との同調。※SelfLoops欠落時はスキップ。
- **前頭前野（fNIRS）**: 酸素化ヘモグロビンの推移。集中と弛緩のバランスを見る。
- **時系列推移**: セッション内の状態変化（入り〜深まり〜終盤）。深い状態に入れた時間帯はどこか、後半で崩れていないか。
- **接続品質（HSI/信号品質）**: 品質が悪い区間の数値は解釈の信頼度を割り引く。品質不良が広範ならレビュー全体に注意書きを添える。

### 出力形式

日本語で報告する。

```
## 瞑想レビュー（YYYY-MM-DD）

### 総合所感
このセッションを一言で（例: 深く入れた / 浅め / 乱れあり）

### 脳波
- α/θ の傾向、FAA、IAF/PAF の所感

### 自律神経（HRV）
- リラックス到達度の傾向（データなしなら明記）

### 前頭前野
- 集中・弛緩のバランス

### 時系列の所感
- セッション内の推移、深く入れた時間帯

### 気づき・次回への提案
- 具体的に2-3個

---
レポート: `tmp/REPORT.md` / 図: `tmp/img/` / サマリー: `tmp/summary.csv`
図込みで読む: `cd tmp && mdcat REPORT.md`
```

`tmp/` は実行のたび上書きされる。mdcat の落とし穴は2つ:
- `-p`（ページャ）を付けるとインライン画像が落ちてOSC 8リンクに置換される。画像を見たいなら付けない。
- レポート内の画像参照は `img/*.png` の相対パスなので、リポジトリルートでなく `tmp/` 内で実行する。

レビュー後はディスカッションに入る。

## スコープ（v1）

- 単発セッションのレビューに絞る。
- 過去セッションとの横断比較（Google Sheets のセッションログ活用）は未実装 — 将来の拡張点。
- レビュー結果の記録保存（journal相当）は未定。
