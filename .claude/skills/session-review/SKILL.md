---
name: session-review
description: 瞑想セッション単位のAIレビュー。Muse脳波＋SelfLoops HRVデータを取得・分析してレポートを生成し、Claudeが脳波・自律神経・前頭前野・時系列の観点で解釈して対話する。「今日の瞑想をレビューして」「瞑想の振り返り」「session-review」「瞑想セッションを分析して」等で起動。瞑想・EEG・HRV・Muse・SelfLoopsの振り返りや所感を求められたら、明示的に「レポート生成」と言われなくても使う。
user-invocable: true
allowed-tools: Bash, Read, Glob
---

# 瞑想セッションレビュースキル

データ取得 → レポート生成 → AIレビュー を1フローで実行し、レビュー後に対話する。
AIレビューの本体は Claude 自身がレポートの数値を読んで解釈すること。ローカルで完結させる（GitHub Actions は使わない）。

## オプション

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--no-fetch` | Step 1（データ取得）をスキップし、`data/muse/` の既存CSVを使う（`.csv`/`.csv.gz` どちらも対象） | fetchあり |
| `--date YYYY-MM-DD` | 対象セッションの日付を指定。同一日に複数セッションがある日は下記の落とし穴を読む | latest（最新ファイル） |

例:
- `/session-review` → 取得→生成→レビュー
- `/session-review --no-fetch` → 取得スキップ、手元データでレビュー
- `/session-review --date 2026-01-10` → 日付指定

## Step 0: 起動時刻と前回エントリの確認

```bash
date '+%Y-%m-%d %H:%M %A'
git -C ~/repo/dailybuild-private pull --rebase --autostash -q
ls ~/repo/dailybuild-private/reports/meditation/journal/*-W*.md | tail -2 | xargs cat   # 直近2週。無ければ凍結した月ファイル 2026-11.md の末尾
```

前回の「次に試すこと」がどうなったかは、レビュー本文で扱う。読まずに始めると毎回同じ提案を出す。

## Step 1 + 2: データ取得とレポート生成

`scripts/run_analysis.sh` が「取得 → 最新Muse CSV選択 → 同一タイムスタンプのSelfLoops自動マッチング → レポート生成」を一括で担う。出力は `tmp/REPORT.md` と `tmp/img/*.png`。

出力の読み手で役割を分ける。`tmp/REPORT.md` は agent が読む1回の詳細（区間・帯域・品質・図）で、本人には案内しない。本人が見るのは Grafana の瞑想画面（回をまたいだ推移と各回の要約の数字。dailybuild が `logs/session_log.csv` から作る）、解釈はチャットで伝える。

```bash
cd /home/tsu-nera/repo/satoru
bash scripts/run_analysis.sh --fetch          # 取得あり（--date指定時は下記）
```

- `--date YYYY-MM-DD` 指定時は fetch を `download_data.sh all <date>` で先に回してから、`data/muse/` の該当CSVパスを渡す:
  ```bash
  bash scripts/download_data.sh all 2026-01-10
  bash scripts/run_analysis.sh data/muse/mindMonitor_2026-01-10--*.csv
  ```
  **glob が2本以上に展開する日（1日に複数セッション）は壊れる。** `run_analysis.sh` の第2引数は出力先ディレクトリなので、2本目のCSVパスが `OUTPUT_DIR` として渡る。`ls data/muse/mindMonitor_<date>--*.csv` で本数を確認し、複数あればどのセッションかをユーザーに確認してフルパス1本だけを渡す。
- `--no-fetch` 時は `--fetch` を付けず `bash scripts/run_analysis.sh`（`data/muse/` の最新CSVを自動選択。`.csv.gz` はそのまま渡せば pandas が自動展開する）。

**落とし穴:**
- fetch は `.env` の `GDRIVE_CREDENTIALS`・`GDRIVE_FOLDER_ID_MUSE`・`GDRIVE_FOLDER_ID_SELFLOOPS` が必須。`.env` が無い／認証エラーが出たら、その旨を報告し `--no-fetch` で手元データを使うよう促す（勝手に `.env` を作らない）。
- SelfLoops が見つからない場合は EEG のみで生成される。これは正常動作。レビューでは HRV セクションが欠落する前提で扱う。
- セッションログ `logs/session_log.csv` に自動 upsert される（`SAVE_TO` 既定は `csv`）。timestamp キーなので再実行しても重複しない。

エラーが出たら握りつぶさず内容を報告する。

タップログ（マインドワンダリングの主観記録）は普段使わない追加オプション。`run_analysis.sh` が「検出されたタップログファイル」を出した日だけ、そのファイルをアップロードし、レビューでも扱う。出なかった日はタップに一切触れない（探さない・「記録なし」と書かない）。

```bash
uv run python scripts/upload_to_gdrive.py --file data/taps/<当日のタップログ>.csv
```

`GDRIVE_FOLDER_ID_TAPS` 未設定なら警告を出してスキップする。冪等だが、Drive UI から手動で置かれた同名ファイルは検出できない（drive.file スコープの制約）。認証はOAuthユーザー認証（初回のみブラウザ同意、テスト状態のOAuth同意画面だと約7日でトークン失効し再同意が必要）。

## Step 2.5: 注釈（種類・主観評価）

レビュー出力の前に、次の3つを**1回だけまとめて**聞く（Step 4 の「文脈は聞かない」の例外）。

- 瞑想の種類（既定 `samatha`。技法を使わず目を閉じて座るだけなら `control`）
- 集中の安定 1〜5（浮つき: 集中が保てないほど低い）
- 眠気 1〜5（沈み込み: 眠いほど高い）

答えた分だけ書く。答えが無ければ書かずに進む（再度聞かない）。`--date` で過去のセッションを見ているときは `--timestamp` に `tmp/REPORT.md` の開始時刻（session_log の timestamp）を渡す。

```bash
uv run python scripts/annotate_session.py --timestamp latest --type samatha --focus 4 --drowsiness 2
```

注釈列は分析の再実行でも backfill でも上書きされない。

注釈の後に評価の進みを出し、レビュー末尾の「次回」に使う。対照は4回に1回で、座った後では今日の回を対照に変えられないので、**次の回が対照の番なら必ず予告する**（座る前に分かるように）。

```bash
uv run python scripts/eval_progress.py
```

## Step 3: AIレビュー

`tmp/REPORT.md` を読み、下記観点で解釈する。レポートに含まれる図（`tmp/img/*.png`）は必要に応じて参照する。本文で回をまたいだ比較に触れるときは、Grafana の瞑想画面のどのパネルで見られるかを一言添える（例: 「除外率のパネルで今回だけ高い」）。

### レビュー観点

計器が測っている構成概念を意識して読む。数値の羅列ではなく「このセッションで何が起きたか」を言語化する。

- **脳波（EEG帯域）**: セッション内の深まりは TP の α・δ の**絶対 dB の入り→終わり**（`tp_*_db_first/last`）で読む。相対 % の α 上昇は δ の減衰で起きるので深まりと読まない。前頭の α は `alpha_osc_db`、ゲインの上下は全帯域が一緒に動くかで見分ける。IAF は日による変動も見る。FAA はレポートに出ないので扱わない。
- **自律神経（HRV）**: HRVトレンドは副交感神経の活性＝リラックス到達度の目安（上昇＝深いリラックス）。心拍推移、呼吸との同調。※SelfLoops欠落時はスキップ。
- **前頭前野（fNIRS）**: 酸素化ヘモグロビンの推移。集中と弛緩のバランスを見る。
- **時系列推移**: セッション内の状態変化（入り〜深まり〜終盤）。深い状態に入れた時間帯はどこか、後半で崩れていないか。
- **接続品質（HSI/信号品質）**: 品質が悪い区間の数値は解釈の信頼度を割り引く。品質不良が広範ならレビュー全体に注意書きを添える。
- **前夜の睡眠**: 眠気・集中・θ/α を読む前に見る。寝不足の朝の眠気や θ/α の高さを深まりと読まない。値は dailybuild の起床日の行（`ssh vaio "grep '^<YYYY-MM-DD>,' ~/repo/dailybuild-private/reports/metrics_daily.csv"` の `sleep_hours`。列は1行目で確かめる）
- **マインドワンダリング（主観）**: タップ数と時間分布。セッション内のどこで注意が途切れたか。客観指標との一致/不一致を必ず見る — αが高いのにタップが多い、HRVは上がっているのに後半にタップが集中する、といった乖離は指標の解釈そのものを変える。※タップログ欠落時はスキップ。

### 出力形式

日本語で報告する。

```
## 瞑想レビュー（YYYY-MM-DD）

### 総合所感
このセッションを一言で（例: 深く入れた / 浅め / 乱れあり）

### 脳波
- TP の α・δ の入り→終わり、前頭 α、IAF の所感

### 自律神経（HRV）
- リラックス到達度の傾向（SelfLoops が無い日はセクションごと省略する）

### 前頭前野
- 集中・弛緩のバランス

### 時系列の所感
- セッション内の推移、深く入れた時間帯

### マインドワンダリング（主観）
- タップ数・時間分布、客観指標との一致/不一致（タップログが無い日はこのセクションごと省略する。使わない日の方が多い前提のため、毎回「記録なし」と書くとノイズになる）

### 前回からの変化
- 前回エントリの「次に試すこと」がどうなったか。未実施なら未実施と書く
- 前回と同じ所見が再現したか、しなかったか

### 気づき・次回への提案
- 具体的に2-3個

評価の進み: サマタ n/20・対照 n/6・主観あり n（`eval_progress.py` の出力）
次回: サマタ / **対照の番**（技法を使わず目を閉じて座るだけ）

---
画面: http://vaio:3000/d/meditation （Step 6 の反映後に今回の回が載る）
```

`tmp/` は実行のたび上書きされる。

レビュー後はディスカッションに入る。

## Step 4: ジャーナル記録

レビューとディスカッションが一段落したら `journal` スキルを呼び、dailybuild-private の瞑想 journal（週ファイル）に残して push する。**確認を取らずに実行する。**

レポートの数値は `logs/session_log.csv` にあるので再掲しない。ジャーナルに書くのは文脈・解釈・次に試すことだけ。

**文脈（意図・体調）が不明でも聞かない（瞑想の種類と主観評価だけは Step 2.5 で聞く）。「未確認」と書いて先に進む。** 毎セッション問診されるコストのほうが、文脈が埋まる利益より大きい。ユーザーが自分から話した分だけ書く。

## Step 5: 生データの削除

ジャーナルまで終わったら、そのセッションの Muse CSV を `data/muse/` から消す。**確認を取らずに実行する。** 正本は Drive の zip で、取り直しは数秒（往復一致は検証済み）。圧縮して手元に残すと Drive と重複したまま積み上がる。

```bash
rm "$CSV_PATH"   # run_analysis.sh が出力した「使用するCSVファイル」
```

`data/muse/` は空が平常。`--no-fetch` は「取得済みだが未レビュー」のファイルがあるときだけ意味を持つ。

## Step 6: セッションログの反映

**確認を取らずに実行する。** dailybuild（vaio）は satoru の `logs/session_log.csv` を読むだけのデータソースとして扱う。push・pull・瞑想画面の CSV の作り直しまでで1セット。

1. `logs/session_log.csv` だけを main に直接 commit・push する（`/direct-commit`）。journal は Step 4 で dailybuild-private に push 済み（satoru には置かない）
2. `ssh vaio 'git -C ~/repo/satoru pull --ff-only && cd ~/repo/dailybuild && uv run python scripts/dashboard.py build --only meditation'`（Grafana の瞑想画面の CSV を作り直す）。失敗したら数秒おいて1回リトライし、それでも駄目なら報告だけして終える（vaio の daily-fetch でも pull・作り直しされる）

## スコープ（v1）

- 単発セッションのレビューに絞る。
- 過去セッションとの横断比較は `logs/session_log.csv` を読んで行う。生データから計算し直す必要があるときは `data/muse/` ではなく scratchpad に落とす（`fetch_from_gdrive.py --download <date> --output <scratchpad>`）。`data/muse/` に混ぜると消してよいものの区別がつかなくなる。ただしaperiodic系の列は2026-08-27以降の行にしか入っていない（Sheets運用時代に一度も書かれなかったため）。
- レビュー結果は `journal` スキルで dailybuild-private の瞑想 journal（週ファイル）に蓄積する（Step 4）。
