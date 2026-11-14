---
name: journal
description: 瞑想セッションのレビュー結果とディスカッションを dailybuild-private の瞑想 journal（週ファイル）に蓄積する。session-review の後に自動で呼ばれる。
argument-hint: "[YYYY-MM-DD]"
user-invocable: true
allowed-tools: Bash, Read, Write, Edit
---

# Journal Skill

`session-review` の解釈・文脈・次に試すことを `~/repo/dailybuild-private/reports/meditation/journal/YYYY-Wxx.md`（ISO 週、`date +%G-W%V`）に残す。週ファイル1本のみ。索引も状態ファイルも台帳も持たない。

置き場所は dailybuild-private（satoru は PUBLIC なので）。同じディレクトリの `2026-09.md` 〜 `2026-11.md` は月ファイル時代の凍結記録で、書き足さない。体調の週ジャーナル（`reports/journal/`）とは別のファイル。そちらへの瞑想の1行は gtd の `reviewing-health` が書く。

**読み手は将来の agent。** 人が読み返す前提で書かない。

**保守は自律実行・確認不要。** 「記録しますか?」と聞かない。

## 数値を書かない

セッションの数値は `logs/session_log.csv` に全部ある（1行1セッション、30列超）。**再掲せず、判断を左右した値だけ根拠として引く。** 過去比較が要るときは月ファイルを何本も読まず CSV を読む。

ジャーナルに書くのは CSV が持てないものだけ:

- **文脈** — 意図・体調・場所・瞑想の種類。**後から復元不能。不明なら「未確認」と書く**（推測で埋めない）
- **解釈** — なぜその数値になったと考えるか
- **次に試すこと**と、**前回それがどうなったか**

## 恒久的な知見はここに書かない

指標の落とし穴・文献知見・「試したが裏目に出た」は memory に置く。ジャーナルには経過だけ残す。振り分けは「来月も参照するか」で決める。

例: 「閉眼では α が高いほど MW が少ない（文献）」「呼吸5bpm は集中が落ちるので戻した」→ memory / 「今回は α↑ とタップ↑ が同時に起きた」→ ジャーナル

## エントリ形式

週ファイルが無ければ作る。H1 は `# YYYY-Wxx`。エントリは日付順。

```markdown
## YYYY-MM-DD HH:MM (N.N分)

**所感**: 一行

- **文脈**: 意図・体調・場所。不明なら「未確認」
- **解釈**: 根拠の値だけ引く
- **主観 vs 客観**: 一致か乖離か。乖離ならどの構成概念の差か（タップログがある日だけ書く。無い日は行ごと省く）
- **前回の結果**: 直前エントリの「次に試すこと」がどうなったか

**次に試すこと**
1. 検証可能な形で書く

**Discussion**: レビュー後の会話で深掘りした内容。無ければ省略
```

アクションの追跡は隣接2エントリで閉じる。週の最初のエントリでは「前回」は前の週ファイル（無ければ凍結した月ファイル）の最後にある。

再レビューでは同じ `## YYYY-MM-DD HH:MM` セクションを書き換える。**Discussion は消さない。**

## 保存

書いたら dailybuild-private に commit・push する（vaio も毎日 commit するので rebase してから）。他の未コミット変更を巻き込まない。

```bash
cd ~/repo/dailybuild-private && git add reports/meditation/journal && git commit -q -m "journal(meditation): YYYY-MM-DD のセッション" -- reports/meditation/journal && git pull --rebase --autostash -q && git push -q
```

## スコープ

週次・月次の rollup は未導入。必要になってから足す。
