---
name: resonance-review
description: 共鳴呼吸（HRV バイオフィードバック、Elite HRV で記録）のセッションを振り返る。Gmail に届いた Elite HRV の export を Drive に取り込み、scripts/elitehrv.py で分析して数行でレビューし、journal に残す。「共鳴呼吸の分析」「HRV のセッションを見て」「Elite HRV のデータが届いた」「resonance-review」等で起動。Elite HRV・共鳴呼吸・RSA・朝の readiness の振り返りを求められたら、明示されなくても使う。Muse の瞑想（サマタ）は session-review。
user-invocable: true
allowed-tools: Bash, Read, Edit, Write
---

# 共鳴呼吸セッションのレビュー

前提の枠組みは memory の `resonance-breathing-plan.md`。要点: **セッションの値は練習の質で、成果ではない**（成果は Fitbit 夜間 HRV）。条件は 6/9（4 bpm、周期15秒）・10分・時間帯固定。

## 1. export を取り込む

export は本人がアプリで実行する（adb でアプリを操作しない）。リンクは数時間で消える。

```bash
cd ~/repo/gtd && uv run scripts/gmail/search.py --body 'elitehrv.com/export/[A-Za-z0-9]+' 'from:support@elitehrv.com newer_than:2d'
cd ~/repo/satoru && uv run python scripts/elitehrv.py import '<リンク>'
```

- メールが無い、またはリンクが切れている（import が download で失敗）→ 本人に export を頼んで待つ
- import が非0で終わったら止まって内容を報告する。「欠けている reading」が出たら Drive の正本を守るための拒否なので、上書きを強行しない
- 成功したらメールをゴミ箱へ: `cd ~/repo/gtd && uv run scripts/gmail/trash.py <thread ID>`（search.py の2列目。MCP の trash_thread は権限不足で失敗する）

## 2. 分析する

練習の種類を1回だけ聞く（`rf` = 呼吸だけ / `rf+metta` = メッター併用）。答えが無ければ直近の biofeedback 行と同じにする。

```bash
uv run python scripts/elitehrv.py analyze --drive --practice rf+metta
```

CSV に無い reading だけを処理し、`logs/resonance_log.csv` に upsert する。practice は biofeedback にだけ付く。手元に生データは置かない（Drive が正本、iiyama と vaio のどちらでも同じ手順）。

「0 / N readings」で処理済みの reading を見直すときは、表は出ないので `logs/resonance_log.csv` の行を読む（2分区間は CSV に無い）。

## 3. レビュー（数行）

出力の表を読んで、次だけを書く:

- **呼吸がそろったか**: Breathing Rate が 4.0 前後か、Breath Period SD（大きいほどリズムが乱れた）
- **RSA振幅**: 主指標。末尾の Comparison 行で直近5回の中央値と比べる
- **区間**: 2分区間のどこで RSA振幅や周期が崩れたか
- **artifact**: 5% を超えたら値の信頼度を割り引くと書く（欠落拍が多い日はベルトの接触）

書かないこと:
- LF/HF・HF Power の解釈。4 bpm では RSA が丸ごと LF に入り HF が空になるので意味を持たない（HF Peak 0.150 は帯域の端で、ピークではない）
- RMSSD の上下を「HRV が上がった/下がった」と語ること。4 bpm では呼吸の深さで動く
- Elite HRV アプリの表示との差。時間領域はほぼ一致、周波数領域は方式違いで ±7% ずれる。一致は目標にしていない

readiness（約2.5分）は時間領域だけ。同じ kind の過去と比べる。2026-01 の共鳴周波数測定（3分 × 数本）も長さで readiness に分類されているので、1月の readiness の値は朝の計測と混ざっている。

## 4. 記録

`journal` スキルの手順で `~/repo/dailybuild-private/reports/meditation/journal/YYYY-Wxx.md` に追記する（Muse のエントリと同じファイル、日付順）。数値は CSV にあるので再掲せず、判断を左右した値だけ引く。

```markdown
## YYYY-MM-DD HH:MM 共鳴呼吸 (10.0分, rf+metta)

**所感**: 一行
- **文脈**: 本人が話した分だけ。不明なら「未確認」
- **解釈**: なぜそうなったと考えるか
- **次に試すこと**: あれば
```

最後に `logs/resonance_log.csv` を main に直接 commit する（`/direct-commit`）。push しないと別の PC から前回比較ができない。
