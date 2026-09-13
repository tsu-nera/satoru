# Tap Logger（マインドワンダリング打刻）

瞑想セッション中に「注意がそれたことに気づいた」瞬間をスマホのブラウザからタップで打刻し、PC側のローカルサーバがCSVに記録する仕組み。記録基盤のみが対象で、既存の解析コード（`lib/sensors/`, `lib/report/`, `logs/session_log.csv`, `scripts/generate_report.py`）には手を入れていない。

## 起動手順

```bash
venv/bin/python scripts/tap_server.py
```

CLI 引数:

| 引数 | 既定値 | 説明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 待ち受けホスト |
| `--port` | `8765` | 待ち受けポート |
| `--output-dir` | `data/taps` | CSV出力先ディレクトリ |

起動すると、スマホから開くべきURL（`http://<PCのLAN IP>:<port>/`）が標準出力に表示される。

```bash
venv/bin/python scripts/tap_server.py --port 8765 --output-dir data/taps
```

## スマホからの接続手順

1. PCとスマホを同一LAN（同じWi-Fi）に接続する
2. PC側で `scripts/tap_server.py` を起動する
3. 起動時に表示されたURL（例: `http://192.168.1.10:8765/`）をスマホのブラウザで開く
4. 「スタート」をタップして計測を開始する
5. 注意がそれたことに気づいたら画面をタップする（計測中はタップ数やカウンタを表示しない仕様）
6. 終了する場合は画面隅の「ストップ」を長押し（1秒）または二度押しして確定する

## CSVスキーマ

出力先: `data/taps/<YYYY-MM-DDTHH-MM-SS>_taps.csv`（1セッション1ファイル）

| カラム | 意味 |
|--------|------|
| `seq` | 0始まりの通し番号 |
| `event` | `start` / `tap` / `stop` |
| `client_ts` | ブラウザ側の絶対時刻（`new Date().toISOString()`、ISO 8601） |
| `server_ts` | サーバ受信時の絶対時刻（`datetime.now().astimezone().isoformat()`、タイムゾーン付きISO 8601） |
| `elapsed_s` | `start` イベントの **server_ts** を基準にした経過秒（float, 小数3桁） |

`client_ts` と `server_ts` を両方残すのは必須要件。Muse App / SelfLoops / スマホの3系統は時計がズレるため、後からこのペアを使ってオフセットを推定できるようにしてある。`elapsed_s` はクライアント時計のズレに影響されないよう、サーバ側時刻を基準に計算する。

## 同期マーカーの運用

録音開始直後に「3回速打ち + 深呼吸1回」を入れる。EEGの動作アーチファクトとHRVの呼吸ピークが揃って出るため、後からMuse / SelfLoops / スマホの3系統のラグを実測して補正できる。

## 注意書き

タップ動作自体がEEGに運動アーチファクト、心拍に一過性の変化を起こす。将来の解析ではタップ ±2秒を除外区間として扱う必要がある。

## 既知の制約

- Screen Wake Lock APIはsecure context必須のため、平文HTTPでのLAN配信では使えない可能性が高い。使えない場合はOS側で画面消灯時間を延ばす運用でカバーすること。
- HTTPS化は本仕組みのスコープ外（未対応）。
