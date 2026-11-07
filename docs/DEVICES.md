# 計測デバイス

satoru が扱う計測デバイスと、データがどこを通って届くか。所持の有無（買った・手放した）は所持品台帳（dailybuild-private の `items.csv`）で管理し、ここには通信・データ仕様・癖だけを書く。家の機器は agent-private の `home/`、スマホは agent-private の `infra/phones.md` に書いている。

「未確認」は、コードやデータから裏付けが取れていない項目。

## 一覧

| デバイス | 役割 | 使用状況 | データの経路 |
|---|---|---|---|
| Muse S Athena (Gen3) | EEG 4ch・fNIRS・PPG・IMU | 現役 | BLE → Mind Monitor → CSV → Google Drive → `data/muse/` |
| Wahoo TICKR（第一世代、2020-05-24 入手） | RR間隔 | 現役（毎回は着けない） | 瞑想: BLE → SelfLoops HRV → CSV → Google Drive → `data/selfloops/`。共鳴呼吸・朝の計測: BLE → Elite HRV → メールの export リンク → `data/elitehrv/` |
| Fitbit Charge 6 | 瞑想時間の記録（YOGA モード）。心拍も取れる | 現役 | Fitbit → Google Health API → dailybuild → Toggl（satoru は扱わない） |
| Muse 2 | EEG 4ch（fNIRS なし） | 2025-12 以降、計測記録なし | （台帳には載っているが現物は未確認） |
| xiaomi（Redmi 12 5G、Android） | Mind Monitor・SelfLoops・Elite HRV・Kubios HRV・Muse App の実行、打刻 | 現役 | 下の各行を参照 |
| スマホのブラウザ（打刻） | マインドワンダリングの打刻 | 現役 | ブラウザ → Tailscale HTTPS → `scripts/tap_server.py` → `data/taps/` |

## Muse S Athena

- 接続時のデバイス名は `MuseS-FA35`（Mind Monitor の `/muse/event/connected` イベントで確認）
- 仕様: [MUSE_S_ATHENA_SPECIFICATIONS.md](MUSE_S_ATHENA_SPECIFICATIONS.md)、CSV: [MIND_MONITOR_CSV_SPECIFICATION.md](MIND_MONITOR_CSV_SPECIFICATION.md)
- 記録の本線は Mind Monitor の CSV。zip で Drive に上がり、`scripts/fetch_from_gdrive.py` で取得する
- OSC でのリアルタイム受信は `lib/recorders/muse_osc.py`（`--source muse_app_osc` / `mind_monitor_osc`）
  - Muse App の OSC Output が送るのは RAW EEG だけで、バンドパワーは含まれない。Optics も約 8.6 Hz しか来ない（期待値は 64 Hz）ので、この経路は保留中。詳細: [MUSE_APP_OSC_OUTPUT.md](MUSE_APP_OSC_OUTPUT.md)
  - 公式 OSC 仕様: [MUSEIO_OSC_SPECIFICATION.md](MUSEIO_OSC_SPECIFICATION.md)
- fNIRS は額の5つの optode。頭皮の血流が混入するので、絶対値は皮質の活動として読まない

## 心拍計 + SelfLoops HRV

- 機種は Wahoo TICKR の第一世代（[共鳴周波数の測定記事](https://futurismo.biz/hrv-resonance-frequency/)、2026-01-20）。BLE と ANT+ のどちらに対応しているか、RR間隔をリアルタイムで出せるかは未確認（2026-01 に Elite HRV で RR間隔を記録できているので、BLE の RR 出力はあるはず）
- SelfLoops の CSV は、1行目が日本語月名の日時（例: `16 9月 2026 08:05:20`）、2行目以降が `timestamp,bpm,rr`（ms）
- Drive 上ではどのファイルも同じ名前（`SelfLoops HRV data`）で保存される。`scripts/rename_gdrive_selfloops.py` が、1行目の日時を使って日付入りの名前に付け替える
- ローダー: `lib/loaders/selfloops.py`

## 心拍計 + Elite HRV

- 共鳴呼吸（Breathing タブの Biofeedback）と朝の readiness reading に使う。ローダーは `lib/loaders/elite_hrv.py`
- アプリのデータは端末の内部領域にあり、adb からは取れない（`/sdcard/Android/data/com.elitehrv.app/` は空）
- 取り出し方: アプリでメール export を実行すると、support@elitehrv.com から「Elite HRV Export」というメールが届く。本文のリンク（`https://app.elitehrv.com/export/<hash>`、数時間で消える）を認証なしの curl で取ると zip が落ちてくる。中身は全期間の reading で、1 reading が1ファイル（`YYYY-MM-DD HH-MM-SS.txt`、RR間隔 ms を1行1値）。呼吸ペースや reading の種類は入っていないので、長さで見分ける（readiness は約2.5分、biofeedback は5分か10分）

## 時計のずれ

Muse（Mind Monitor）、SelfLoops、スマホの3系統は、それぞれ別の時計で記録している。打刻ログは `client_ts` と `server_ts` を両方持つ。録音開始直後に「3回速打ち + 深呼吸1回」を入れて、系統間のずれを後から実測する。詳細: [TAP_LOGGER.md](TAP_LOGGER.md)
