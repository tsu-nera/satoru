# Muse App OSC Output 仕様

> **⚠️ 保留中**: この機能は開発を一時中断しています。
> Muse AppのOSC OutputはRAW EEGのみを送信し、バンドパワー（Delta/Theta/Alpha/Beta/Gamma）は含まれません。
> Mind Monitor互換のレポート生成には、RAW EEGからFFTでバンドパワーを計算する追加実装が必要です。
> また、Opticsデータの送信頻度が低い（8.6Hz、期待値64Hz）ため、fNIRSグラフにも問題があります。

Muse公式アプリ（Muse: Meditation & Sleep）のOSC Output機能に関する調査結果。

## OSC仕様の系譜

Muse関連のOSC仕様には2つの系統がある。

```
┌─────────────────────────────────────────────────────────┐
│ Interaxon公式SDK（Muse-IO / Muse Direct）               │
│   - `/muse/eeg` 形式を定義（公式仕様）                   │
│   - 周波数帯パワー `/muse/elements/alpha_absolute` 等   │
└─────────────────────────────────────────────────────────┘
                          ↓ 準拠
┌─────────────────────────────────────────────────────────┐
│ Mind Monitor（サードパーティ）                          │
│   - Muse Lab、Muse-IO、Muse Directと互換                │
│   - `/muse/xxx` 形式を使用                              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Muse App（公式瞑想アプリ）                              │
│   - `/eeg` 形式（独自・簡略化）                         │
│   - 周波数帯パワーなし（生データのみ）                  │
│   - 公式SDKとは異なる実装                               │
└─────────────────────────────────────────────────────────┘
```

**注意**: Muse AppのOSC仕様は公式ドキュメントがなく、実際の通信から解析した結果である。

**公式SDK仕様（アーカイブ）**: [MuseIO Available Data](https://web.archive.org/web/20181105231756/http://developer.choosemuse.com/tools/available-data)

## 概要

Muse公式アプリには隠れたOSC Output機能があり、瞑想セッション中にリアルタイムでEEGデータをストリーミングできる。これにより、**公式アプリの瞑想ガイドやニューロフィードバック機能を使いながら、同時に独自のデータ処理が可能**になる。

Mind Monitorとの違い：
- Mind Monitorは公式アプリと同時に使用できない（Bluetooth接続の排他制御）
- 公式アプリのOSC Outputを使えば、瞑想しながらデータ取得が可能

## OSC設定

### アプリ側の設定

Muse App内のOSC Output設定：
- **IP**: 受信側PCのIPアドレス（例: `192.168.100.136`）
- **Port**: `5000`（任意）
- **Streaming Enabled**: ON

### 受信側の設定

- プロトコル: UDP
- ポート: 5000（アプリ側と合わせる）

## OSCパス仕様

Muse AppのOSCパスはMind Monitor（`/muse/eeg`形式）とは異なり、シンプルな形式を使用。

### 確認済みOSCパス

| パス | 内容 | データ形式 | レート |
|------|------|-----------|--------|
| `/eeg` | 生EEG | `(TP9, AF7, AF8, TP10)` float×4 | 256Hz |
| `/acc` | 加速度計 | `(x, y, z)` float×3 | 52Hz |
| `/gyro` | ジャイロスコープ | `(x, y, z)` float×3 | 52Hz |
| `/optics` | 光学センサー（fNIRS） | float×8 | 64Hz |
| `/drlref` | DRL/Reference電極 | `(drl, ref)` float×2 | - |
| `/test/annotation` | テストメッセージ | `('test',)` | 手動 |

### データ例

```
/eeg: (724.95, 724.95, 724.95, 724.95)
/acc: (-0.94, 0.31, -0.14)
/gyro: (-2.22, 1.90, -2.99)
/optics: (4.52, 14.42, 4.65, 13.85, 15.52, 12.37, 15.00, 10.19)
/drlref: (2720863.0, 143911.375)
```

### 公式SDK（MuseIO）仕様との比較

公式SDK（MuseIO）が定義するOSCパスと、Muse App / Mind Monitorの対応：

| データ種別 | 公式SDK (MuseIO) | Muse App | Mind Monitor |
|-----------|------------------|----------|--------------|
| 生EEG | `/muse/eeg` | `/eeg` | `/muse/eeg` |
| 加速度計 | `/muse/acc` | `/acc` | `/muse/acc` |
| ジャイロ | - | `/gyro` | `/muse/gyro` |
| DRL/REF | `/muse/drlref` | `/drlref` | - |
| バッテリー | `/muse/batt` | 未確認 | `/muse/batt` |
| Delta絶対値 | `/muse/elements/delta_absolute` | なし | `/muse/elements/delta_absolute` |
| Theta絶対値 | `/muse/elements/theta_absolute` | なし | `/muse/elements/theta_absolute` |
| Alpha絶対値 | `/muse/elements/alpha_absolute` | なし | `/muse/elements/alpha_absolute` |
| Beta絶対値 | `/muse/elements/beta_absolute` | なし | `/muse/elements/beta_absolute` |
| Gamma絶対値 | `/muse/elements/gamma_absolute` | なし | `/muse/elements/gamma_absolute` |
| Horseshoe | `/muse/elements/horseshoe` | なし | `/muse/elements/horseshoe` |
| 瞬き検出 | `/muse/elements/blink` | 未確認 | `/muse/elements/blink` |
| 歯の食いしばり | `/muse/elements/jaw_clench` | 未確認 | `/muse/elements/jaw_clench` |
| 集中度(実験的) | `/muse/elements/experimental/concentration` | なし | 隠し機能あり |
| リラックス度(実験的) | `/muse/elements/experimental/mellow` | なし | 隠し機能あり |
| fNIRS/Optics | - | `/optics` | `/muse/optics` |
| PPG | - | 未確認 | `/muse/ppg` |

**結論**: Muse Appは公式SDK仕様の `/muse/` プレフィックスを省略した簡略形式を使用し、周波数帯パワーなどの高レベル処理データは送信しない。

## WSL2からの接続方法

### 前提条件

- Windows 11 22H2以降
- WSL2

### 1. WSL2のmirroredモード有効化

WSL2はデフォルトでNATモードのため、外部（スマホ）から直接アクセスできない。mirroredモードを有効にすると、WindowsとWSL2が同じIPアドレスを共有する。

`%USERPROFILE%\.wslconfig` を編集：

```ini
[wsl2]
networkingMode=mirrored
```

設定後、WSLを再起動：

```powershell
wsl --shutdown
```

### 2. Windowsファイアウォール設定

管理者PowerShellで実行：

```powershell
New-NetFirewallRule -DisplayName "Muse OSC UDP" -Direction Inbound -Protocol UDP -LocalPort 5000 -Action Allow
```

### 3. IPアドレス確認

WSL2内で確認（mirroredモードならWindowsと同じIPになる）：

```bash
hostname -I
# 例: 192.168.100.136
```

### 4. OSCサーバー起動

```bash
source venv/bin/activate
python scripts/osc_receiver.py --port 5000
```

### 5. Muse Appで接続

- IP: WSL2/WindowsのIPアドレス
- Port: 5000
- Streaming Enabled: ON

## サンプルスクリプト

`scripts/osc_receiver.py` - OSC受信テスト用スクリプト

```bash
# 実行方法
source venv/bin/activate
python scripts/osc_receiver.py --port 5000
```

## 注意事項

- Muse AppのOSC機能は公式ドキュメントがほとんどない
- 周波数帯パワー（alpha, theta等）は送信されないため、必要な場合は生EEGから自前で計算する必要がある
- ネットワーク遅延により、リアルタイム性に若干の影響がある可能性

## 参考リンク

- [MuseIO OSC 仕様書](MUSEIO_OSC_SPECIFICATION.md) - 公式SDK（MuseIO）の完全仕様
- [Mind Monitor Technical Manual](https://mind-monitor.com/Technical_Manual.php)
- [MindMonitorPython GitHub](https://github.com/Enigma644/MindMonitorPython)
