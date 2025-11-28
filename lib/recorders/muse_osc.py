"""
Muse App OSC Recorder - Mind Monitor互換CSV出力

Muse AppのOSC Output機能からデータを受信し、
Mind Monitor互換のCSV形式で保存する。

⚠️ 保留中: この機能は開発を一時中断しています。
既知の問題:
- バンドパワー列（Delta/Theta/Alpha/Beta/Gamma）が空のまま出力される
  → Muse AppはRAW EEGのみ送信、FFT計算が必要
- Opticsデータの送信頻度が低い（8.6Hz、期待値64Hz）
  → fNIRSグラフが平行線になる
- generate_report.pyはMind Monitorの事前計算されたバンドパワー列を期待している
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from pythonosc import dispatcher, osc_server

# Mind Monitor CSV列順序（59列）
MIND_MONITOR_COLUMNS = [
    'TimeStamp',
    # 周波数帯パワー (20列) - 空欄
    'Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
    'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
    'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
    'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10',
    'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10',
    # RAW EEG (4列)
    'RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10',
    # AUX (4列) - 空欄
    'AUX_1', 'AUX_2', 'AUX_3', 'AUX_4',
    # センサー (6列)
    'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z',
    'Gyro_X', 'Gyro_Y', 'Gyro_Z',
    # Optics (16列)
    'Optics1', 'Optics2', 'Optics3', 'Optics4',
    'Optics5', 'Optics6', 'Optics7', 'Optics8',
    'Optics9', 'Optics10', 'Optics11', 'Optics12',
    'Optics13', 'Optics14', 'Optics15', 'Optics16',
    # その他
    'Heart_Rate',
    'HeadBandOn',
    'HSI_TP9', 'HSI_AF7', 'HSI_AF8', 'HSI_TP10',
    'Battery',
    'Elements',
]


class MuseOSCRecorder:
    """
    Muse App OSC受信 → Mind Monitor互換CSV出力

    Muse AppのOSCパス:
    - /eeg: (TP9, AF7, AF8, TP10) float×4 @ 256Hz
    - /acc: (x, y, z) float×3 @ 52Hz
    - /gyro: (x, y, z) float×3 @ 52Hz
    - /optics: float×8 @ 64Hz
    """

    def __init__(self, output_dir: Path):
        """
        Parameters
        ----------
        output_dir : Path
            CSV出力先ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rows: list[dict] = []
        self.start_time: datetime | None = None

        # 低頻度データの直前値（EEG受信時に使用）
        self.last_acc = [0.0, 0.0, 0.0]
        self.last_gyro = [0.0, 0.0, 0.0]
        self.last_optics = [0.0] * 8

        self._server = None

    def on_eeg(self, address: str, *args) -> None:
        """
        EEGデータ受信ハンドラー（256Hz）

        EEGは最高頻度なので、これをトリガーに1行作成する。
        """
        if self.start_time is None:
            self.start_time = datetime.now()

        row = self._create_row(args)
        self.rows.append(row)

    def on_acc(self, address: str, *args) -> None:
        """加速度データ受信ハンドラー（52Hz）"""
        self.last_acc = list(args[:3])

    def on_gyro(self, address: str, *args) -> None:
        """ジャイロデータ受信ハンドラー（52Hz）"""
        self.last_gyro = list(args[:3])

    def on_optics(self, address: str, *args) -> None:
        """Opticsデータ受信ハンドラー（64Hz）"""
        self.last_optics = list(args[:8])

    def _create_row(self, eeg_args: tuple) -> dict:
        """
        Mind Monitor互換の行データを作成

        Parameters
        ----------
        eeg_args : tuple
            (TP9, AF7, AF8, TP10) のEEG値

        Returns
        -------
        dict
            Mind Monitor CSV互換の行データ
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        row = {
            'TimeStamp': timestamp,
            # RAW EEG
            'RAW_TP9': eeg_args[0],
            'RAW_AF7': eeg_args[1],
            'RAW_AF8': eeg_args[2],
            'RAW_TP10': eeg_args[3],
            # Accelerometer
            'Accelerometer_X': self.last_acc[0],
            'Accelerometer_Y': self.last_acc[1],
            'Accelerometer_Z': self.last_acc[2],
            # Gyro
            'Gyro_X': self.last_gyro[0],
            'Gyro_Y': self.last_gyro[1],
            'Gyro_Z': self.last_gyro[2],
            # Optics (8チャンネル)
            'Optics1': self.last_optics[0],
            'Optics2': self.last_optics[1],
            'Optics3': self.last_optics[2],
            'Optics4': self.last_optics[3],
            'Optics5': self.last_optics[4],
            'Optics6': self.last_optics[5],
            'Optics7': self.last_optics[6],
            'Optics8': self.last_optics[7],
            # 固定値
            'HeadBandOn': 1,
        }

        return row

    def save(self) -> Path | None:
        """
        バッファをCSVファイルに保存

        Returns
        -------
        Path or None
            保存したファイルパス。データがない場合はNone。
        """
        if not self.rows:
            return None

        if self.start_time is None:
            self.start_time = datetime.now()

        # DataFrameを作成し、Mind Monitor列順序に合わせる
        df = pd.DataFrame(self.rows)

        # 存在しない列を空欄で追加
        for col in MIND_MONITOR_COLUMNS:
            if col not in df.columns:
                df[col] = ''

        # 列順序を Mind Monitor 形式に合わせる
        df = df[MIND_MONITOR_COLUMNS]

        # ファイル名はMind Monitor形式に近づける
        filename = f"muse_app_{self.start_time:%Y-%m-%d--%H-%M-%S}.csv"
        path = self.output_dir / filename
        df.to_csv(path, index=False)

        return path

    def start_server(self, ip: str = '0.0.0.0', port: int = 5000) -> None:
        """
        OSCサーバーを起動

        Parameters
        ----------
        ip : str
            リッスンするIPアドレス（デフォルト: 0.0.0.0 = 全インターフェース）
        port : int
            リッスンするポート（デフォルト: 5000）
        """
        disp = dispatcher.Dispatcher()

        # Muse App OSCパス（/muse/プレフィックスなし）
        disp.map("/eeg", self.on_eeg)
        disp.map("/acc", self.on_acc)
        disp.map("/gyro", self.on_gyro)
        disp.map("/optics", self.on_optics)

        self._server = osc_server.ThreadingOSCUDPServer((ip, port), disp)

        print("=" * 60)
        print("Muse App OSC Recorder")
        print("=" * 60)
        print(f"Listening on {ip}:{port}")
        print()
        print("Muse Appの設定:")
        print(f"  - IP: このPCのIPアドレス")
        print(f"  - Port: {port}")
        print(f"  - Streaming Enabled: ON")
        print()
        print("Ctrl+C で記録終了・CSV保存")
        print("=" * 60)

        self._server.serve_forever()

    @property
    def record_count(self) -> int:
        """記録済みサンプル数"""
        return len(self.rows)
