"""
Muse OSC Recorder - Muse App / Mind Monitor 両対応

--source パラメータで切り替え:
- muse_app_osc: Muse App の OSC Output（RAW EEG のみ、バンドパワーは後計算）
- mind_monitor_osc_osc: Mind Monitor の OSC（バンドパワー, HSI, Battery, Elements 含む）
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from pythonosc import dispatcher, osc_server

# Mind Monitor CSV列順序（59列）
MIND_MONITOR_COLUMNS = [
    'TimeStamp',
    # 周波数帯パワー (20列)
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

BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
SENSOR_NAMES = ['TP9', 'AF7', 'AF8', 'TP10']


class MuseOSCRecorder:
    """
    Muse App / Mind Monitor 両対応 OSC レコーダー

    source='muse_app_osc':
        /eeg, /acc, /gyro, /optics を受信
    source='mind_monitor_osc':
        /muse/eeg, /muse/acc, /muse/gyro + band power, HSI, battery, elements を受信
    """

    def __init__(self, output_dir: Path, source: str = 'muse_app_osc'):
        self.source = source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rows: list[dict] = []
        self.start_time: datetime | None = None

        # 低頻度データの直前値（EEG受信時に使用）
        self.last_acc = [0.0, 0.0, 0.0]
        self.last_gyro = [0.0, 0.0, 0.0]
        self.last_optics = [0.0] * 8

        # Mind Monitor 固有データ
        self.last_band_power: dict[str, list[float]] = {
            band: [0.0] * 4 for band in BAND_NAMES
        }
        self.last_hsi = [0.0] * 4
        self.last_battery = 0.0
        self.last_elements = ''

        self._server = None

    def on_eeg(self, address: str, *args) -> None:
        """EEGデータ受信ハンドラー（256Hz）"""
        if self.start_time is None:
            self.start_time = datetime.now()

        row = self._create_row(args)
        self.rows.append(row)

        count = len(self.rows)
        if count % 256 == 0:
            elapsed = (datetime.now() - self.start_time).seconds
            m, s = divmod(elapsed, 60)
            print(f"[{m:02d}:{s:02d}] {count} samples | EEG: TP9={args[0]:.1f}, AF7={args[1]:.1f}, AF8={args[2]:.1f}, TP10={args[3]:.1f}")

    def on_acc(self, address: str, *args) -> None:
        """加速度データ受信ハンドラー"""
        self.last_acc = list(args[:3])

    def on_gyro(self, address: str, *args) -> None:
        """ジャイロデータ受信ハンドラー"""
        self.last_gyro = list(args[:3])

    def on_optics(self, address: str, *args) -> None:
        """Opticsデータ受信ハンドラー"""
        self.last_optics = list(args[:8])

    # --- Mind Monitor 固有ハンドラー ---

    def on_band_power(self, address: str, *args) -> None:
        """バンドパワー受信ハンドラー（/muse/elements/delta_absolute 等）"""
        # address: /muse/elements/{band}_absolute
        band = address.split('/')[-1].replace('_absolute', '')
        if band in self.last_band_power:
            self.last_band_power[band] = list(args[:4])

    def on_horseshoe(self, address: str, *args) -> None:
        """HSI（Horseshoe Indicator）受信ハンドラー"""
        self.last_hsi = list(args[:4])

    def on_battery(self, address: str, *args) -> None:
        """バッテリー受信ハンドラー"""
        if args:
            self.last_battery = args[0]

    def on_blink(self, address: str, *args) -> None:
        """Blink イベント受信ハンドラー"""
        self.last_elements = 'blink'

    def on_jaw_clench(self, address: str, *args) -> None:
        """Jaw Clench イベント受信ハンドラー"""
        self.last_elements = 'jaw_clench'

    def _create_row(self, eeg_args: tuple) -> dict:
        """Mind Monitor互換の行データを作成"""
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
            # 固定値
            'HeadBandOn': 1,
        }

        if self.source == 'muse_app_osc':
            # Optics (8チャンネル)
            for i in range(8):
                row[f'Optics{i + 1}'] = self.last_optics[i]
        elif self.source == 'mind_monitor_osc':
            # Band Power (20列)
            for band in BAND_NAMES:
                for i, sensor in enumerate(SENSOR_NAMES):
                    col = f'{band.capitalize()}_{sensor}'
                    row[col] = self.last_band_power[band][i]
            # HSI
            for i, sensor in enumerate(SENSOR_NAMES):
                row[f'HSI_{sensor}'] = self.last_hsi[i]
            # Battery & Elements
            row['Battery'] = self.last_battery
            row['Elements'] = self.last_elements
            # Elements はイベントなので1回記録したらクリア
            self.last_elements = ''

        return row

    def save(self) -> Path | None:
        """バッファをCSVファイルに保存"""
        if not self.rows:
            return None

        if self.start_time is None:
            self.start_time = datetime.now()

        df = pd.DataFrame(self.rows)

        # 存在しない列を空欄で追加
        for col in MIND_MONITOR_COLUMNS:
            if col not in df.columns:
                df[col] = ''

        df = df[MIND_MONITOR_COLUMNS]

        filename = f"{self.source}_{self.start_time:%Y-%m-%d--%H-%M-%S}.csv"
        path = self.output_dir / filename
        df.to_csv(path, index=False)

        return path

    def start_server(self, ip: str = '0.0.0.0', port: int = 5000) -> None:
        """OSCサーバーを起動"""
        disp = dispatcher.Dispatcher()

        if self.source == 'muse_app_osc':
            disp.map("/eeg", self.on_eeg)
            disp.map("/acc", self.on_acc)
            disp.map("/gyro", self.on_gyro)
            disp.map("/optics", self.on_optics)
        elif self.source == 'mind_monitor_osc':
            disp.map("/muse/eeg", self.on_eeg)
            disp.map("/muse/acc", self.on_acc)
            disp.map("/muse/gyro", self.on_gyro)
            # Band Power
            for band in BAND_NAMES:
                disp.map(f"/muse/elements/{band}_absolute", self.on_band_power)
            # HSI, Battery, Elements
            disp.map("/muse/elements/horseshoe", self.on_horseshoe)
            disp.map("/muse/elements/blink", self.on_blink)
            disp.map("/muse/elements/jaw_clench", self.on_jaw_clench)
            disp.map("/muse/batt", self.on_battery)

        self._server = osc_server.ThreadingOSCUDPServer((ip, port), disp)

        source_label = 'Muse App' if self.source == 'muse_app_osc' else 'Mind Monitor'
        print("=" * 60)
        print(f"Muse OSC Recorder ({source_label})")
        print("=" * 60)
        print(f"Listening on {ip}:{port}")
        print()
        if self.source == 'muse_app_osc':
            print("Muse Appの設定:")
            print(f"  - IP: このPCのIPアドレス")
            print(f"  - Port: {port}")
            print(f"  - Streaming Enabled: ON")
        else:
            print("Mind Monitorの設定:")
            print(f"  - OSC Stream Target IP: このPCのIPアドレス")
            print(f"  - OSC Stream Target Port: {port}")
        print()
        print("Ctrl+C で記録終了・CSV保存")
        print("=" * 60)

        self._server.serve_forever()

    @property
    def record_count(self) -> int:
        """記録済みサンプル数"""
        return len(self.rows)
