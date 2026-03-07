"""
Muse App Prototype v1: Recoveries Chime

Mind Monitor経由でpre-computed band power（AF7, AF8）を受信し、
β/α比からMind State分類 + Recoveries検出 → チャイム音でフィードバック。

使い方:
    source venv/bin/activate
    python src/app/muse_app.py [options]

オプション:
    --ip IP             リッスンIP (default: 0.0.0.0)
    --port PORT         リッスンポート (default: 5000)
    --calibration SEC   キャリブレーション秒数 (default: 60)
    --no-audio          音声無効
"""

from __future__ import annotations

import argparse
import collections
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from enum import Enum
from typing import Deque, List, Optional

import numpy as np
from pythonosc import dispatcher, osc_server

# --------------------------------------------------------------------------- #
# 定数
# --------------------------------------------------------------------------- #

ALPHA_OSC_PATH = '/muse/elements/alpha_absolute'
BETA_OSC_PATH = '/muse/elements/beta_absolute'
HSI_PATH = '/muse/elements/horseshoe'

HSI_GOOD_THRESHOLD = 2
# AF7=index 1, AF8=index 2
FRONTAL_CHANNELS = [1, 2]

# 移動平均ウィンドウ（~10Hz × 4秒）
SMOOTHING_WINDOW = 40

# キャリブレーション
DEFAULT_CALIBRATION_SEC = 60

# Mind State閾値（Z-score）
# Calm = Z ≥ 0 (≈ median)、Active = Z ≤ -1.8 MAD
CALM_Z_THRESHOLD = 0.0
ACTIVE_Z_THRESHOLD = -1.8

# Recoveries検出パラメータ
RECOVERY_PRE_SEC = 3
RECOVERY_POST_SEC = 3
# 相対percentile判定: drop=25%ile以下, rise=55%ile以上（分析レポート準拠）
RECOVERY_DROP_PERCENTILE = 25
RECOVERY_RISE_PERCENTILE = 55
RECOVERY_COOLDOWN_SEC = 30

# 音声フィードバック
AUDIO_SAMPLE_RATE = 44100
AUDIO_DURATION = 0.5
AUDIO_FREQUENCY = 528
AUDIO_COOLDOWN = 30.0

# 連続音階フィードバック (C4〜C5)
SCALE_NOTES = [
    ('C4', 261.63),
    ('D4', 293.66),
    ('E4', 329.63),
    ('F4', 349.23),
    ('G4', 392.00),
    ('A4', 440.00),
    ('B4', 493.88),
    ('C5', 523.25),
]
SCALE_Z_MIN = -1.8   # C4 に対応する Z-score
SCALE_Z_MAX = 2.4    # C5 に対応する Z-score
SCALE_TONE_DURATION = 0.2  # 各トーンの長さ（秒）
SCALE_INTERVAL = 1.0       # トーン再生間隔（秒）

# コンソール更新間隔
CONSOLE_UPDATE_INTERVAL = 1.0


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class AppPhase(Enum):
    CALIBRATING = "CALIBRATING"
    ACTIVE = "ACTIVE"


class MindState(Enum):
    CALM = "Calm"
    NEUTRAL = "Neutral"
    ACTIVE = "Active"


# --------------------------------------------------------------------------- #
# AudioFeedback
# --------------------------------------------------------------------------- #

class AudioFeedback:
    """サイン波WAVを生成してaplayで再生するクラス（クールダウン付き）。"""

    def __init__(
        self,
        frequency: float = AUDIO_FREQUENCY,
        duration: float = AUDIO_DURATION,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        cooldown: float = AUDIO_COOLDOWN,
        enabled: bool = True,
    ):
        self.frequency = frequency
        self.duration = duration
        self.sample_rate = sample_rate
        self.cooldown = cooldown
        self.enabled = enabled
        self._last_played: float = 0.0
        self._lock = threading.Lock()
        self._use_aplay = self._check_aplay()

    def _check_aplay(self) -> bool:
        try:
            result = subprocess.run(
                ['aplay', '--version'],
                capture_output=True, timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _generate_wav(self) -> bytes:
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        fade_samples = int(self.sample_rate * 0.05)
        envelope = np.ones(len(t))
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        wave_data = (np.sin(2 * np.pi * self.frequency * t) * envelope * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(wave_data.tobytes())
        return wav_buf.getvalue()

    def _play_thread(self) -> None:
        try:
            if self._use_aplay:
                wav_bytes = self._generate_wav()
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(wav_bytes)
                    tmp_path = f.name
                try:
                    subprocess.run(
                        ['aplay', '-q', tmp_path],
                        timeout=self.duration + 2,
                        capture_output=True,
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            else:
                sys.stdout.write('\a')
                sys.stdout.flush()
        except Exception as e:
            print(f"[Audio] Error: {e}", file=sys.stderr)

    def play_if_ready(self) -> bool:
        if not self.enabled:
            return False
        now = time.time()
        with self._lock:
            if now - self._last_played < self.cooldown:
                return False
            self._last_played = now
        t = threading.Thread(target=self._play_thread, daemon=True)
        t.start()
        return True


# --------------------------------------------------------------------------- #
# ContinuousAudioFeedback - Z-scoreに応じた音階フィードバック
# --------------------------------------------------------------------------- #

class ContinuousAudioFeedback:
    """Z-scoreを音階にマッピングし、1秒ごとにトーンを再生する。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._use_aplay = AudioFeedback(enabled=False)._use_aplay
        self._lock = threading.Lock()
        self._current_z: Optional[float] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _z_to_note(self, z: float) -> tuple[str, float]:
        """Z-scoreを最も近い音階にスナップする。"""
        z_clamped = max(SCALE_Z_MIN, min(SCALE_Z_MAX, z))
        z_range = SCALE_Z_MAX - SCALE_Z_MIN
        idx = (z_clamped - SCALE_Z_MIN) / z_range * (len(SCALE_NOTES) - 1)
        idx = int(round(idx))
        idx = max(0, min(len(SCALE_NOTES) - 1, idx))
        return SCALE_NOTES[idx]

    def _generate_tone(self, frequency: float) -> bytes:
        """指定周波数の短いトーンをWAVで生成。"""
        duration = SCALE_TONE_DURATION
        t = np.linspace(0, duration, int(AUDIO_SAMPLE_RATE * duration), endpoint=False)
        fade_samples = int(AUDIO_SAMPLE_RATE * 0.02)
        envelope = np.ones(len(t))
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        wave_data = (np.sin(2 * np.pi * frequency * t) * envelope * 32767 * 0.5).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes(wave_data.tobytes())
        return wav_buf.getvalue()

    def _play_tone(self, frequency: float) -> None:
        """トーンを再生。"""
        try:
            if self._use_aplay:
                wav_bytes = self._generate_tone(frequency)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(wav_bytes)
                    tmp_path = f.name
                try:
                    subprocess.run(
                        ['aplay', '-q', tmp_path],
                        timeout=SCALE_TONE_DURATION + 2,
                        capture_output=True,
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            else:
                sys.stdout.write('\a')
                sys.stdout.flush()
        except Exception as e:
            print(f"[Audio] Error: {e}", file=sys.stderr)

    def update_z(self, z: float) -> None:
        """現在のZ-scoreを更新する。"""
        with self._lock:
            self._current_z = z

    def _loop(self) -> None:
        """1秒ごとにトーンを再生するループ。"""
        while self._running:
            with self._lock:
                z = self._current_z
            if z is not None:
                note_name, freq = self._z_to_note(z)
                self._play_tone(freq)
            time.sleep(SCALE_INTERVAL)

    def start(self) -> None:
        if not self.enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False


# --------------------------------------------------------------------------- #
# MuseAppState - メインロジック
# --------------------------------------------------------------------------- #

class MuseAppState:
    """OSCデータの受信、α/β比計算、キャリブレーション、Mind State分類、Recoveries検出。"""

    def __init__(self, calibration_sec: int = DEFAULT_CALIBRATION_SEC,
                 audio: Optional[AudioFeedback] = None,
                 continuous_audio: Optional[ContinuousAudioFeedback] = None):
        self._lock = threading.Lock()
        self.calibration_sec = calibration_sec
        self.audio = audio
        self.continuous_audio = continuous_audio

        # 最新のOSC値（4ch）
        self._alpha_raw: Optional[List[float]] = None
        self._beta_raw: Optional[List[float]] = None
        self._hsi: Optional[List[float]] = None

        # 移動平均
        self._ratio_buffer: Deque[float] = collections.deque(maxlen=SMOOTHING_WINDOW)

        # キャリブレーション (median/MAD)
        self.phase = AppPhase.CALIBRATING
        self._calib_start: Optional[float] = None
        self._calib_ratios: List[float] = []
        self.calib_median: float = 0.0
        self.calib_mad: float = 1.0

        # Z-score履歴（~10Hz）— Recoveries検出用（直近6秒 + percentile計算用に余裕を持つ）
        self._z_history: Deque[float] = collections.deque(maxlen=int(10 * 120))  # 直近120秒

        # Mind State
        self.current_state = MindState.NEUTRAL
        self.current_z: float = 0.0
        self.current_ratio: float = 0.0

        # Recoveries
        self.recovery_count: int = 0
        self._last_recovery_time: float = 0.0

        # セッション統計
        self._session_start: Optional[float] = None
        self._state_durations = {MindState.CALM: 0.0, MindState.NEUTRAL: 0.0, MindState.ACTIVE: 0.0}
        self._last_state_time: Optional[float] = None

        # コンソール制御
        self._last_console_update: float = 0.0

    def handle_alpha(self, address: str, *args: float) -> None:
        with self._lock:
            self._alpha_raw = [float(a) for a in args[:4]]
        self._process()

    def handle_beta(self, address: str, *args: float) -> None:
        with self._lock:
            self._beta_raw = [float(a) for a in args[:4]]

    def handle_hsi(self, address: str, *args: float) -> None:
        with self._lock:
            self._hsi = [float(a) for a in args[:4]]

    def _get_valid_frontal_values(self, raw: List[float]) -> List[float]:
        """HSIフィルタを通してAF7, AF8の有効値を返す。"""
        hsi = self._hsi
        values = []
        for ch in FRONTAL_CHANNELS:
            if ch < len(raw):
                val = raw[ch]
                if val != val:  # NaN check
                    continue
                if hsi is not None and len(hsi) > ch and hsi[ch] > HSI_GOOD_THRESHOLD:
                    continue
                values.append(val)
        return values

    def _process(self) -> None:
        with self._lock:
            if self._alpha_raw is None or self._beta_raw is None:
                return

            alpha_vals = self._get_valid_frontal_values(self._alpha_raw)
            beta_vals = self._get_valid_frontal_values(self._beta_raw)

            if not alpha_vals or not beta_vals:
                return

            # Bels → linear → ratio
            alpha_linear = np.mean([10 ** v for v in alpha_vals])
            beta_linear = np.mean([10 ** v for v in beta_vals])

            if beta_linear <= 0:
                return

            ratio = alpha_linear / beta_linear
            self._ratio_buffer.append(ratio)

            if len(self._ratio_buffer) < 5:
                return

            smoothed = float(np.mean(self._ratio_buffer))
            now = time.time()

            # --- キャリブレーション ---
            if self.phase == AppPhase.CALIBRATING:
                if self._calib_start is None:
                    self._calib_start = now
                    print(f"\n[CALIBRATING] Started. Collecting data for {self.calibration_sec}s...")

                elapsed = now - self._calib_start
                self._calib_ratios.append(smoothed)

                # 進捗表示（5秒ごと）
                if now - self._last_console_update >= 5.0:
                    self._last_console_update = now
                    print(f"[CALIBRATING] {int(elapsed)}/{self.calibration_sec}s  "
                          f"ratio={smoothed:.2f}  samples={len(self._calib_ratios)}")

                if elapsed >= self.calibration_sec:
                    arr = np.array(self._calib_ratios)
                    self.calib_median = float(np.median(arr))
                    self.calib_mad = float(np.median(np.abs(arr - self.calib_median)))
                    if self.calib_mad < 1e-6:
                        self.calib_mad = 1.0
                    self.phase = AppPhase.ACTIVE
                    self._session_start = now
                    self._last_state_time = now
                    print(f"\n[CALIBRATION COMPLETE] median={self.calib_median:.3f}  MAD={self.calib_mad:.3f}")
                    print("=" * 50)
                    print("Mind State tracking started!")
                    print("=" * 50)
                    if self.continuous_audio:
                        self.continuous_audio.start()
                return

            # --- アクティブフェーズ ---
            z = (smoothed - self.calib_median) / self.calib_mad
            self.current_z = z
            self.current_ratio = smoothed

            # Mind State分類
            if z >= CALM_Z_THRESHOLD:
                new_state = MindState.CALM
            elif z <= ACTIVE_Z_THRESHOLD:
                new_state = MindState.ACTIVE
            else:
                new_state = MindState.NEUTRAL

            # 滞在時間の更新
            if self._last_state_time is not None:
                dt = now - self._last_state_time
                self._state_durations[self.current_state] += dt
            self._last_state_time = now
            self.current_state = new_state

            # Z-score履歴に追加
            self._z_history.append(z)

            # 連続音階フィードバックにZ-scoreを送信
            if self.continuous_audio:
                self.continuous_audio.update_z(z)

            # Recoveries検出
            self._check_recovery(now)

            # コンソール出力
            if now - self._last_console_update >= CONSOLE_UPDATE_INTERVAL:
                self._last_console_update = now
                ts = datetime.now().strftime('%H:%M:%S')
                state_str = f"{new_state.value:7s}"
                note_str = ""
                if self.continuous_audio and self.continuous_audio.enabled:
                    note_name, _ = self.continuous_audio._z_to_note(z)
                    note_str = f"  {note_name}"
                print(f"[{ts}] {state_str}  Z:{z:+.2f}  Ratio:{smoothed:.2f}{note_str}")

    def _check_recovery(self, now: float) -> None:
        """Recoveries検出: セッション内Z-score分布に対する相対percentileで判定。"""
        # 最小間隔チェック
        if now - self._last_recovery_time < RECOVERY_COOLDOWN_SEC:
            return

        # Z-score履歴が十分か（最低30秒分のデータ）
        samples_per_sec = 10
        pre_samples = RECOVERY_PRE_SEC * samples_per_sec
        post_samples = RECOVERY_POST_SEC * samples_per_sec
        total_needed = pre_samples + post_samples
        min_for_percentile = 30 * samples_per_sec

        if len(self._z_history) < max(total_needed, min_for_percentile):
            return

        z_list = list(self._z_history)

        # セッション内Z-score分布からpercentile閾値を算出
        drop_threshold = float(np.percentile(z_list, RECOVERY_DROP_PERCENTILE))
        rise_threshold = float(np.percentile(z_list, RECOVERY_RISE_PERCENTILE))

        # 「後3秒」= 直近のpost_samples、「前3秒」= その直前のpre_samples
        post_z = z_list[-post_samples:]
        pre_z = z_list[-(pre_samples + post_samples):-post_samples]

        pre_mean = float(np.mean(pre_z))
        post_mean = float(np.mean(post_z))

        if pre_mean <= drop_threshold and post_mean >= rise_threshold and post_mean >= CALM_Z_THRESHOLD:
            self.recovery_count += 1
            self._last_recovery_time = now
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"\n[{ts}] *** RECOVERY #{self.recovery_count} ***  "
                  f"pre_z={pre_mean:.2f} → post_z={post_mean:.2f}  "
                  f"(drop<={drop_threshold:.2f}, rise>={rise_threshold:.2f})")
            if self.audio:
                self.audio.play_if_ready()

    def print_summary(self) -> None:
        """セッション終了時のサマリー表示。"""
        # 最後のstateの滞在時間を加算
        if self._last_state_time is not None:
            dt = time.time() - self._last_state_time
            self._state_durations[self.current_state] += dt

        if self._session_start is None:
            print("\nNo active session data.")
            return

        total_sec = time.time() - self._session_start

        def fmt_duration(sec: float) -> str:
            m, s = divmod(int(sec), 60)
            return f"{m:2d}m {s:02d}s"

        calm_sec = self._state_durations[MindState.CALM]
        neutral_sec = self._state_durations[MindState.NEUTRAL]
        active_sec = self._state_durations[MindState.ACTIVE]
        calm_pct = (calm_sec / total_sec * 100) if total_sec > 0 else 0

        print()
        print("=" * 40)
        print("       Session Summary")
        print("=" * 40)
        print(f"Duration:    {fmt_duration(total_sec)}")
        print(f"Calm:        {fmt_duration(calm_sec)} ({calm_pct:.0f}%)")
        print(f"Neutral:     {fmt_duration(neutral_sec)}")
        print(f"Active:      {fmt_duration(active_sec)}")
        print(f"Recoveries:  {self.recovery_count}")
        print("=" * 40)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Muse App Prototype v1: Recoveries Chime',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--ip', default='0.0.0.0', help='Listen IP (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Listen port (default: 5000)')
    parser.add_argument('--calibration', type=int, default=DEFAULT_CALIBRATION_SEC,
                        help=f'Calibration seconds (default: {DEFAULT_CALIBRATION_SEC})')
    parser.add_argument('--no-audio', action='store_true', help='Disable audio feedback')
    parser.add_argument('--feedback', choices=['recovery', 'continuous'], default='recovery',
                        help='Feedback mode: recovery (chime on recovery) or continuous (scale tones)')
    args = parser.parse_args()

    audio = None
    continuous_audio = None
    if not args.no_audio:
        if args.feedback == 'recovery':
            audio = AudioFeedback(enabled=True)
        else:
            continuous_audio = ContinuousAudioFeedback(enabled=True)

    state = MuseAppState(calibration_sec=args.calibration, audio=audio, continuous_audio=continuous_audio)

    # OSC Dispatcher
    disp = dispatcher.Dispatcher()
    disp.map(ALPHA_OSC_PATH, state.handle_alpha)
    disp.map(BETA_OSC_PATH, state.handle_beta)
    disp.map(HSI_PATH, state.handle_hsi)

    server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), disp)

    print("=" * 50)
    print("  Muse App Prototype v1: Recoveries Chime")
    print("=" * 50)
    feedback_str = 'disabled' if args.no_audio else args.feedback
    print(f"Listen:       {args.ip}:{args.port}")
    print(f"Calibration:  {args.calibration}s")
    print(f"Feedback:     {feedback_str}")
    print()
    print("Mind Monitor settings:")
    print(f"  Host: <your PC IP>  Port: {args.port}")
    print()
    print("Waiting for data... (Ctrl+C to exit)")
    print("=" * 50)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if continuous_audio:
            continuous_audio.stop()
        state.print_summary()
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()
