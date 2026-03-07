"""
Alpha Meditation Neurofeedback - リアルタイム・ニューロフィードバック

Muse S + Mind Monitor からOSCでpre-computed帯域パワーを受信し、
Alpha相対パワーが高い（リラックス状態）ときに音声・視覚フィードバックを返す。

使い方:
    source venv/bin/activate
    python src/app/alpha_neurofeedback.py [options]

オプション:
    --ip IP           リッスンIP (default: 0.0.0.0)
    --port PORT       リッスンポート (default: 5000)
    --threshold T     Alpha相対パワー閾値 (default: 0.35)
    --no-audio        音声フィードバック無効
    --no-visual       視覚フィードバック無効（コンソール出力のみ）
    --history SEC     履歴ウィンドウ秒数 (default: 120)
"""

from __future__ import annotations

import argparse
import array
import collections
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# matplotlib はオプション（--no-visual 時は不要）
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from pythonosc import dispatcher, osc_server

# プロジェクトルートをパスに追加
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from lib.sensors.eeg.alpha_power import DEFAULT_PARAMS, AlphaPowerMethod


# --------------------------------------------------------------------------- #
# 定数
# --------------------------------------------------------------------------- #

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_COLORS = {
    'delta': '#8B4513',
    'theta': '#4169E1',
    'alpha': '#228B22',
    'beta': '#FF8C00',
    'gamma': '#DC143C',
}
BAND_OSC_PATHS = {
    'delta': '/muse/elements/delta_absolute',
    'theta': '/muse/elements/theta_absolute',
    'alpha': '/muse/elements/alpha_absolute',
    'beta':  '/muse/elements/beta_absolute',
    'gamma': '/muse/elements/gamma_absolute',
}
HSI_PATH = '/muse/elements/horseshoe'
HSI_GOOD_THRESHOLD = 2  # HSI ≤ 2 のチャネルを有効とみなす

# Brain Recharge Score 計算パラメータ
_LINEAR_PARAMS = DEFAULT_PARAMS[AlphaPowerMethod.LINEAR]
BRS_SLOPE = _LINEAR_PARAMS['slope']        # 5.07
BRS_INTERCEPT = _LINEAR_PARAMS['intercept'] # 36.9

# 音声フィードバック設定
AUDIO_SAMPLE_RATE = 44100
AUDIO_DURATION = 0.5   # 秒
AUDIO_FREQUENCY = 528  # Hz (Alpha瞑想に使われる528Hzソルフェジオ)
AUDIO_COOLDOWN = 3.0   # 秒（連続再生防止）

NUM_CHANNELS = 4  # TP9, AF7, AF8, TP10


# --------------------------------------------------------------------------- #
# NeurofeedbackState - スレッドセーフな共有状態
# --------------------------------------------------------------------------- #

class NeurofeedbackState:
    """スレッドセーフな共有状態管理クラス。"""

    def __init__(self, history_sec: int = 120, sample_rate: float = 10.0):
        self._lock = threading.Lock()
        max_len = int(history_sec * sample_rate)

        # 各帯域のBels値（チャネル平均）
        self.abs_bands: Dict[str, Optional[float]] = {b: None for b in BANDS}

        # HSI品質 (list of 4 values: TP9, AF7, AF8, TP10)
        self.hsi: Optional[List[float]] = None

        # 時系列履歴
        self.alpha_rel_history: Deque[float] = collections.deque(maxlen=max_len)
        self.brs_history: Deque[float] = collections.deque(maxlen=max_len)
        self.timestamps: Deque[float] = collections.deque(maxlen=max_len)

        # 各帯域の時系列（視覚化用）
        self.band_history: Dict[str, Deque[float]] = {
            b: collections.deque(maxlen=max_len) for b in BANDS
        }

        # 現在値
        self.current_alpha_rel: Optional[float] = None
        self.current_brs: Optional[float] = None
        self.last_received: Optional[float] = None

    def update_band(self, band: str, values: Tuple[float, ...]) -> None:
        with self._lock:
            valid = [v for v in values if not (v != v)]  # NaN除去
            if valid:
                self.abs_bands[band] = float(np.mean(valid))
                self.band_history[band].append(self.abs_bands[band])
            self.last_received = time.time()

    def update_hsi(self, values: Tuple[float, ...]) -> None:
        with self._lock:
            self.hsi = list(values)

    def compute_and_record(self) -> Optional[Tuple[float, float]]:
        """Alpha相対パワーとBRSを計算して履歴に追記。(alpha_rel, brs) を返す。"""
        with self._lock:
            # HSIフィルタリング: 良好なチャネルのみ使用
            # Mind Monitor の各帯域値は [TP9, AF7, AF8, TP10] の順
            hsi = self.hsi

            # 全帯域が揃っているか確認
            if any(self.abs_bands[b] is None for b in BANDS):
                return None

            # HSI考慮: チャネルマスクを作成
            if hsi is not None and len(hsi) == NUM_CHANNELS:
                channel_mask = [h <= HSI_GOOD_THRESHOLD for h in hsi]
                good_channels = sum(channel_mask)
                if good_channels == 0:
                    # 全チャネル不良 → 全チャネル使用（フォールバック）
                    channel_mask = [True] * NUM_CHANNELS
            else:
                channel_mask = [True] * NUM_CHANNELS

            # 相対Alpha計算: 10^alpha / Σ(10^band)
            # Mind Monitor は Bels 単位なので 10^bels = linear power
            band_powers: Dict[str, float] = {}
            for b in BANDS:
                band_powers[b] = 10 ** self.abs_bands[b]

            total_power = sum(band_powers.values())
            if total_power <= 0:
                return None

            alpha_rel = band_powers['alpha'] / total_power

            # Brain Recharge Score = slope × (alpha_bels × 10) + intercept
            alpha_db = self.abs_bands['alpha'] * 10  # Bels → dB
            brs = BRS_SLOPE * alpha_db + BRS_INTERCEPT

            # 履歴に追記
            now = time.time()
            self.alpha_rel_history.append(alpha_rel)
            self.brs_history.append(brs)
            self.timestamps.append(now)
            self.current_alpha_rel = alpha_rel
            self.current_brs = brs

            return alpha_rel, brs

    def get_snapshot(self) -> dict:
        """現在の状態のスナップショットを取得（表示用）。"""
        with self._lock:
            return {
                'alpha_rel': self.current_alpha_rel,
                'brs': self.current_brs,
                'hsi': self.hsi[:] if self.hsi else None,
                'abs_bands': dict(self.abs_bands),
                'alpha_rel_history': list(self.alpha_rel_history),
                'brs_history': list(self.brs_history),
                'timestamps': list(self.timestamps),
                'band_history': {b: list(v) for b, v in self.band_history.items()},
            }


# --------------------------------------------------------------------------- #
# AlphaScorer - スコアリング（計算ロジックのラッパー）
# --------------------------------------------------------------------------- #

class AlphaScorer:
    """Alpha相対パワーとBrain Recharge Scoreの計算を担うクラス。"""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def is_above_threshold(self, alpha_rel: float) -> bool:
        return alpha_rel >= self.threshold

    def score_label(self, brs: float) -> str:
        if brs >= 80:
            return "Excellent"
        elif brs >= 65:
            return "Good"
        elif brs >= 50:
            return "Fair"
        else:
            return "Low"


# --------------------------------------------------------------------------- #
# AudioFeedback - WAV生成 + aplay再生
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

        # aplay が使えるか確認
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
        """サイン波WAVデータを生成してbytesで返す。"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        # フェードイン・フェードアウト（クリックノイズ防止）
        fade_samples = int(self.sample_rate * 0.05)
        envelope = np.ones(len(t))
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        wave_data = (np.sin(2 * np.pi * self.frequency * t) * envelope * 32767).astype(np.int16)

        buf = wave_data.tobytes()
        import io
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(buf)
        return wav_buf.getvalue()

    def _play_thread(self) -> None:
        """バックグラウンドで音声再生。"""
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
                # フォールバック: terminal bell
                sys.stdout.write('\a')
                sys.stdout.flush()
        except Exception as e:
            print(f"[Audio] 再生エラー: {e}", file=sys.stderr)

    def play_if_ready(self) -> bool:
        """クールダウンが経過していれば再生。再生した場合 True を返す。"""
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
# VisualFeedback - matplotlib 4パネル表示
# --------------------------------------------------------------------------- #

class VisualFeedback:
    """matplotlib FuncAnimation による4パネルのリアルタイム表示。"""

    def __init__(
        self,
        state: NeurofeedbackState,
        scorer: AlphaScorer,
        update_interval_ms: int = 200,
    ):
        self.state = state
        self.scorer = scorer
        self.update_interval_ms = update_interval_ms

        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Alpha Meditation Neurofeedback', fontsize=14, fontweight='bold')
        self.fig.patch.set_facecolor('#1a1a2e')

        for ax in self.axes.flat:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#4a4a8a')

        ax_alpha, ax_brs, ax_bands, ax_status = self.axes.flat

        # 左上: Alpha相対パワー時系列
        ax_alpha.set_title('Alpha Relative Power')
        ax_alpha.set_ylabel('Relative Power')
        ax_alpha.set_ylim(0, 1)
        self._line_alpha, = ax_alpha.plot([], [], color='#00ff88', linewidth=1.5)
        self._thresh_line = ax_alpha.axhline(
            y=scorer.threshold, color='#ff6b6b', linestyle='--', alpha=0.8,
            label=f'Threshold ({scorer.threshold:.2f})'
        )
        ax_alpha.legend(loc='upper right', facecolor='#16213e', labelcolor='white', fontsize=8)

        # 右上: Brain Recharge Score時系列
        ax_brs.set_title('Brain Recharge Score')
        ax_brs.set_ylabel('Score')
        ax_brs.set_ylim(0, 100)
        self._line_brs, = ax_brs.plot([], [], color='#4fc3f7', linewidth=1.5)

        # 左下: 5帯域パワー
        ax_bands.set_title('Band Powers (Bels)')
        ax_bands.set_ylabel('Power (Bels)')
        self._band_lines = {}
        for band in BANDS:
            line, = ax_bands.plot([], [], color=BAND_COLORS[band], linewidth=1.2, label=band)
            self._band_lines[band] = line
        ax_bands.legend(loc='upper right', facecolor='#16213e', labelcolor='white', fontsize=8)

        # 右下: HSI品質インジケーター + 現在スコア
        ax_status.set_title('Status')
        ax_status.axis('off')
        self._status_text = ax_status.text(
            0.5, 0.5, 'Waiting for data...',
            transform=ax_status.transAxes,
            ha='center', va='center',
            fontsize=14, color='white',
            fontfamily='monospace',
        )

        plt.tight_layout()

    def _make_time_axis(self, timestamps: List[float]) -> List[float]:
        """タイムスタンプをリスト → 相対秒数に変換。"""
        if not timestamps:
            return []
        t0 = timestamps[0]
        return [t - t0 for t in timestamps]

    def _hsi_label(self, hsi: Optional[List[float]]) -> str:
        if hsi is None:
            return "HSI: N/A"
        ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
        labels = []
        for name, val in zip(ch_names, hsi):
            if val <= 1:
                symbol = '●'
                color_code = 'G'
            elif val <= 2:
                symbol = '◐'
                color_code = 'Y'
            else:
                symbol = '○'
                color_code = 'R'
            labels.append(f"{name}:{symbol}")
        return "  ".join(labels)

    def update(self, frame: int) -> list:
        snap = self.state.get_snapshot()
        timestamps = snap['timestamps']
        rel_times = self._make_time_axis(timestamps)

        ax_alpha, ax_brs, ax_bands, ax_status = self.axes.flat

        # Alpha相対パワー
        alpha_hist = snap['alpha_rel_history']
        if rel_times and alpha_hist:
            self._line_alpha.set_data(rel_times, alpha_hist)
            ax_alpha.set_xlim(max(0, rel_times[-1] - 60), rel_times[-1] + 1)

        # BRS
        brs_hist = snap['brs_history']
        if rel_times and brs_hist:
            self._line_brs.set_data(rel_times, brs_hist)
            ax_brs.set_xlim(max(0, rel_times[-1] - 60), rel_times[-1] + 1)

        # 帯域パワー
        band_hist = snap['band_history']
        any_band = False
        for band in BANDS:
            bh = band_hist[band]
            if bh:
                n = len(bh)
                band_times = rel_times[-n:] if rel_times else list(range(n))
                self._band_lines[band].set_data(band_times, bh)
                any_band = True
        if any_band and rel_times:
            ax_bands.relim()
            ax_bands.autoscale_view()
            ax_bands.set_xlim(max(0, rel_times[-1] - 60), rel_times[-1] + 1)

        # ステータス
        alpha_rel = snap['alpha_rel']
        brs = snap['brs']
        hsi = snap['hsi']

        if alpha_rel is not None and brs is not None:
            above = self.scorer.is_above_threshold(alpha_rel)
            label = self.scorer.score_label(brs)
            status_color = '#00ff88' if above else '#aaaaaa'
            status_str = (
                f"Alpha: {alpha_rel:.3f}\n"
                f"BRS:   {brs:.1f}\n"
                f"State: {label}\n\n"
                f"{self._hsi_label(hsi)}"
            )
            if above:
                status_str += "\n\n★ ALPHA ACTIVE ★"
        else:
            status_color = '#888888'
            status_str = f"Waiting for data...\n{self._hsi_label(hsi)}"

        self._status_text.set_text(status_str)
        self._status_text.set_color(status_color)

        return [self._line_alpha, self._line_brs] + list(self._band_lines.values()) + [self._status_text]

    def start(self) -> None:
        self._anim = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=self.update_interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


# --------------------------------------------------------------------------- #
# OSCハンドラー
# --------------------------------------------------------------------------- #

def make_band_handler(band: str, state: NeurofeedbackState, scorer: AlphaScorer, audio: AudioFeedback):
    """帯域パワー受信ハンドラーを生成。"""
    def handler(address: str, *args: float) -> None:
        values = tuple(float(a) for a in args if a is not None)
        state.update_band(band, values)

        # Alpha受信時のみスコアを計算
        if band == 'alpha':
            result = state.compute_and_record()
            if result is not None:
                alpha_rel, brs = result
                label = scorer.score_label(brs)
                ts = datetime.now().strftime('%H:%M:%S')
                above = scorer.is_above_threshold(alpha_rel)
                marker = " *** ALPHA HIGH ***" if above else ""
                print(f"[{ts}] Alpha: {alpha_rel:.3f}  BRS: {brs:.1f}  ({label}){marker}")
                if above:
                    played = audio.play_if_ready()
                    if played:
                        print(f"[{ts}] [AUDIO] Playing feedback tone")
    return handler


def make_hsi_handler(state: NeurofeedbackState):
    """HSI品質ハンドラーを生成。"""
    def handler(address: str, *args: float) -> None:
        state.update_hsi(args)
    return handler


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Alpha Meditation Neurofeedback - Muse S + Mind Monitor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ip', default='0.0.0.0', help='リッスンIP (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='リッスンポート (default: 5000)')
    parser.add_argument('--threshold', type=float, default=0.35,
                        help='Alpha相対パワーの閾値 (default: 0.35)')
    parser.add_argument('--no-audio', action='store_true', help='音声フィードバック無効')
    parser.add_argument('--no-visual', action='store_true', help='視覚フィードバック無効（コンソール出力のみ）')
    parser.add_argument('--history', type=int, default=120, help='履歴ウィンドウ秒数 (default: 120)')
    args = parser.parse_args()

    if not args.no_visual and not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib が使用できません。--no-visual モードで起動します。", file=sys.stderr)
        args.no_visual = True

    # コンポーネント初期化
    state = NeurofeedbackState(history_sec=args.history)
    scorer = AlphaScorer(threshold=args.threshold)
    audio = AudioFeedback(enabled=not args.no_audio)

    # OSC Dispatcher 設定
    disp = dispatcher.Dispatcher()
    for band in BANDS:
        disp.map(BAND_OSC_PATHS[band], make_band_handler(band, state, scorer, audio))
    disp.map(HSI_PATH, make_hsi_handler(state))

    # OSCサーバーをバックグラウンドスレッドで起動
    server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), disp)

    print("=" * 60)
    print("Alpha Meditation Neurofeedback")
    print("=" * 60)
    print(f"Listen:    {args.ip}:{args.port}")
    print(f"Threshold: {args.threshold:.2f} (Alpha relative power)")
    print(f"Audio:     {'disabled' if args.no_audio else 'enabled'}")
    print(f"Visual:    {'disabled' if args.no_visual else 'enabled'}")
    print(f"History:   {args.history}s")
    print()
    print("Mind Monitor settings:")
    print(f"  Host: <your PC IP address>")
    print(f"  Port: {args.port}")
    print()
    print("Waiting for data... (Ctrl+C to exit)")
    print("=" * 60)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        if not args.no_visual:
            visual = VisualFeedback(state, scorer)
            visual.start()  # ブロッキング（メインスレッドでFuncAnimation実行）
        else:
            # コンソール出力モード
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n終了します")
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()
