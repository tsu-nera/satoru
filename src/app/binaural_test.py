"""
バイノーラルビート技術検証スクリプト

WSL2 + PulseAudio環境でのステレオ音声再生を検証する。
ヘッドフォン必須: 左右で異なる周波数を再生することでバイノーラルビートを生成。

使い方:
    source venv/bin/activate
    python src/app/binaural_test.py --step diagnose  # 環境確認
    python src/app/binaural_test.py --duration 10    # デフォルト（フェード付き）

オプション:
    --carrier   float   キャリア周波数 Hz (default: 200)
    --beat      float   ビート周波数 Hz (default: 8 = θ波)
    --duration  float   再生秒数 (default: 10)
    --amplitude float   音量 0.0-1.0 (default: 0.5)
    --step      str     テストステップ: diagnose, basic, fade, loop (default: fade)
    --loops     int     ループ回数 (--step loop 時, default: 1)
    --save      str     WAVファイル保存パス（指定時は再生なし）
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
import wave

import numpy as np


# --------------------------------------------------------------------------- #
# WAV生成
# --------------------------------------------------------------------------- #

def generate_stereo_wav(
    carrier_hz: float,
    beat_hz: float,
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    fade_sec: float = 0.05,
) -> bytes:
    """
    バイノーラルビート用ステレオWAVを生成する。

    左耳: carrier_hz
    右耳: carrier_hz + beat_hz
    """
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    fade_n = int(sample_rate * fade_sec)
    envelope = np.ones(n_samples)
    if fade_n > 0:
        envelope[:fade_n] = np.linspace(0, 1, fade_n)
        envelope[-fade_n:] = np.linspace(1, 0, fade_n)

    left = np.sin(2 * np.pi * carrier_hz * t) * envelope
    right = np.sin(2 * np.pi * (carrier_hz + beat_hz) * t) * envelope

    # インターリーブ: [L0, R0, L1, R1, ...]
    stereo = np.empty(n_samples * 2, dtype=np.int16)
    stereo[0::2] = (left * amplitude * 32767).astype(np.int16)
    stereo[1::2] = (right * amplitude * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# BinauralPlayer
# --------------------------------------------------------------------------- #

class BinauralPlayer:
    """ステレオWAVを再生するクラス。

    再生優先順位:
      1. powershell.exe (WSL2からWindows AudioStackに直接渡す。ノイズなし)
      2. paplay (PulseAudioネイティブ。WSLgのRDP経由のため音質劣化あり)
      3. aplay  (ALSA→PulseAudio変換層経由。さらに音質劣化)
    """

    def __init__(self) -> None:
        self._win_temp = self._detect_win_temp()
        self._use_paplay = self._check_cmd(['paplay', '--version'])
        self._use_aplay = self._check_cmd(['aplay', '--version'])

    @staticmethod
    def _check_cmd(cmd: list[str]) -> bool:
        try:
            return subprocess.run(cmd, capture_output=True, timeout=2).returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _detect_win_temp(self) -> str | None:
        """WSL2環境でWindowsの一時ディレクトリを検出する。"""
        try:
            # wslvar は wslu パッケージ提供。クリーンな出力。
            result = subprocess.run(
                ['wslvar', 'TEMP'],
                capture_output=True, text=True, timeout=3,
            )
            win_path = result.stdout.strip()
            if not win_path or result.returncode != 0:
                return None
            # C:\Users\...\Temp → /mnt/c/Users/.../Temp
            if len(win_path) >= 3 and win_path[1] == ':':
                drive = win_path[0].lower()
                wsl_path = f"/mnt/{drive}" + win_path[2:].replace('\\', '/')
                if os.path.isdir(wsl_path):
                    return wsl_path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    def _best_player(self) -> str:
        if self._win_temp:
            return 'powershell'
        if self._use_paplay:
            return 'paplay'
        if self._use_aplay:
            return 'aplay'
        return 'none'

    def run_diagnostic(self) -> bool:
        """環境診断を実行し、ステレオ対応を確認する。"""
        print("[DIAG] === Binaural Beat Diagnostic ===")

        # 1. Windows直接再生 (powershell.exe)
        if self._win_temp:
            print(f"[DIAG] Windows temp: {self._win_temp}")
            print("[DIAG] powershell.exe: available → ノイズなし再生可能")
        else:
            print("[DIAG] Windows temp: not detected (non-WSL2?)")

        # 2. PulseAudio (paplay)
        print(f"[DIAG] paplay: {'OK' if self._use_paplay else 'not found'}")

        # 3. ALSA (aplay)
        print(f"[DIAG] aplay: {'OK' if self._use_aplay else 'not found'}")

        # 4. WAVヘッダー確認
        test_wav = generate_stereo_wav(440, 8, 0.1, amplitude=0.3)
        with io.BytesIO(test_wav) as buf:
            with wave.open(buf, 'rb') as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                sampwidth = wf.getsampwidth()
        print(f"[DIAG] WAV header: channels={channels}, rate={rate}, sampwidth={sampwidth}")

        # 5. テスト再生
        best = self._best_player()
        print(f"[DIAG] Best player: {best}")
        print("[DIAG] Testing 0.5s stereo playback...")
        test_wav = generate_stereo_wav(440, 8, 0.5, amplitude=0.3)
        player_name, rc = self._run_player(test_wav, timeout=5.0)
        if rc == 0:
            print(f"[DIAG] Test playback: OK via {player_name}")
            print("[DIAG] Stereo support: CONFIRMED")
            return True
        else:
            print(f"[DIAG] Test playback: FAILED ({player_name}, rc={rc})")
            print("[DIAG]   Fallback: use --save to export WAV, play on Windows with VLC")
            return False

    def _run_powershell(self, wav_bytes: bytes, timeout: float) -> int:
        """WAVをWindows tempに書き出し、PowerShellのSoundPlayerで再生。

        WSLgのRDPオーディオパイプラインを完全にバイパスし、
        Windows AudioStack で直接再生するためノイズが発生しない。
        """
        wav_path = os.path.join(self._win_temp, '_binaural_tmp.wav')
        try:
            with open(wav_path, 'wb') as f:
                f.write(wav_bytes)
            # WSLパス → Windowsパス変換
            win_path = subprocess.run(
                ['wslpath', '-w', wav_path],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            result = subprocess.run(
                ['powershell.exe', '-NoProfile', '-c',
                 f"(New-Object System.Media.SoundPlayer '{win_path}').PlaySync()"],
                timeout=timeout,
                capture_output=True,
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            return -1
        except Exception:
            return -2
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _run_paplay(self, wav_bytes: bytes, timeout: float) -> int:
        """paplay (PulseAudioネイティブ) でWAVを再生。"""
        try:
            result = subprocess.run(
                ['paplay', '--latency-msec=200', '/dev/stdin'],
                input=wav_bytes, timeout=timeout, capture_output=True,
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            return -1
        except Exception:
            return -2

    def _run_aplay(self, wav_bytes: bytes, timeout: float) -> int:
        """aplay (ALSA経由PulseAudio) でWAVを再生。"""
        try:
            result = subprocess.run(
                ['aplay', '-q', '--buffer-size=65536', '-'],
                input=wav_bytes, timeout=timeout, capture_output=True,
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            return -1
        except Exception:
            return -2

    def _run_player(self, wav_bytes: bytes, timeout: float) -> tuple[str, int]:
        """利用可能な最良のプレーヤーで再生し、(player名, returncode) を返す。"""
        if self._win_temp:
            rc = self._run_powershell(wav_bytes, timeout)
            if rc == 0:
                return ('powershell', 0)
            print(f"[BinauralPlayer] powershell failed (rc={rc}), falling back...")
        if self._use_paplay:
            rc = self._run_paplay(wav_bytes, timeout)
            if rc == 0:
                return ('paplay', 0)
        if self._use_aplay:
            rc = self._run_aplay(wav_bytes, timeout)
            return ('aplay', rc)
        return ('none', -3)

    def play_once(self, wav_bytes: bytes, duration: float) -> bool:
        """WAVを1回再生する。成功時True。"""
        if self._best_player() == 'none':
            print("[BinauralPlayer] no player available", file=sys.stderr)
            return False
        _, rc = self._run_player(wav_bytes, timeout=duration + 10.0)
        return rc == 0

    def play_loop(self, wav_bytes: bytes, duration: float, n_loops: int) -> None:
        """WAVをN回ループ再生する。"""
        for i in range(n_loops):
            print(f"[Loop] {i + 1}/{n_loops}")
            ok = self.play_once(wav_bytes, duration)
            if not ok:
                print("[Loop] Playback failed, stopping.", file=sys.stderr)
                break


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Binaural Beat技術検証スクリプト (WSL2 + PulseAudio)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--carrier', type=float, default=200.0,
                        help='キャリア周波数 Hz (default: 200)')
    parser.add_argument('--beat', type=float, default=8.0,
                        help='ビート周波数 Hz (default: 8 = θ波)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='再生秒数 (default: 10)')
    parser.add_argument('--amplitude', type=float, default=0.5,
                        help='音量 0.0-1.0 (default: 0.5)')
    parser.add_argument('--step', choices=['diagnose', 'basic', 'fade', 'loop'],
                        default='fade',
                        help='テストステップ (default: fade)')
    parser.add_argument('--loops', type=int, default=1,
                        help='ループ回数 (--step loop 時, default: 1)')
    parser.add_argument('--save', type=str, default=None,
                        help='WAVファイル保存パス（指定時は再生なし）')
    args = parser.parse_args()

    player = BinauralPlayer()

    print(f"Carrier: {args.carrier} Hz  |  Beat: {args.beat} Hz  |  "
          f"Duration: {args.duration}s  |  Amplitude: {args.amplitude}")
    print(f"Step: {args.step}")
    print()

    if args.step == 'diagnose':
        ok = player.run_diagnostic()
        sys.exit(0 if ok else 1)

    # WAV生成
    fade_sec = 0.0 if args.step == 'basic' else 0.05
    wav_bytes = generate_stereo_wav(
        carrier_hz=args.carrier,
        beat_hz=args.beat,
        duration=args.duration,
        amplitude=args.amplitude,
        fade_sec=fade_sec,
    )
    print(f"Generated stereo WAV: {len(wav_bytes)} bytes, "
          f"Left={args.carrier:.1f}Hz, Right={args.carrier + args.beat:.1f}Hz")

    # --save: ファイル保存のみ
    if args.save:
        with open(args.save, 'wb') as f:
            f.write(wav_bytes)
        print(f"Saved: {args.save}")
        print("(再生なし。Windows側でVLC等で確認してください)")
        return

    # 再生
    best = player._best_player()
    if best == 'none':
        print("WARNING: no player found. Use --save to export WAV.")
        sys.exit(1)
    print(f"Player: {best}")

    print("Playing... (ヘッドフォン推奨)")
    print(f"  Left ear:  {args.carrier:.1f} Hz")
    print(f"  Right ear: {args.carrier + args.beat:.1f} Hz")
    print(f"  Beat freq: {args.beat:.1f} Hz (θ波: 4-8Hz, α波: 8-13Hz, β波: 13-30Hz)")

    start = time.time()

    if args.step == 'loop':
        player.play_loop(wav_bytes, args.duration, args.loops)
    else:
        ok = player.play_once(wav_bytes, args.duration)
        elapsed = time.time() - start
        if ok:
            print(f"Done. ({elapsed:.1f}s)")
        else:
            print(f"Playback failed after {elapsed:.1f}s")
            print("Fallback options:")
            print("  1. python src/app/binaural_test.py --save /tmp/binaural.wav")
            print("     → Windows側でVLCで /tmp/binaural.wav を開く")
            print("  2. pip install sounddevice  # 本番実装向け")
            sys.exit(1)


if __name__ == '__main__':
    main()
