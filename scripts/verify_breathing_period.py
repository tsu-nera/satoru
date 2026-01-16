#!/usr/bin/env python3
"""
周波数を実際の呼吸周期（秒）に変換して検証
"""

print("=" * 70)
print("周波数 → 呼吸周期の変換")
print("=" * 70)

# 候補周波数
candidates = [
    ("HF Peak", 0.1562, "検出されたHF Peak"),
    ("0.25 Hz", 0.25, "グラフで目立つ周波数"),
    ("Mean BR", 0.0633, "レポートのMean Breathing Rate"),
]

print("\n【周波数から呼吸周期を計算】")
print("計算式: 呼吸周期(秒) = 60 / (周波数 × 60)")
print("       = 1 / 周波数\n")

for name, freq_hz, description in candidates:
    freq_bpm = freq_hz * 60
    period_sec = 1 / freq_hz

    print(f"{name}: {freq_hz:.4f} Hz")
    print(f"  = {freq_bpm:.1f} bpm")
    print(f"  = {period_sec:.1f} 秒に1回呼吸")
    print(f"  ({description})")
    print()

print("=" * 70)
print("瞑想中の実際の呼吸（ユーザー証言）")
print("=" * 70)
print("\n呼吸周期: 15-30秒に1回")
print("\nこれを周波数に変換すると:")

breathing_periods = [15, 20, 25, 30]
for period in breathing_periods:
    freq_hz = 1 / period
    freq_bpm = freq_hz * 60

    # どの帯域か
    if freq_hz < 0.04:
        band = "VLF"
    elif freq_hz < 0.15:
        band = "LF"
    elif freq_hz < 0.4:
        band = "HF"
    else:
        band = "VHF"

    print(f"  {period}秒/回 = {freq_hz:.4f} Hz = {freq_bpm:.2f} bpm ({band}帯域)")

print("\n" + "=" * 70)
print("結論")
print("=" * 70)

print("""
実際の呼吸が15-30秒に1回なら:
  → 周波数は 0.033-0.067 Hz (2.0-4.0 bpm)
  → VLF-LF帯域に該当

各候補の検証:
  ✓ Mean BR = 0.0633 Hz (15.8秒/回)
    → 実際の呼吸周期と一致！

  ✗ HF Peak = 0.1562 Hz (6.4秒/回)
    → 瞑想中の呼吸としては速すぎる
    → 呼吸ではなく別の生理現象？

  ✗ 0.25 Hz = 4.0秒/回
    → さらに速い、呼吸ではありえない

問題:
  "Spectral BR = 9.4 bpm (6.4秒/回)" が実際の呼吸と合わない
  → この"Spectral BR"は何を検出しているのか？
  → 呼吸分析の実装を確認する必要がある
""")

print("=" * 70)
