"""
ISF (Infra-Slow Fluctuation) マルチモーダル解析

EEG ISF / fNIRS LFO / HRV Mayer波 の相互相関を検証する。

Muse S の RAW EEG に内蔵されたハイパスフィルタのカットオフを推定し、
ISF帯域 (0.05-0.1 Hz) で EEG・fNIRS・HRV 間の
neurovascular coupling を評価する。
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.loaders.mind_monitor import load_mind_monitor_csv, get_eeg_data, get_optics_data
from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
from lib.sensors.fnirs import calculate_hbo_hbr

# ============================================================
# 設定
# ============================================================
MUSE_CSV = PROJECT_ROOT / "data/muse/mindMonitor_2026-02-06--07-42-47_4574839398217372417.csv"
SELFLOOPS_CSV = PROJECT_ROOT / "data/selfloops/selfloops_2026-02-06--07-42-47.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
EEG_FS = 256  # Muse S nominal sampling rate
ISF_BAND = (0.05, 0.1)  # ISF band (Hz)
LFO_BAND = (0.01, 0.1)  # Broader LFO band for fNIRS
MAYER_BAND = (0.07, 0.13)  # Mayer wave band in HRV (~0.1 Hz)
RESAMPLE_FS = 1.0  # Common resampling rate (Hz) for cross-modal comparison
WARMUP_SEC = 60  # Skip first 60 sec


# ============================================================
# 1. HPF カットオフ推定
# ============================================================
def estimate_hpf_cutoff(raw_eeg, fs, channel_name="AF7"):
    """EEG PSD からハイパスフィルタのカットオフ周波数を推定"""
    nperseg = min(fs * 120, len(raw_eeg) // 2)
    f, psd = signal.welch(raw_eeg, fs=fs, nperseg=nperseg)

    # 0.005-0.5 Hz でPSDピーク = HPF通過後の最大パワー周波数
    mask = (f > 0.005) & (f < 0.5)
    f_low, psd_low = f[mask], psd[mask]
    peak_idx = np.argmax(psd_low)
    peak_freq = f_low[peak_idx]

    # -3dB point (HPF cutoff)
    psd_3db = psd_low[peak_idx] / 2
    below_peak = psd_low[:peak_idx]
    cutoff_3db = None
    if len(below_peak) > 0:
        closest = np.argmin(np.abs(below_peak - psd_3db))
        cutoff_3db = f_low[closest]

    # Slope analysis per band
    slopes = {}
    for f_lo, f_hi, label in [
        (0.008, 0.02, "0.008-0.02"),
        (0.02, 0.05, "0.02-0.05"),
        (0.05, 0.1, "0.05-0.1"),
        (0.1, 0.5, "0.1-0.5"),
        (0.5, 4, "0.5-4"),
    ]:
        m = (f >= f_lo) & (f <= f_hi) & (f > 0) & (psd > 0)
        if np.sum(m) > 2:
            slope = np.polyfit(np.log10(f[m]), np.log10(psd[m]), 1)[0]
            slopes[label] = slope

    return {
        "channel": channel_name,
        "psd_peak_hz": peak_freq,
        "cutoff_3db_hz": cutoff_3db,
        "slopes": slopes,
        "freqs": f,
        "psd": psd,
    }


# ============================================================
# 2. ISF帯域信号の抽出（1 Hz リサンプル → バンドパス）
# ============================================================
def extract_isf_band(data, fs_orig, band, target_fs=RESAMPLE_FS):
    """高サンプリングレートの信号を1Hzにリサンプルし、バンドパスフィルタ適用"""
    n_sec = int(len(data) / fs_orig)
    # 1秒平均でダウンサンプル
    resampled = np.array([np.nanmean(data[int(i * fs_orig):int((i + 1) * fs_orig)]) for i in range(n_sec)])

    # NaN補間
    nans = np.isnan(resampled)
    if nans.all():
        return resampled
    if nans.any():
        x = np.arange(len(resampled))
        resampled[nans] = np.interp(x[nans], x[~nans], resampled[~nans])

    # バンドパスフィルタ
    sos = signal.butter(3, list(band), btype="bandpass", fs=target_fs, output="sos")
    filtered = signal.sosfiltfilt(sos, resampled)
    return filtered


def extract_isf_from_rr(rr_intervals, time_sec, band, target_fs=RESAMPLE_FS):
    """R-R間隔時系列からISF/Mayer波帯域を抽出"""
    # 等間隔にリサンプリング (1 Hz)
    duration = time_sec[-1] - time_sec[0]
    n_sec = int(duration)
    t_regular = np.linspace(time_sec[0], time_sec[0] + n_sec - 1, n_sec)
    rr_interp = np.interp(t_regular, time_sec, rr_intervals)

    # デトレンド
    rr_detrend = signal.detrend(rr_interp)

    # バンドパスフィルタ
    sos = signal.butter(3, list(band), btype="bandpass", fs=target_fs, output="sos")
    filtered = signal.sosfiltfilt(sos, rr_detrend)
    return filtered, t_regular


# ============================================================
# 3. 相互相関解析
# ============================================================
def cross_correlate(sig_a, sig_b, max_lag_sec=30, fs=1.0):
    """2信号間のクロスコリレーション（ラグ付きピアソン相関）"""
    max_lag = int(max_lag_sec * fs)
    n = len(sig_a)
    if n < 2 * max_lag + 10:
        max_lag = n // 4

    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []
    for lag in lags:
        if lag >= 0:
            a = sig_a[:n - lag] if lag > 0 else sig_a
            b = sig_b[lag:] if lag > 0 else sig_b
        else:
            a = sig_a[-lag:]
            b = sig_b[:n + lag]
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            correlations.append(0.0)
        else:
            r, _ = pearsonr(a, b)
            correlations.append(r)

    correlations = np.array(correlations)
    peak_idx = np.argmax(np.abs(correlations))
    return {
        "lags": lags / fs,
        "correlations": correlations,
        "peak_lag_sec": lags[peak_idx] / fs,
        "peak_r": correlations[peak_idx],
        "zero_lag_r": correlations[max_lag],
    }


# ============================================================
# 4. PSD比較
# ============================================================
def compute_psd_1hz(data, fs=1.0, nperseg=256):
    """1Hzリサンプル済み信号のPSD"""
    nperseg = min(nperseg, len(data) // 2)
    f, psd = signal.welch(data, fs=fs, nperseg=nperseg)
    return f, psd


# ============================================================
# メイン解析
# ============================================================
def main():
    print("=" * 60)
    print("ISF Multi-Modal Analysis")
    print("=" * 60)

    # ----------------------------------------------------------
    # データ読み込み
    # ----------------------------------------------------------
    print("\n[1] Loading data...")
    df_muse = load_mind_monitor_csv(str(MUSE_CSV), warmup_seconds=WARMUP_SEC)
    eeg_data = get_eeg_data(df_muse)
    optics_data = get_optics_data(df_muse)

    df_sl = load_selfloops_csv(str(SELFLOOPS_CSV), warmup_seconds=WARMUP_SEC)
    hrv_data = get_hrv_data(df_sl)

    n_eeg = len(eeg_data["AF7"])
    duration_sec = n_eeg / EEG_FS
    print(f"  EEG samples: {n_eeg} ({duration_sec / 60:.1f} min)")
    print(f"  fNIRS samples: {len(optics_data['left_730'])}")
    print(f"  HRV R-R intervals: {len(hrv_data['rr_intervals_clean'])}")

    # ----------------------------------------------------------
    # HPF カットオフ推定
    # ----------------------------------------------------------
    print("\n[2] Estimating HPF cutoff...")
    hpf_results = {}
    for ch in ["AF7", "AF8", "TP9", "TP10"]:
        hpf_results[ch] = estimate_hpf_cutoff(eeg_data[ch], EEG_FS, ch)
        r = hpf_results[ch]
        print(f"  {ch}: PSD peak={r['psd_peak_hz']:.3f} Hz, -3dB={r['cutoff_3db_hz']:.3f} Hz")

    # ----------------------------------------------------------
    # fNIRS: HbO/HbR 計算
    # ----------------------------------------------------------
    print("\n[3] Computing fNIRS HbO/HbR...")
    left_hbo, left_hbr = calculate_hbo_hbr(
        optics_data["left_730"], optics_data["left_850"]
    )
    right_hbo, right_hbr = calculate_hbo_hbr(
        optics_data["right_730"], optics_data["right_850"]
    )
    print(f"  Left  HbO: mean={np.nanmean(left_hbo):.3f}, std={np.nanstd(left_hbo):.4f}")
    print(f"  Right HbO: mean={np.nanmean(right_hbo):.3f}, std={np.nanstd(right_hbo):.4f}")

    # ----------------------------------------------------------
    # ISF帯域信号の抽出
    # ----------------------------------------------------------
    print("\n[4] Extracting ISF band signals...")

    # EEG ISF (AF7)
    eeg_isf = extract_isf_band(eeg_data["AF7"], EEG_FS, ISF_BAND)

    # fNIRS ISF (HbO Left)
    fnirs_isf = extract_isf_band(left_hbo, EEG_FS, ISF_BAND)

    # HRV Mayer wave
    rr_clean = hrv_data["rr_intervals_clean"]
    rr_time = hrv_data["time"]
    hrv_mayer, hrv_time = extract_isf_from_rr(rr_clean, rr_time, MAYER_BAND)

    # 共通長に切り詰め
    min_len = min(len(eeg_isf), len(fnirs_isf), len(hrv_mayer))
    eeg_isf = eeg_isf[:min_len]
    fnirs_isf = fnirs_isf[:min_len]
    hrv_mayer = hrv_mayer[:min_len]
    t_common = np.arange(min_len)

    print(f"  Common duration: {min_len} sec ({min_len / 60:.1f} min)")

    # ----------------------------------------------------------
    # 相互相関解析
    # ----------------------------------------------------------
    print("\n[5] Cross-correlation analysis...")

    pairs = [
        ("EEG ISF", "fNIRS ISF (HbO)", eeg_isf, fnirs_isf),
        ("EEG ISF", "HRV Mayer", eeg_isf, hrv_mayer),
        ("fNIRS ISF (HbO)", "HRV Mayer", fnirs_isf, hrv_mayer),
    ]

    xcorr_results = {}
    for name_a, name_b, sig_a, sig_b in pairs:
        key = f"{name_a} vs {name_b}"
        xcorr_results[key] = cross_correlate(sig_a, sig_b)
        r = xcorr_results[key]
        print(f"  {key}:")
        print(f"    zero-lag r = {r['zero_lag_r']:.3f}")
        print(f"    peak r = {r['peak_r']:.3f} at lag = {r['peak_lag_sec']:.1f} sec")

    # ----------------------------------------------------------
    # PSD比較
    # ----------------------------------------------------------
    print("\n[6] Computing PSDs for comparison...")

    # 1 Hz resampled (before bandpass) for PSD
    eeg_1hz = extract_isf_band(eeg_data["AF7"], EEG_FS, (0.001, 0.49))  # wide band
    fnirs_1hz = extract_isf_band(left_hbo, EEG_FS, (0.001, 0.49))

    # Actually, for PSD comparison we want unfiltered resampled data
    n_sec = int(len(eeg_data["AF7"]) / EEG_FS)
    eeg_resamp = np.array([np.nanmean(eeg_data["AF7"][i * EEG_FS:(i + 1) * EEG_FS]) for i in range(n_sec)])
    fnirs_resamp = np.array([np.nanmean(left_hbo[i * EEG_FS:(i + 1) * EEG_FS]) for i in range(n_sec)])

    # NaN handling for fNIRS
    nans = np.isnan(fnirs_resamp)
    if nans.any() and not nans.all():
        x = np.arange(len(fnirs_resamp))
        fnirs_resamp[nans] = np.interp(x[nans], x[~nans], fnirs_resamp[~nans])

    f_eeg, psd_eeg = compute_psd_1hz(eeg_resamp)
    f_fnirs, psd_fnirs = compute_psd_1hz(fnirs_resamp)

    # HRV PSD
    rr_dur = int(rr_time[-1] - rr_time[0])
    t_rr = np.linspace(rr_time[0], rr_time[0] + rr_dur - 1, rr_dur)
    rr_resamp = np.interp(t_rr, rr_time, rr_clean)
    rr_resamp = signal.detrend(rr_resamp)
    f_hrv, psd_hrv = compute_psd_1hz(rr_resamp)

    # ----------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------
    print("\n[7] Generating plots...")
    _plot_hpf_estimation(hpf_results)
    _plot_psd_comparison(f_eeg, psd_eeg, f_fnirs, psd_fnirs, f_hrv, psd_hrv)
    _plot_isf_timeseries(t_common, eeg_isf, fnirs_isf, hrv_mayer)
    _plot_cross_correlations(xcorr_results)
    _plot_summary(t_common, eeg_isf, fnirs_isf, hrv_mayer, xcorr_results,
                  f_eeg, psd_eeg, f_fnirs, psd_fnirs, f_hrv, psd_hrv, hpf_results)

    # ----------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    af7_r = hpf_results["AF7"]
    print(f"HPF -3dB cutoff (AF7): {af7_r['cutoff_3db_hz']:.3f} Hz")
    print(f"HPF PSD peak (AF7):    {af7_r['psd_peak_hz']:.3f} Hz")
    print()
    for key, r in xcorr_results.items():
        print(f"{key}:")
        print(f"  zero-lag r = {r['zero_lag_r']:.3f}, peak r = {r['peak_r']:.3f} (lag={r['peak_lag_sec']:.1f}s)")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
    return xcorr_results, hpf_results


# ============================================================
# Plotting functions
# ============================================================
def _plot_hpf_estimation(hpf_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, ch in zip(axes.flat, ["TP9", "AF7", "AF8", "TP10"]):
        r = hpf_results[ch]
        f, psd = r["freqs"], r["psd"]
        ax.loglog(f[1:], psd[1:], linewidth=0.8)
        ax.axvline(0.1, color="r", linestyle="--", alpha=0.5, label="0.1 Hz")
        ax.axvline(0.05, color="g", linestyle="--", alpha=0.5, label="0.05 Hz")
        if r["cutoff_3db_hz"]:
            ax.axvline(r["cutoff_3db_hz"], color="orange", linestyle="-", alpha=0.8,
                       label=f"-3dB: {r['cutoff_3db_hz']:.3f} Hz")
        ax.set_title(f"{ch}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("HPF Cutoff Estimation per Channel", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_hpf_estimation.png", dpi=150)
    plt.close()


def _plot_psd_comparison(f_eeg, psd_eeg, f_fnirs, psd_fnirs, f_hrv, psd_hrv):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, f, psd, title, color in [
        (axes[0], f_eeg, psd_eeg, "EEG (AF7)", "C0"),
        (axes[1], f_fnirs, psd_fnirs, "fNIRS (HbO Left)", "C1"),
        (axes[2], f_hrv, psd_hrv, "HRV (R-R intervals)", "C2"),
    ]:
        mask = f > 0
        ax.semilogy(f[mask], psd[mask], color=color, linewidth=1)
        ax.axvspan(ISF_BAND[0], ISF_BAND[1], alpha=0.15, color="red", label="ISF 0.05-0.1 Hz")
        ax.axvspan(MAYER_BAND[0], MAYER_BAND[1], alpha=0.1, color="blue", label="Mayer 0.07-0.13 Hz")
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.5)

    plt.suptitle("PSD Comparison (1 Hz resampled)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_psd_comparison.png", dpi=150)
    plt.close()


def _plot_isf_timeseries(t, eeg_isf, fnirs_isf, hrv_mayer):
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

    axes[0].plot(t / 60, eeg_isf, "C0", linewidth=0.8)
    axes[0].set_ylabel("EEG ISF\n(0.05-0.1 Hz)")
    axes[0].set_title("ISF Band Time Series")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t / 60, fnirs_isf, "C1", linewidth=0.8)
    axes[1].set_ylabel("fNIRS ISF\n(HbO, 0.05-0.1 Hz)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t / 60, hrv_mayer, "C2", linewidth=0.8)
    axes[2].set_ylabel("HRV Mayer\n(R-R, 0.07-0.13 Hz)")
    axes[2].set_xlabel("Time (min)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_isf_timeseries.png", dpi=150)
    plt.close()


def _plot_cross_correlations(xcorr_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for ax, (key, r) in zip(axes, xcorr_results.items()):
        ax.plot(r["lags"], r["correlations"], linewidth=1)
        ax.axhline(0, color="k", linestyle="-", alpha=0.3)
        ax.axvline(0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(r["peak_lag_sec"], color="r", linestyle="--", alpha=0.5,
                   label=f"peak: r={r['peak_r']:.3f} @ {r['peak_lag_sec']:.1f}s")
        ax.set_title(key)
        ax.set_xlabel("Lag (sec)")
        ax.set_ylabel("Correlation")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Cross-Correlation (ISF / Mayer wave bands)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_cross_correlations.png", dpi=150)
    plt.close()


def _plot_summary(t, eeg_isf, fnirs_isf, hrv_mayer, xcorr_results,
                  f_eeg, psd_eeg, f_fnirs, psd_fnirs, f_hrv, psd_hrv, hpf_results):
    """All-in-one summary figure"""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # Row 1: HPF estimation (AF7) + PSD comparison
    ax_hpf = fig.add_subplot(gs[0, 0])
    r = hpf_results["AF7"]
    ax_hpf.loglog(r["freqs"][1:], r["psd"][1:], linewidth=0.8)
    ax_hpf.axvline(0.1, color="r", linestyle="--", alpha=0.5)
    ax_hpf.axvline(0.05, color="g", linestyle="--", alpha=0.5)
    if r["cutoff_3db_hz"]:
        ax_hpf.axvline(r["cutoff_3db_hz"], color="orange", linewidth=2, alpha=0.8,
                        label=f"-3dB: {r['cutoff_3db_hz']:.3f} Hz")
    ax_hpf.set_title("EEG PSD (AF7) - HPF Detection")
    ax_hpf.legend(fontsize=7)
    ax_hpf.grid(True, alpha=0.3)

    # PSD overlays (normalized)
    ax_psd = fig.add_subplot(gs[0, 1:])
    for f, psd, label, color in [
        (f_eeg, psd_eeg, "EEG", "C0"),
        (f_fnirs, psd_fnirs, "fNIRS HbO", "C1"),
        (f_hrv, psd_hrv, "HRV R-R", "C2"),
    ]:
        mask = f > 0
        psd_norm = psd[mask] / psd[mask].max()
        ax_psd.semilogy(f[mask], psd_norm, label=label, color=color, linewidth=1)
    ax_psd.axvspan(ISF_BAND[0], ISF_BAND[1], alpha=0.15, color="red", label="ISF band")
    ax_psd.set_title("Normalized PSD Comparison")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_xlim(0, 0.5)
    ax_psd.legend(fontsize=8)
    ax_psd.grid(True, alpha=0.3)

    # Row 2-3: Time series
    for row, (data, label, color) in enumerate([
        (eeg_isf, "EEG ISF (0.05-0.1 Hz)", "C0"),
        (fnirs_isf, "fNIRS ISF (HbO, 0.05-0.1 Hz)", "C1"),
        (hrv_mayer, "HRV Mayer wave (R-R, 0.07-0.13 Hz)", "C2"),
    ]):
        ax = fig.add_subplot(gs[1, row])
        ax.plot(t / 60, data, color=color, linewidth=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Time (min)")
        ax.grid(True, alpha=0.3)

    # Row 3: Cross-correlations
    for col, (key, r) in enumerate(xcorr_results.items()):
        ax = fig.add_subplot(gs[2, col])
        ax.plot(r["lags"], r["correlations"], linewidth=1)
        ax.axhline(0, color="k", alpha=0.3)
        ax.axvline(0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(r["peak_lag_sec"], color="r", linestyle="--", alpha=0.6)
        ax.set_title(f"{key}\npeak r={r['peak_r']:.3f} @ {r['peak_lag_sec']:.0f}s", fontsize=9)
        ax.set_xlabel("Lag (sec)")
        ax.set_ylabel("r")
        ax.grid(True, alpha=0.3)

    # Row 4: Summary text
    ax_text = fig.add_subplot(gs[3, :])
    ax_text.axis("off")
    summary_lines = [
        f"HPF -3dB Cutoff (AF7): {hpf_results['AF7']['cutoff_3db_hz']:.3f} Hz  |  PSD Peak: {hpf_results['AF7']['psd_peak_hz']:.3f} Hz",
        "",
    ]
    for key, r in xcorr_results.items():
        summary_lines.append(f"{key}:  zero-lag r = {r['zero_lag_r']:.3f},  peak r = {r['peak_r']:.3f} (lag = {r['peak_lag_sec']:.1f} sec)")
    summary_text = "\n".join(summary_lines)
    ax_text.text(0.05, 0.5, summary_text, transform=ax_text.transAxes,
                 fontsize=12, fontfamily="monospace", verticalalignment="center")

    plt.suptitle("ISF Multi-Modal Analysis Summary", fontsize=16, y=0.98)
    plt.savefig(OUTPUT_DIR / "05_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
