"""
Desk vs Worn Comparison Analysis

ヘッドバンドを机に置いた状態（非装着）と装着時のデータを比較し、
16, 20, 24, 28, 32 Hz のピークが機器由来かを判定

もし机置き状態でも同じピークが現れれば → 機器由来のアーチファクト
もし机置き状態では消失すれば → 脳波由来または接触関連
"""

import sys
sys.path.insert(0, '/home/tsu-nera/repo/satoru')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch
import warnings

warnings.filterwarnings('ignore')

from lib.loaders.mind_monitor import load_mind_monitor_csv
from lib.sensors.eeg import prepare_mne_raw, calculate_psd

# 出力ディレクトリ
OUTPUT_DIR = Path('/home/tsu-nera/repo/satoru/analysis/2025-12-08')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# データパス
WORN_DATA_PATH = '/home/tsu-nera/repo/satoru/data/mindMonitor_2025-12-08--07-38-33_3000223989795832513.csv'
DESK_DATA_PATH = '/home/tsu-nera/repo/satoru/data/mindMonitor_2025-12-08--08-30-18_8915321799368696312.csv'

# 対象ピーク周波数
TARGET_PEAKS = [16, 20, 24, 28, 32]


def load_desk_data(csv_path):
    """
    机置きデータを読み込む（HeadBandOnフィルタなし）
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['Time_sec'] = (df['TimeStamp'] - df['TimeStamp'].iloc[0]).dt.total_seconds()

    # 机置きなので HeadBandOn=0 のはず、フィルタなしで全データ使用
    return df


def prepare_raw_no_filter(df, sfreq=256.0):
    """
    フィルタなしでMNE RawArrayを作成（機器ノイズを保持）
    """
    import mne

    raw_cols = [c for c in df.columns if c.startswith('RAW_')]
    if not raw_cols:
        return None

    numeric = df[raw_cols].apply(pd.to_numeric, errors='coerce')
    numeric = numeric.interpolate(method='linear').ffill().bfill()

    ch_names = list(numeric.columns)
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    data = numeric.to_numpy().T * 1e-6
    raw = mne.io.RawArray(data, info, copy='auto', verbose=False)

    return raw


def analyze_accelerometer(df):
    """
    加速度センサーで持ち上げイベントを検出
    """
    acc_cols = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
    if not all(c in df.columns for c in acc_cols):
        return None

    acc_data = df[acc_cols].apply(pd.to_numeric, errors='coerce')

    # 加速度の大きさ
    acc_magnitude = np.sqrt(
        acc_data['Accelerometer_X']**2 +
        acc_data['Accelerometer_Y']**2 +
        acc_data['Accelerometer_Z']**2
    )

    # 静止時（机置き）は約1g、動きがあると変動
    acc_std_rolling = acc_magnitude.rolling(window=50).std()

    # 動きの閾値
    movement_threshold = 0.05
    is_moving = acc_std_rolling > movement_threshold

    return {
        'time': df['Time_sec'].values,
        'magnitude': acc_magnitude.values,
        'is_moving': is_moving.values,
        'movement_ratio': is_moving.sum() / len(is_moving) if len(is_moving) > 0 else 0
    }


def compare_psd_at_frequencies(psd_worn, psd_desk, target_freqs, tolerance=1.0):
    """
    装着時と机置き時のPSDを比較
    """
    results = []

    for target_freq in target_freqs:
        # 装着時
        mask_worn = np.abs(psd_worn['freqs'] - target_freq) < tolerance
        if mask_worn.any():
            worn_power = np.mean(psd_worn['psds'][:, mask_worn])
            worn_power_db = 10 * np.log10(worn_power + 1e-10)
        else:
            worn_power_db = np.nan

        # 机置き時
        mask_desk = np.abs(psd_desk['freqs'] - target_freq) < tolerance
        if mask_desk.any():
            desk_power = np.mean(psd_desk['psds'][:, mask_desk])
            desk_power_db = 10 * np.log10(desk_power + 1e-10)
        else:
            desk_power_db = np.nan

        # 差分
        power_diff = worn_power_db - desk_power_db if not (np.isnan(worn_power_db) or np.isnan(desk_power_db)) else np.nan

        results.append({
            'frequency': target_freq,
            'worn_power_db': worn_power_db,
            'desk_power_db': desk_power_db,
            'power_diff_db': power_diff,
        })

    return results


def plot_comparison(psd_worn, psd_desk, acc_data, target_freqs, output_path):
    """比較プロットを生成"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. PSD比較（全体）
    ax1 = axes[0, 0]
    freqs_worn = psd_worn['freqs']
    psd_avg_worn = np.mean(psd_worn['psds'], axis=0)
    psd_db_worn = 10 * np.log10(psd_avg_worn + 1e-10)

    freqs_desk = psd_desk['freqs']
    psd_avg_desk = np.mean(psd_desk['psds'], axis=0)
    psd_db_desk = 10 * np.log10(psd_avg_desk + 1e-10)

    ax1.plot(freqs_worn, psd_db_worn, 'b-', linewidth=1, label='Worn (装着)', alpha=0.8)
    ax1.plot(freqs_desk, psd_db_desk, 'r-', linewidth=1, label='Desk (机置き)', alpha=0.8)

    for freq in target_freqs:
        ax1.axvline(freq, color='gray', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power (dB)')
    ax1.set_title('PSD Comparison: Worn vs Desk')
    ax1.set_xlim(0, 50)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. ターゲット周波数付近の詳細比較
    ax2 = axes[0, 1]
    for target_freq in target_freqs:
        mask_worn = (freqs_worn >= target_freq - 3) & (freqs_worn <= target_freq + 3)
        mask_desk = (freqs_desk >= target_freq - 3) & (freqs_desk <= target_freq + 3)

        if mask_worn.any():
            ax2.plot(freqs_worn[mask_worn], psd_db_worn[mask_worn], 'b-',
                    alpha=0.6, label=f'{target_freq}Hz Worn' if target_freq == 16 else '')
        if mask_desk.any():
            ax2.plot(freqs_desk[mask_desk], psd_db_desk[mask_desk], 'r--',
                    alpha=0.6, label=f'{target_freq}Hz Desk' if target_freq == 16 else '')

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (dB)')
    ax2.set_title('Detailed View at Target Frequencies')
    ax2.legend(['Worn', 'Desk'])
    ax2.grid(True, alpha=0.3)

    # 3. パワー差分バーチャート
    ax3 = axes[1, 0]
    comparison = compare_psd_at_frequencies(psd_worn, psd_desk, target_freqs)
    freqs_plot = [r['frequency'] for r in comparison]
    worn_powers = [r['worn_power_db'] for r in comparison]
    desk_powers = [r['desk_power_db'] for r in comparison]

    x = np.arange(len(freqs_plot))
    width = 0.35

    ax3.bar(x - width/2, worn_powers, width, label='Worn', color='blue', alpha=0.7)
    ax3.bar(x + width/2, desk_powers, width, label='Desk', color='red', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{f} Hz' for f in freqs_plot])
    ax3.set_ylabel('Power (dB)')
    ax3.set_title('Power at Target Frequencies')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 加速度データ（持ち上げイベント検出）
    ax4 = axes[1, 1]
    if acc_data is not None:
        time = acc_data['time']
        magnitude = acc_data['magnitude']
        is_moving = acc_data['is_moving']

        ax4.plot(time, magnitude, 'b-', linewidth=0.5, alpha=0.7, label='Acc Magnitude')

        # 動いている区間をハイライト
        moving_regions = np.where(is_moving)[0]
        if len(moving_regions) > 0:
            ax4.fill_between(time, 0, 2, where=is_moving,
                           color='red', alpha=0.3, label='Movement detected')

        ax4.set_xlabel('Time (sec)')
        ax4.set_ylabel('Acceleration (g)')
        ax4.set_title(f'Accelerometer Data (Movement ratio: {acc_data["movement_ratio"]:.1%})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 2)
    else:
        ax4.text(0.5, 0.5, 'No accelerometer data', ha='center', va='center')
        ax4.set_title('Accelerometer Data')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def generate_comparison_report(comparison, acc_data, output_path):
    """比較レポートを生成"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Desk vs Worn Comparison Report\n\n")
        f.write("## 実験概要\n")
        f.write("- **目的**: 16, 20, 24, 28, 32 Hz のピークが機器由来かを判定\n")
        f.write("- **方法**: ヘッドバンドを装着状態と机置き状態で比較\n")
        f.write("- **判定基準**: 机置きでもピークが存在すれば機器由来\n\n")

        f.write("## 加速度センサー分析\n")
        if acc_data is not None:
            f.write(f"- 動き検出率: {acc_data['movement_ratio']:.1%}\n")
            f.write(f"- 平均加速度: {np.nanmean(acc_data['magnitude']):.3f} g\n")
            f.write("- 持ち上げイベントが検出されています\n\n")
        else:
            f.write("- 加速度データなし\n\n")

        f.write("## パワー比較結果\n")
        f.write("| 周波数 (Hz) | 装着時 (dB) | 机置き (dB) | 差分 (dB) | 判定 |\n")
        f.write("|------------|------------|------------|----------|------|\n")

        artifact_count = 0
        brain_count = 0

        for r in comparison:
            worn = r['worn_power_db']
            desk = r['desk_power_db']
            diff = r['power_diff_db']

            # 判定ロジック:
            # 机置きでもパワーが高い（差が小さい）→ 機器由来
            # 装着時のみパワーが高い（差が大きい）→ 脳波/接触由来
            if not np.isnan(diff):
                if abs(diff) < 5:  # 5dB未満の差
                    judgment = "機器由来"
                    artifact_count += 1
                elif diff > 10:  # 装着時が10dB以上高い
                    judgment = "脳波/接触由来"
                    brain_count += 1
                else:
                    judgment = "不明確"
            else:
                judgment = "データなし"

            worn_str = f"{worn:.1f}" if not np.isnan(worn) else "N/A"
            desk_str = f"{desk:.1f}" if not np.isnan(desk) else "N/A"
            diff_str = f"{diff:.1f}" if not np.isnan(diff) else "N/A"

            f.write(f"| {r['frequency']} | {worn_str} | {desk_str} | {diff_str} | {judgment} |\n")

        f.write("\n")

        # 総合判定
        f.write("## 総合判定\n\n")

        if artifact_count > brain_count:
            f.write("### 結論: **機器由来のアーチファクト**\n\n")
            f.write("ヘッドバンドを装着していない机置き状態でも同様のピークが観測されたため、\n")
            f.write("これらのピークは**機器内部で発生しているノイズ**であると判定されます。\n\n")
            f.write("#### 原因の可能性\n")
            f.write("1. **ADC/DACのクロックノイズ**: 内部デジタル回路のスイッチングノイズ\n")
            f.write("2. **電源レギュレータノイズ**: DC-DCコンバータの動作周波数\n")
            f.write("3. **Bluetoothモジュール**: 無線通信の干渉\n")
            f.write("4. **水晶発振器の高調波**: 内部クロックの漏れ\n\n")
            f.write("#### 対策\n")
            f.write("- これらの周波数帯の分析結果は慎重に解釈する必要があります\n")
            f.write("- ノッチフィルタで除去することを検討\n")
            f.write("- Beta帯（13-30Hz）の分析では、この影響を考慮する\n")
        elif brain_count > artifact_count:
            f.write("### 結論: **脳波/接触由来の可能性**\n\n")
            f.write("装着時にのみパワーが高く、机置きでは低下しているため、\n")
            f.write("これらのピークは接触や生体信号に関連している可能性があります。\n")
        else:
            f.write("### 結論: **判定困難**\n\n")
            f.write("装着状態と机置き状態で混在した結果が得られました。\n")
            f.write("追加の分析が必要です。\n")

    return output_path


def main():
    print("=" * 60)
    print("Desk vs Worn Comparison Analysis")
    print("=" * 60)

    # 装着時データのロード
    print("\n[1] Loading worn data...")
    df_worn = load_mind_monitor_csv(WORN_DATA_PATH, warmup_seconds=60)
    print(f"    Loaded {len(df_worn)} records (worn)")

    # 机置きデータのロード
    print("\n[2] Loading desk data...")
    df_desk = load_desk_data(DESK_DATA_PATH)
    print(f"    Loaded {len(df_desk)} records (desk)")

    # HeadBandOn状態を確認
    headband_on_count = (df_desk['HeadBandOn'] == 1).sum()
    headband_off_count = (df_desk['HeadBandOn'] == 0).sum()
    print(f"    HeadBandOn=1: {headband_on_count}, HeadBandOn=0: {headband_off_count}")

    # 加速度センサー分析
    print("\n[3] Analyzing accelerometer data...")
    acc_data = analyze_accelerometer(df_desk)
    if acc_data is not None:
        print(f"    Movement detected: {acc_data['movement_ratio']:.1%} of time")

    # MNE Raw作成（フィルタなし）
    print("\n[4] Preparing MNE raw data...")
    raw_worn = prepare_raw_no_filter(df_worn)
    raw_desk = prepare_raw_no_filter(df_desk)

    if raw_worn is None or raw_desk is None:
        print("ERROR: Could not prepare raw data")
        return

    print(f"    Worn: {raw_worn.n_times} samples")
    print(f"    Desk: {raw_desk.n_times} samples")

    # PSD計算
    print("\n[5] Calculating PSD...")
    psd_worn = calculate_psd(raw_worn)
    psd_desk = calculate_psd(raw_desk)

    # 比較分析
    print("\n[6] Comparing power at target frequencies...")
    comparison = compare_psd_at_frequencies(psd_worn, psd_desk, TARGET_PEAKS)

    for r in comparison:
        print(f"    {r['frequency']} Hz: Worn={r['worn_power_db']:.1f}dB, "
              f"Desk={r['desk_power_db']:.1f}dB, Diff={r['power_diff_db']:.1f}dB")

    # プロット生成
    print("\n[7] Generating plots...")
    plot_path = OUTPUT_DIR / 'desk_vs_worn_comparison.png'
    plot_comparison(psd_worn, psd_desk, acc_data, TARGET_PEAKS, plot_path)
    print(f"    Plot saved: {plot_path}")

    # レポート生成
    print("\n[8] Generating report...")
    report_path = OUTPUT_DIR / 'desk_vs_worn_comparison_report.md'
    generate_comparison_report(comparison, acc_data, report_path)
    print(f"    Report saved: {report_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return comparison


if __name__ == '__main__':
    results = main()
