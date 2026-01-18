#!/usr/bin/env python3
"""
共鳴周波数測定プロトコル生成スクリプト

測定順序効果を排除するため、ランダム化された測定順序を生成します。
"""

import random
from datetime import datetime


def generate_breathing_rates(min_rate=3.5, max_rate=7.0, step=0.5):
    """
    測定する呼吸レートのリストを生成

    Parameters
    ----------
    min_rate : float
        最小呼吸レート (breaths/min)
    max_rate : float
        最大呼吸レート (breaths/min)
    step : float
        ステップサイズ (breaths/min)

    Returns
    -------
    list of float
        呼吸レートのリスト
    """
    rates = []
    current = min_rate
    while current <= max_rate:
        rates.append(current)
        current += step
    return rates


def calculate_inhale_exhale(rate, ratio=0.4):
    """
    呼吸レートから吸気・呼気時間を計算

    Parameters
    ----------
    rate : float
        呼吸レート (breaths/min)
    ratio : float
        吸気時間の比率（デフォルト: 0.4、つまり吸気40%、呼気60%）

    Returns
    -------
    inhale : float
        吸気時間（秒）
    exhale : float
        呼気時間（秒）

    Examples
    --------
    >>> inhale, exhale = calculate_inhale_exhale(6.0)
    >>> print(f"6.0 bpm: Inhale {inhale:.1f}s, Exhale {exhale:.1f}s")
    6.0 bpm: Inhale 4.0s, Exhale 6.0s
    """
    cycle_duration = 60.0 / rate  # 1サイクルの時間（秒）
    inhale = cycle_duration * ratio
    exhale = cycle_duration * (1 - ratio)
    return inhale, exhale


def generate_protocol(
    breathing_rates,
    replicate_rate=None,
    seed=None
):
    """
    ランダム化された測定プロトコルを生成

    Parameters
    ----------
    breathing_rates : list of float
        測定する呼吸レートのリスト
    replicate_rate : float or None
        再現性確認のため複数回測定するレート
        Noneの場合、中央値のレートを選択
    seed : int or None
        乱数シード（再現性のため）

    Returns
    -------
    list of dict
        測定プロトコル
        各要素: {'trial': int, 'rate': float, 'inhale': float, 'exhale': float}
    """
    if seed is not None:
        random.seed(seed)

    # 再現性確認レートの選択
    if replicate_rate is None:
        # 中央値のレートを選択
        sorted_rates = sorted(breathing_rates)
        replicate_rate = sorted_rates[len(sorted_rates) // 2]

    # プロトコル作成
    protocol = []

    # 全レート + 再現性確認レート
    all_rates = breathing_rates + [replicate_rate]

    # ランダム化
    random.shuffle(all_rates)

    # 各レートの吸気・呼気時間を計算
    for trial_num, rate in enumerate(all_rates, start=1):
        inhale, exhale = calculate_inhale_exhale(rate)
        protocol.append({
            'trial': trial_num,
            'rate': rate,
            'inhale': round(inhale, 1),
            'exhale': round(exhale, 1),
        })

    return protocol, replicate_rate


def print_protocol(protocol, replicate_rate):
    """
    測定プロトコルを表示

    Parameters
    ----------
    protocol : list of dict
        測定プロトコル
    replicate_rate : float
        再現性確認レート
    """
    print("=" * 70)
    print("共鳴周波数測定プロトコル")
    print("=" * 70)
    print(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"測定回数: {len(protocol)}回")
    print(f"再現性確認レート: {replicate_rate} bpm")
    print()

    print("測定順序:")
    print("-" * 70)
    print(f"{'Trial':<8}{'Rate (bpm)':<15}{'Inhale (s)':<15}{'Exhale (s)':<15}")
    print("-" * 70)

    for item in protocol:
        marker = " *" if item['rate'] == replicate_rate else ""
        print(f"{item['trial']:<8}{item['rate']:<15.1f}{item['inhale']:<15.1f}{item['exhale']:<15.1f}{marker}")

    print("-" * 70)
    print("* = 再現性確認のための複数回測定")
    print()

    print("測定手順:")
    print("-" * 70)
    print("1. 測定前準備:")
    print("   - 静かな環境で測定")
    print("   - 測定開始10分前から安静座位")
    print("   - カフェイン・アルコールを避ける")
    print()
    print("2. 各測定:")
    print("   - Elite HRVアプリで呼吸ガイドに従う")
    print("   - 5分間測定")
    print("   - 測定後2-3分休憩")
    print()
    print("3. データ保存:")
    print("   - ファイル名: YYYY-MM-DD HH-MM-SS.txt")
    print("   - 保存先: issues/014_rf_hrv/data/")
    print("=" * 70)


def export_protocol_for_script(protocol, output_file):
    """
    rf_hrv_analysis.py用のプロトコルデータを出力

    Parameters
    ----------
    protocol : list of dict
        測定プロトコル
    output_file : str
        出力ファイルパス
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# このコードをrf_hrv_analysis.pyのtrials変数に貼り付けてください\n\n")
        f.write("trials = [\n")
        for item in protocol:
            f.write(f"    {{'trial': {item['trial']}, 'rate': {item['rate']}, ")
            f.write(f"'file': 'YYYY-MM-DD HH-MM-SS.txt',\n")
            f.write(f"     'inhale': {item['inhale']}, 'exhale': {item['exhale']}}},\n")
        f.write("]\n")

    print(f"\n✓ スクリプト用プロトコルを保存: {output_file}")


def main():
    """メイン処理"""
    # パラメータ設定
    MIN_RATE = 3.5  # 最小呼吸レート (bpm)
    MAX_RATE = 7.0  # 最大呼吸レート (bpm)
    STEP = 0.5      # ステップサイズ (bpm)
    REPLICATE_RATE = 5.0  # 再現性確認レート (bpm)
    SEED = 42       # 乱数シード（変更すると順序が変わる）

    # 呼吸レート生成
    breathing_rates = generate_breathing_rates(MIN_RATE, MAX_RATE, STEP)

    # プロトコル生成
    protocol, replicate_rate = generate_protocol(
        breathing_rates,
        replicate_rate=REPLICATE_RATE,
        seed=SEED
    )

    # 表示
    print_protocol(protocol, replicate_rate)

    # スクリプト用に出力
    output_file = "measurement_protocol.txt"
    export_protocol_for_script(protocol, output_file)


if __name__ == '__main__':
    main()
