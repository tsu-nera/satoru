#!/usr/bin/env python3
"""
Alpha Power パラメータ検証スクリプト

Museアプリの Brain Recharge Score と計算値を比較し、
最適なパラメータ（slope, intercept, offset）を推定します。

Usage:
    # 対話モード（データを1つずつ入力）
    python validate_alpha_power_params.py

    # ファイル指定モード
    python validate_alpha_power_params.py --sessions sessions.csv

    # 直接指定モード
    python validate_alpha_power_params.py \
        --data data/file1.csv --score 54 \
        --data data/file2.csv --score 62

sessions.csv の形式:
    path,muse_score
    data/mindMonitor_2025-11-22.csv,54
    data/mindMonitor_2025-11-24.csv,62
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from lib.sensors.eeg.alpha_power import (
    DEFAULT_PARAMS,
    AlphaPowerMethod,
    calculate_alpha_power,
)


@dataclass
class SessionData:
    """セッションデータ"""

    path: str
    muse_score: float
    alpha_db: Optional[float] = None


def load_session_data(path: str) -> float:
    """CSVからAlpha dBを計算"""
    df = pd.read_csv(path, low_memory=False)
    result = calculate_alpha_power(df, method=AlphaPowerMethod.OFFSET, offset=0)
    return result.alpha_db


def analyze_sessions(sessions: List[SessionData]) -> dict:
    """セッションデータを分析"""
    # Alpha dBを計算
    for s in sessions:
        if s.alpha_db is None:
            print(f"読み込み中: {s.path}")
            s.alpha_db = load_session_data(s.path)

    # DataFrameに変換
    df = pd.DataFrame(
        [{'muse': s.muse_score, 'alpha_db': s.alpha_db, 'path': s.path} for s in sessions]
    )

    results = {
        'n_sessions': len(sessions),
        'sessions': df,
    }

    # 相関分析
    if len(sessions) >= 2:
        corr = df['muse'].corr(df['alpha_db'])
        results['correlation'] = corr
    else:
        results['correlation'] = None

    # 線形回帰
    if len(sessions) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['alpha_db'], df['muse']
        )
        results['linear'] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
        }

        # 線形回帰の誤差
        df['linear_calc'] = slope * df['alpha_db'] + intercept
        df['linear_error'] = df['linear_calc'] - df['muse']
        results['linear_mae'] = df['linear_error'].abs().mean()
    else:
        results['linear'] = None
        results['linear_mae'] = None

    # オフセット方式
    offset = df['muse'].mean() - df['alpha_db'].mean()
    results['offset'] = {
        'offset': offset,
    }
    df['offset_calc'] = df['alpha_db'] + offset
    df['offset_error'] = df['offset_calc'] - df['muse']
    results['offset_mae'] = df['offset_error'].abs().mean()

    results['sessions'] = df

    return results


def print_report(results: dict):
    """分析結果をレポート出力"""
    print()
    print('=' * 60)
    print('Alpha Power パラメータ検証レポート')
    print('=' * 60)
    print()

    print(f"セッション数: {results['n_sessions']}")
    print()

    # セッション詳細
    print('--- セッション詳細 ---')
    df = results['sessions']
    for _, row in df.iterrows():
        print(f"  {Path(row['path']).name}")
        print(f"    Muse: {row['muse']:.0f} dBx, Alpha: {row['alpha_db']:.2f} dB")

    print()

    # 相関
    if results['correlation'] is not None:
        print(f"相関係数: {results['correlation']:.3f}")
        if results['correlation'] >= 0.9:
            print("  -> 非常に強い正の相関")
        elif results['correlation'] >= 0.7:
            print("  -> 強い正の相関")
        elif results['correlation'] >= 0.4:
            print("  -> 中程度の正の相関")
        else:
            print("  -> 弱い相関（データ確認推奨）")
    print()

    # 線形回帰
    print('--- 線形方式 ---')
    if results['linear'] is not None:
        lin = results['linear']
        print(f"推定式: score = {lin['slope']:.2f} × alpha_db + {lin['intercept']:.1f}")
        print(f"R² = {lin['r_squared']:.3f}, p値 = {lin['p_value']:.4f}")
        print(f"平均絶対誤差: {results['linear_mae']:.1f} dBx")
        print()
        print('検証:')
        for _, row in df.iterrows():
            calc = lin['slope'] * row['alpha_db'] + lin['intercept']
            err = calc - row['muse']
            print(f"  Muse={row['muse']:.0f}, 計算={calc:.1f}, 誤差={err:+.1f}")
    else:
        print("  データが不足しています（2セッション以上必要）")
    print()

    # オフセット方式
    print('--- オフセット方式 ---')
    off = results['offset']
    print(f"推定式: score = alpha_db + {off['offset']:.1f}")
    print(f"平均絶対誤差: {results['offset_mae']:.1f} dBx")
    print()
    print('検証:')
    for _, row in df.iterrows():
        calc = row['alpha_db'] + off['offset']
        err = calc - row['muse']
        print(f"  Muse={row['muse']:.0f}, 計算={calc:.1f}, 誤差={err:+.1f}")
    print()

    # 現在のデフォルト値との比較
    print('--- 現在のデフォルト値との比較 ---')
    current_linear = DEFAULT_PARAMS[AlphaPowerMethod.LINEAR]
    current_offset = DEFAULT_PARAMS[AlphaPowerMethod.OFFSET]

    print(f"現在の線形: slope={current_linear['slope']}, intercept={current_linear['intercept']}")
    if results['linear'] is not None:
        lin = results['linear']
        print(f"推定の線形: slope={lin['slope']:.2f}, intercept={lin['intercept']:.1f}")
        slope_diff = abs(lin['slope'] - current_linear['slope'])
        intercept_diff = abs(lin['intercept'] - current_linear['intercept'])
        if slope_diff > 1.0 or intercept_diff > 5.0:
            print("  -> 差異が大きい。パラメータ更新を検討してください。")
        else:
            print("  -> 現在のパラメータで概ね良好です。")
    print()

    print(f"現在のオフセット: {current_offset['offset']}")
    print(f"推定のオフセット: {off['offset']:.1f}")
    print()

    # 推奨
    print('--- 推奨 ---')
    if results['linear'] is not None and results['linear_mae'] < results['offset_mae']:
        print("線形方式の方が精度が高いです（推奨）")
    else:
        print("オフセット方式の方がシンプルで精度も許容範囲です")

    if results['n_sessions'] < 5:
        print(f"注意: サンプル数が少ないです（{results['n_sessions']}件）。追加データでの検証を推奨します。")

    print()
    print('=' * 60)


def interactive_mode():
    """対話モードでデータを入力"""
    print("Alpha Power パラメータ検証ツール")
    print("データを入力してください（終了するには空行を入力）")
    print()

    sessions = []
    while True:
        path = input("CSVファイルパス: ").strip()
        if not path:
            break

        if not Path(path).exists():
            print(f"  ファイルが見つかりません: {path}")
            continue

        try:
            score = float(input("Museアプリのスコア (dBx): ").strip())
        except ValueError:
            print("  無効な数値です")
            continue

        sessions.append(SessionData(path=path, muse_score=score))
        print(f"  追加しました（計 {len(sessions)} 件）")
        print()

    return sessions


def main():
    parser = argparse.ArgumentParser(
        description='Alpha Power パラメータ検証',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--sessions',
        type=str,
        help='セッション一覧CSVファイル (path,muse_score)',
    )
    parser.add_argument(
        '--data',
        type=str,
        action='append',
        help='データCSVファイルパス（--scoreとペアで複数指定可）',
    )
    parser.add_argument(
        '--score',
        type=float,
        action='append',
        help='Museアプリのスコア（--dataとペアで複数指定可）',
    )

    args = parser.parse_args()

    sessions = []

    # セッションCSVから読み込み
    if args.sessions:
        sessions_df = pd.read_csv(args.sessions)
        for _, row in sessions_df.iterrows():
            sessions.append(SessionData(path=row['path'], muse_score=row['muse_score']))

    # 直接指定から読み込み
    if args.data and args.score:
        if len(args.data) != len(args.score):
            print("エラー: --data と --score の数が一致しません")
            sys.exit(1)
        for path, score in zip(args.data, args.score):
            sessions.append(SessionData(path=path, muse_score=score))

    # 引数がない場合は対話モード
    if not sessions:
        sessions = interactive_mode()

    if len(sessions) < 2:
        print("エラー: 最低2セッション必要です")
        sys.exit(1)

    # 分析実行
    results = analyze_sessions(sessions)
    print_report(results)


if __name__ == '__main__':
    main()
