#!/usr/bin/env python3
"""
DFA計算の詳細デバッグ v2

R-R間隔を直接渡して、NeuroKit2とKubiosの差異を調査
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data

# データファイルパス
data_path = project_root / 'data/selfloops/selfloops_2026-01-16--07-18-39.csv'

print('='*60)
print('DFA計算デバッグ v2 - R-R間隔を直接使用')
print('='*60)
print()

# データ読み込み
print(f'Loading: {data_path}')
sl_df = load_selfloops_csv(str(data_path), warmup_seconds=60.0)

# HRVデータ取得
hrv_data = get_hrv_data(sl_df, clean_artifacts=True)
rr_intervals = hrv_data['rr_intervals_clean']

print(f'R-R間隔数: {len(rr_intervals)}')
print(f'平均R-R間隔: {np.mean(rr_intervals):.2f} ms')
print()

# 1. NeuroKit2のデフォルトスケール
print('1. NeuroKit2 デフォルトスケールでのDFA')
print('-'*60)
# integrate=Trueがデフォルト（cumsum実行）
dfa1, info1 = nk.fractal_dfa(rr_intervals, show=False)
print(f'DFA α: {dfa1:.4f}')
print(f'スケール範囲: {info1["scale"][0]}-{info1["scale"][-1]} (n={len(info1["scale"])})')
print()

# 2. Kubiosと同じ設定を試す
# Kubios: Short-term (α1) = 4-16 beats, Long-term (α2) = 16-64 beats
print('2. Kubios互換スケール設定')
print('-'*60)

# α1: 4-16 beats
dfa_alpha1_kubios, info_alpha1 = nk.fractal_dfa(
    rr_intervals,
    scale=range(4, 17),  # 4-16拍
    show=False
)
print(f'DFA α1 (4-16拍): {dfa_alpha1_kubios:.4f}')

# α2: 16-64 beats
dfa_alpha2_kubios, info_alpha2 = nk.fractal_dfa(
    rr_intervals,
    scale=range(16, 65),  # 16-64拍
    show=False
)
print(f'DFA α2 (16-64拍): {dfa_alpha2_kubios:.4f}')
print()

# 3. NeuroKit2のhrv()関数の内部スケール確認
print('3. NeuroKit2 hrv()関数の結果')
print('-'*60)
peaks = nk.intervals_to_peaks(rr_intervals, sampling_rate=1000)
hrv_result = nk.hrv(peaks, sampling_rate=1000, show=False)

print(f'HRV_DFA_alpha1: {hrv_result["HRV_DFA_alpha1"].iloc[0]:.4f}')
print(f'HRV_DFA_alpha2: {hrv_result["HRV_DFA_alpha2"].iloc[0]:.4f}')
print()

# 4. integrate=Falseで試す（cumsum無し）
print('4. integrate=False (cumsum無し)')
print('-'*60)
dfa_no_integrate, info_no_int = nk.fractal_dfa(
    rr_intervals,
    integrate=False,
    scale=range(4, 17),
    show=False
)
print(f'DFA α1 (integrate=False): {dfa_no_integrate:.4f}')
print()

# 5. 比較表
print('5. 結果まとめ')
print('='*60)
print('| ツール/設定                | α1      | α2      |')
print('|:--------------------------|:--------|:--------|')
print(f'| **Kubios (参照)**         | 0.8000  | 0.2700  |')
print(f'| NeuroKit2 hrv()           | {hrv_result["HRV_DFA_alpha1"].iloc[0]:.4f}  | {hrv_result["HRV_DFA_alpha2"].iloc[0]:.4f}  |')
print(f'| NK2 Kubios互換(4-16/16-64)| {dfa_alpha1_kubios:.4f}  | {dfa_alpha2_kubios:.4f}  |')
print(f'| NK2 integrate=False       | {dfa_no_integrate:.4f}  | -       |')
print()

# 6. 可能性のある原因
print('6. 考えられる原因')
print('-'*60)
print('A. スケール範囲の違い')
print('   - Kubios: α1=4-16, α2=16-64')
print('   - NeuroKit2 hrv(): 異なるスケール範囲の可能性')
print()
print('B. デトレンド次数（order）の違い')
print('   - NeuroKit2: デフォルトorder=1 (線形)')
print('   - Kubios: 異なるorder値の可能性')
print()
print('C. integrate (累積和) の適用')
print('   - NeuroKit2: デフォルトTrue')
print('   - Kubios: 不明')
print()
print('D. R-R間隔の単位・前処理')
print('   - NeuroKit2: ms単位で計算')
print('   - Kubios: 秒単位の可能性')
print()

# 7. orderパラメータを変えて試す
print('7. デトレンド次数(order)を変えて試す')
print('-'*60)
for order in [1, 2, 3]:
    dfa_ord, _ = nk.fractal_dfa(
        rr_intervals,
        scale=range(4, 17),
        order=order,
        show=False
    )
    print(f'order={order}: α1 = {dfa_ord:.4f}')
print()

print('='*60)
print('デバッグ完了')
print('='*60)
