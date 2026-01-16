#!/usr/bin/env python3
"""
DFA計算の詳細をデバッグするスクリプト

Hoshiyama et al. (2008)の論文値との比較のため、
NeuroKit2のDFA計算パラメータを詳しく調査します。
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

# データファイルパス（最新のテストデータ）
data_path = project_root / 'data/selfloops/selfloops_2026-01-16--07-18-39.csv'

print('='*60)
print('DFA計算デバッグ')
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
print(f'R-R間隔範囲: {np.min(rr_intervals):.2f} - {np.max(rr_intervals):.2f} ms')
print()

# デフォルトのDFA計算
print('1. デフォルトDFA計算')
print('-'*60)
peaks = nk.intervals_to_peaks(rr_intervals, sampling_rate=1000)
dfa_default, info_default = nk.fractal_dfa(peaks, sampling_rate=1000, show=False)

print(f'DFA α (default): {dfa_default:.4f}')
print(f'info_default keys: {list(info_default.keys())}')
if 'scale' in info_default:
    print(f'スケール範囲: {info_default["scale"][:3]} ... {info_default["scale"][-3:]}')
    print(f'スケール数: {len(info_default["scale"])}')
print()

# 短期スケール(4-11拍)に限定したDFA計算 → α1
print('2. 短期スケール(4-11拍)のDFA α1')
print('-'*60)
dfa_alpha1, info_alpha1 = nk.fractal_dfa(
    peaks,
    sampling_rate=1000,
    scale=range(4, 12),  # 4-11拍
    show=False
)
print(f'DFA α1 (4-11拍): {dfa_alpha1:.4f}')
print(f'スケール: {list(info_alpha1["scale"])}')
print()

# 長期スケール(12-64拍)のDFA → α2
print('3. 長期スケール(12-64拍)のDFA α2')
print('-'*60)
dfa_alpha2, info_alpha2 = nk.fractal_dfa(
    peaks,
    sampling_rate=1000,
    scale=range(12, 65),  # 12-64拍
    show=False
)
print(f'DFA α2 (12-64拍): {dfa_alpha2:.4f}')
print(f'スケール範囲: {list(info_alpha2["scale"][:3])} ... {list(info_alpha2["scale"][-3:])}')
print()

# NeuroKit2のhrv()関数で計算されたα1, α2を確認
print('4. NeuroKit2 hrv()関数の結果')
print('-'*60)
hrv_result = nk.hrv(peaks, sampling_rate=1000, show=False)
if 'HRV_DFA_alpha1' in hrv_result.columns:
    print(f'HRV_DFA_alpha1: {hrv_result["HRV_DFA_alpha1"].iloc[0]:.4f}')
if 'HRV_DFA_alpha2' in hrv_result.columns:
    print(f'HRV_DFA_alpha2: {hrv_result["HRV_DFA_alpha2"].iloc[0]:.4f}')
print()

# 論文の基準値と比較
print('5. 論文(Hoshiyama et al., 2008)との比較')
print('-'*60)
print('基準値:')
print('  熟練瞑想者: α1 ≈ 0.5')
print('  初心者: α1 ≈ 0.78')
print()
print(f'あなたのデータ: α1 = {hrv_result["HRV_DFA_alpha1"].iloc[0]:.4f}')
print()

if hrv_result["HRV_DFA_alpha1"].iloc[0] < 0.5:
    print('→ 反相関（平均への回帰）')
elif 0.5 <= hrv_result["HRV_DFA_alpha1"].iloc[0] < 0.6:
    print('→ ランダムウォーク（熟練瞑想者レベル）')
elif 0.6 <= hrv_result["HRV_DFA_alpha1"].iloc[0] < 0.9:
    print('→ 中程度の長期相関')
elif 0.9 <= hrv_result["HRV_DFA_alpha1"].iloc[0] < 1.0:
    print('→ 強い長期相関')
else:
    print('→ 1/fノイズまたはそれ以上')
print()

# 可視化
print('6. DFAプロットの可視化')
print('-'*60)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# デフォルト
_, info_def_vis = nk.fractal_dfa(peaks, sampling_rate=1000, show=True)
plt.savefig(project_root / 'tmp/hrv_test/dfa_default.png', dpi=150, bbox_inches='tight')
plt.close()

# α1
_, info_a1_vis = nk.fractal_dfa(peaks, sampling_rate=1000, scale=range(4, 12), show=True)
plt.savefig(project_root / 'tmp/hrv_test/dfa_alpha1.png', dpi=150, bbox_inches='tight')
plt.close()

# α2
_, info_a2_vis = nk.fractal_dfa(peaks, sampling_rate=1000, scale=range(12, 65), show=True)
plt.savefig(project_root / 'tmp/hrv_test/dfa_alpha2.png', dpi=150, bbox_inches='tight')
plt.close()

print('✓ プロット保存完了:')
print('  - tmp/hrv_test/dfa_default.png')
print('  - tmp/hrv_test/dfa_alpha1.png')
print('  - tmp/hrv_test/dfa_alpha2.png')
print()

print('='*60)
print('デバッグ完了')
print('='*60)
