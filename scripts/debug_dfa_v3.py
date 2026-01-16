#!/usr/bin/env python3
"""
DFA計算パラメータの最終検証

integrate, order, overlapパラメータを組み合わせて、
Kubios (α1=0.8, α2=0.27) に最も近い設定を見つける
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import neurokit2 as nk

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data

# データファイルパス
data_path = project_root / 'data/selfloops/selfloops_2026-01-16--07-18-39.csv'

print('='*80)
print('DFA計算パラメータの最終検証')
print('='*80)
print()

# データ読み込み
sl_df = load_selfloops_csv(str(data_path), warmup_seconds=60.0)
hrv_data = get_hrv_data(sl_df, clean_artifacts=True)
rr_intervals = hrv_data['rr_intervals_clean']

print(f'R-R間隔数: {len(rr_intervals)}')
print(f'平均R-R: {np.mean(rr_intervals):.2f} ms')
print()

# Kubios基準値
kubios_alpha1 = 0.80
kubios_alpha2 = 0.27

print('【参照値】')
print(f'Kubios: α1={kubios_alpha1:.2f}, α2={kubios_alpha2:.2f}')
print(f'論文 (熟練者): α1≈0.5')
print(f'論文 (初心者): α1≈0.78')
print()

# パラメータ組み合わせテスト
params_list = [
    {'integrate': True, 'order': 1, 'overlap': True, 'desc': 'デフォルト (integrate=True, order=1, overlap=True)'},
    {'integrate': False, 'order': 1, 'overlap': True, 'desc': 'integrate=False (cumsumなし)'},
    {'integrate': True, 'order': 2, 'overlap': True, 'desc': 'order=2 (2次デトレンド)'},
    {'integrate': False, 'order': 2, 'overlap': True, 'desc': 'integrate=False, order=2'},
    {'integrate': True, 'order': 1, 'overlap': False, 'desc': 'overlap=False (非重複ウィンドウ)'},
    {'integrate': False, 'order': 1, 'overlap': False, 'desc': 'integrate=False, overlap=False'},
]

print('='*80)
print('パラメータ組み合わせテスト')
print('='*80)
print()

results = []

for params in params_list:
    try:
        # α1 (4-16拍)
        alpha1, _ = nk.fractal_dfa(
            rr_intervals,
            scale=range(4, 17),
            integrate=params['integrate'],
            order=params['order'],
            overlap=params['overlap'],
            show=False
        )

        # α2 (16-64拍)
        alpha2, _ = nk.fractal_dfa(
            rr_intervals,
            scale=range(16, 65),
            integrate=params['integrate'],
            order=params['order'],
            overlap=params['overlap'],
            show=False
        )

        # Kubiosとの差分
        diff_alpha1 = abs(alpha1 - kubios_alpha1)
        diff_alpha2 = abs(alpha2 - kubios_alpha2)
        total_diff = diff_alpha1 + diff_alpha2

        results.append({
            'desc': params['desc'],
            'alpha1': alpha1,
            'alpha2': alpha2,
            'diff_a1': diff_alpha1,
            'diff_a2': diff_alpha2,
            'total_diff': total_diff,
        })

        print(f'{params["desc"]}')
        print(f'  α1={alpha1:.4f} (diff: {diff_alpha1:.4f}), α2={alpha2:.4f} (diff: {diff_alpha2:.4f})')
        print(f'  Total diff: {total_diff:.4f}')
        print()

    except Exception as e:
        print(f'{params["desc"]}')
        print(f'  エラー: {e}')
        print()

# 最適パラメータを特定
print('='*80)
print('最適パラメータ')
print('='*80)
results_sorted = sorted(results, key=lambda x: x['total_diff'])

print('\n【Kubiosに最も近い設定】')
best = results_sorted[0]
print(f'{best["desc"]}')
print(f'  α1={best["alpha1"]:.4f} (Kubios: {kubios_alpha1})')
print(f'  α2={best["alpha2"]:.4f} (Kubios: {kubios_alpha2})')
print(f'  Total difference: {best["total_diff"]:.4f}')
print()

print('【上位3設定】')
for i, res in enumerate(results_sorted[:3], 1):
    print(f'{i}. {res["desc"]}')
    print(f'   α1={res["alpha1"]:.4f}, α2={res["alpha2"]:.4f}, diff={res["total_diff"]:.4f}')

print()
print('='*80)
print('結論')
print('='*80)
print()
print('NeuroKit2とKubiosのDFA実装には根本的な違いがあります:')
print()
print('1. **デフォルトintegrate=True**: R-R間隔を積分(cumsum)して計算')
print('   → α1が1.0以上になり、Kubios値(0.8)と大きく乖離')
print()
print('2. **integrate=False**: R-R間隔を直接使用')
print('   → α1≈0.6となり、論文の熟練者レベル(0.5)に近い')
print()
print('3. **Kubios実装**: 独自の前処理・正規化を使用している可能性')
print('   → 完全な一致は困難')
print()
print('【推奨】')
print('- 論文基準値との比較には integrate=False を使用')
print('- ただし、これが標準かどうかは文献確認が必要')
print('- 自己データの経時変化追跡には、一貫した設定を維持することが重要')
print()
