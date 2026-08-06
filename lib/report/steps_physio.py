"""
fNIRS / 動作検出+心拍数 / HRV / 呼吸の解析ステップ

各関数は `results` dict を引数で受け取って直接 mutate し、後続ブロックが使う
中間オブジェクトを return する（部分的な results 書き込み挙動を維持するため、
results をコピーして最後にマージしてはいけない）。
"""

import matplotlib.pyplot as plt
import pandas as pd

from lib import (
    analyze_fnirs as compute_fnirs,
)
from lib import (
    analyze_motion as compute_motion,
)
from lib import (
    get_heart_rate_data,
    get_optics_data,
)
from lib.visualization import (
    create_motion_stats_table,
    plot_fnirs_muse_style,
    plot_motion_heart_rate,
)

from .step import analysis_step


@analysis_step('fNIRS解析', exceptions=(KeyError,))
def analyze_fnirs(df, img_dir, results):
    """fNIRS統計・時系列プロットを計算し、results に格納する。"""
    fnirs_results = None
    optics_data = get_optics_data(df)
    if optics_data and len(optics_data['time']) > 0:
        print('計算中: fNIRS統計...')
        fnirs_results = compute_fnirs(optics_data)

        # fNIRS統計をDataFrame化（lateralityを除外してleft/rightのみ）
        hemisphere_stats = {k: v for k, v in fnirs_results['stats'].items() if k != 'laterality'}
        df_stats = pd.DataFrame(hemisphere_stats).T
        df_stats = df_stats.rename(
            index={"left": "左半球", "right": "右半球"},
            columns={
                "hbo_mean": "HbO平均", "hbo_min": "HbO最小", "hbo_max": "HbO最大",
                "hbr_mean": "HbR平均", "hbr_min": "HbR最小", "hbr_max": "HbR最大",
                "hbt_mean": "HbT平均", "hbd_mean": "HbD平均",
            },
        )
        results['fnirs_stats'] = df_stats[
            ["HbO平均", "HbO最小", "HbO最大", "HbR平均", "HbR最小", "HbR最大", "HbT平均", "HbD平均"]
        ]
        results['fnirs_laterality'] = fnirs_results['stats']['laterality']

        print('プロット中: fNIRS時系列...')
        fig_fnirs, _ = plot_fnirs_muse_style(fnirs_results)
        fnirs_img_name = 'fnirs_muse_style.png'
        fig_fnirs.savefig(img_dir / fnirs_img_name, dpi=150, bbox_inches='tight')
        plt.close(fig_fnirs)
        results['fnirs_img'] = fnirs_img_name

    return fnirs_results


@analysis_step('心拍数データまたは動作検出解析')
def analyze_motion_and_hr(df, img_dir, results, selfloops_data):
    """心拍数データ取得・動作検出（加速度・ジャイロ）を行い、results に格納する。"""
    # 心拍数データ取得（Selfloops優先、なければMuse）
    if selfloops_data and selfloops_data.exists():
        # Selfloopsデータから心拍数を取得
        print(f'Loading Selfloops HR data: {selfloops_data}')
        # 別名で束縛する。同名でimportするとget_heart_rate_dataが関数スコープの
        # ローカル変数になり、else側（Muse経路）がUnboundLocalErrorで落ちる
        from lib.loaders.base import get_heart_rate_data as get_hr_from_ecg
        from lib.loaders.selfloops import load_selfloops_csv
        sl_df = load_selfloops_csv(str(selfloops_data), warmup_seconds=0.0)
        hr_data = get_hr_from_ecg(sl_df)
        hr_data_source = 'Selfloops'
    else:
        # Museデータから心拍数を取得
        hr_data = get_heart_rate_data(df)
        hr_data_source = 'Muse'

    results['hr_data_source'] = hr_data_source  # レポート表示用

    # 動作検出（10秒間隔）
    print('計算中: 動作検出（加速度・ジャイロ）...')
    motion_result = compute_motion(df, interval='10s')

    # 統計情報をDataFrame化（心拍数情報を含む）
    motion_stats = create_motion_stats_table(motion_result, hr_data=hr_data)

    # 時系列プロット（動作検出のみ、心拍数は含まない）
    print('プロット中: 動作検出時系列...')
    motion_img_name = 'motion_only.png'
    fig_motion, _ = plot_motion_heart_rate(motion_result, hr_data=None, df=df)
    fig_motion.savefig(img_dir / motion_img_name, dpi=150, bbox_inches='tight')
    plt.close(fig_motion)

    # postureネスト構造で保存（テンプレート用）
    results['posture'] = {
        'motion_img': motion_img_name,
        'summary_table': motion_stats,
    }

    # 内部処理用データ
    results['motion_ratio'] = motion_result['motion_ratio']

    return hr_data


@analysis_step('HRV解析', show_traceback=True)
def analyze_hrv(df, img_dir, results, selfloops_data, hr_data):
    """自律神経系分析（HRV）。成功時は (hrv_result, hrv_data) を返す。"""
    hrv_result = None
    hrv_data = None

    if selfloops_data and selfloops_data.exists():
        print('計算中: HRV解析（自律神経系）...')
        from lib.loaders.selfloops import get_hrv_data, load_selfloops_csv
        from lib.sensors.ecg.hrv import calculate_hrv_standard_set

        # Selfloopsファイルパスを保存（レポート表示用）
        results['selfloops_file'] = selfloops_data.name

        sl_df = load_selfloops_csv(str(selfloops_data), warmup_seconds=60.0)
        hrv_data = get_hrv_data(sl_df, clean_artifacts=True)

        # R-R間隔の品質情報を保存
        if 'quality_stats' in hrv_data:
            results['rr_quality_stats'] = hrv_data['quality_stats']
            quality_stats = hrv_data['quality_stats']
            print(f"  R-R間隔品質: {quality_stats['quality_rate']:.1f}% "
                  f"({quality_stats['outliers_count']}/{quality_stats['total_intervals']} outliers)")

        # セッション時間チェック
        total_duration = hrv_data['time'][-1] - hrv_data['time'][0]
        if total_duration < 180:
            print(f'⚠️  HRV解析スキップ: 記録時間が短すぎます（{total_duration:.0f}秒 < 180秒）')
        else:
            hrv_result = calculate_hrv_standard_set(hrv_data)
            results['hrv_stats'] = hrv_result.statistics
            results['hrv_result'] = hrv_result  # 時系列データ用に保存

            print('プロット中: HRV時系列...')
            from lib.sensors.ecg.analysis import analyze_hrv as compute_hrv_indices
            from lib.sensors.ecg.visualization.hrv_plot import plot_hrv_frequency, plot_hrv_time_series

            hrv_img_name = 'hrv_time_series.png'
            plot_hrv_time_series(
                hrv_result,
                img_path=str(img_dir / hrv_img_name),
                title='HRV Time Series Analysis',
                hr_data=hr_data
            )
            results['hrv_img'] = hrv_img_name

            # HRV周波数解析
            print('プロット中: HRV周波数解析...')
            hrv_freq_img_name = 'hrv_frequency.png'
            hrv_indices = compute_hrv_indices(hrv_data, show=False)
            plot_hrv_frequency(
                hrv_data,
                hrv_indices=hrv_indices,
                img_path=str(img_dir / hrv_freq_img_name)
            )
            results['hrv_freq_img'] = hrv_freq_img_name

    return hrv_result, hrv_data


@analysis_step('呼吸分析', show_traceback=True)
def analyze_respiration(hrv_data, results):
    """呼吸分析（ECG-Derived Respiration）。"""
    respiration_result = None

    if hrv_data is not None:
        print('計算中: 呼吸分析（ECG-Derived Respiration）...')
        from lib.sensors.ecg.respiration import calculate_breathing_rate

        respiration_result = calculate_breathing_rate(
            hrv_data,
            target_fs=8.0,
            peak_distance=8.0,
            window_minutes=3.0
        )

        # 結果を保存（内部処理用）
        results['respiration_result'] = respiration_result

        print(f'  平均BR: {respiration_result.breathing_rate:.1f} bpm')

    return respiration_result
