#!/usr/bin/env python3
"""Alpha-HRV Correlation Analysis for Latest 3 Datasets"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent.parent))

from lib.loaders.mind_monitor import load_mind_monitor_csv
from lib.loaders.selfloops import load_selfloops_csv
from lib.sensors.ecg.hrv import calculate_rmssd, calculate_hr_stats


def analyze_session(muse_file: Path, selfloops_file: Path, session_name: str):
    """Analyze correlation between Alpha waves and HRV for one session"""
    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"{'='*60}")

    # Load data
    muse_df = load_mind_monitor_csv(muse_file)
    selfloops_df = load_selfloops_csv(selfloops_file)

    # Extract Alpha values (average of all channels)
    alpha_channels = [col for col in muse_df.columns if col.startswith('Alpha_')]
    print(f"Alpha channels: {alpha_channels}")
    muse_df['Alpha_Avg'] = muse_df[alpha_channels].mean(axis=1)

    # Resample to 1-minute windows for correlation analysis
    muse_df['TimeStamp'] = pd.to_datetime(muse_df['TimeStamp'])
    muse_resampled = muse_df.set_index('TimeStamp').resample('1min')['Alpha_Avg'].mean()
    print(f"Muse resampled points: {len(muse_resampled)}")

    # Calculate HRV for 1-minute windows
    selfloops_df['TimeStamp'] = pd.to_datetime(selfloops_df['TimeStamp'])
    hrv_windows = []

    for timestamp in muse_resampled.index:
        window_start = timestamp
        window_end = timestamp + pd.Timedelta(minutes=1)

        window_data = selfloops_df[
            (selfloops_df['TimeStamp'] >= window_start) &
            (selfloops_df['TimeStamp'] < window_end)
        ]

        if len(window_data) > 10:
            rr_intervals = window_data['R-R (ms)'].values
            rmssd = calculate_rmssd(rr_intervals)
            sdnn = np.std(rr_intervals, ddof=1)
            pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(np.diff(rr_intervals)) * 100
            hrv_windows.append({
                'TimeStamp': timestamp,
                'RMSSD': rmssd,
                'SDNN': sdnn,
                'pNN50': pnn50
            })

    hrv_df = pd.DataFrame(hrv_windows).set_index('TimeStamp')
    print(f"HRV windows: {len(hrv_df)}")

    # Align data
    combined_df = pd.concat([muse_resampled, hrv_df], axis=1).dropna()
    print(f"Combined data points: {len(combined_df)}")

    if len(combined_df) < 5:
        print(f"⚠️ Insufficient data points: {len(combined_df)}")
        return None

    # Calculate correlations
    correlations = {}
    for hrv_metric in ['RMSSD', 'SDNN', 'pNN50']:
        corr, p_value = stats.pearsonr(combined_df['Alpha_Avg'], combined_df[hrv_metric])
        correlations[hrv_metric] = {'correlation': corr, 'p_value': p_value}
        print(f"\n{hrv_metric}:")
        print(f"  Correlation: {corr:.3f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

    return {
        'session': session_name,
        'n_points': len(combined_df),
        'correlations': correlations,
        'data': combined_df
    }


def main():
    # Define latest 3 datasets
    datasets = [
        {
            'name': '2026-01-23',
            'muse': Path('data/muse/mindMonitor_2026-01-23--07-40-44_259012742913319410.csv'),
            'selfloops': Path('data/selfloops/selfloops_2026-01-23--07-40-42.csv')
        },
        {
            'name': '2026-01-22',
            'muse': Path('data/muse/mindMonitor_2026-01-22--07-42-10_7649427734927614254.csv'),
            'selfloops': Path('data/selfloops/selfloops_2026-01-22--07-42-08.csv')
        },
        {
            'name': '2026-01-17',
            'muse': Path('data/muse/mindMonitor_2026-01-17--07-04-54_2846907355672700406.csv'),
            'selfloops': Path('data/selfloops/selfloops_2026-01-17--07-04-52.csv')
        }
    ]

    results = []
    for dataset in datasets:
        result = analyze_session(
            dataset['muse'],
            dataset['selfloops'],
            dataset['name']
        )
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    summary_data = []
    for result in results:
        for metric, stats_dict in result['correlations'].items():
            summary_data.append({
                'Session': result['session'],
                'Metric': metric,
                'Correlation': stats_dict['correlation'],
                'P-value': stats_dict['p_value'],
                'Significant': 'Yes' if stats_dict['p_value'] < 0.05 else 'No'
            })

    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))

    # Save results
    output_dir = Path('issues/016_alpha_hrv/alpha_hrv_output')
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / 'correlation_summary.csv', index=False)

    # Create visualization
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 5*len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results):
        data = result['data']

        for j, metric in enumerate(['RMSSD', 'SDNN', 'pNN50']):
            ax = axes[i, j]

            ax.scatter(data['Alpha_Avg'], data[metric], alpha=0.6)

            # Add regression line
            z = np.polyfit(data['Alpha_Avg'], data[metric], 1)
            p = np.poly1d(z)
            ax.plot(data['Alpha_Avg'], p(data['Alpha_Avg']), "r--", alpha=0.8)

            corr_stats = result['correlations'][metric]
            ax.set_title(f"{result['session']} - {metric}\nr={corr_stats['correlation']:.3f}, p={corr_stats['p_value']:.4f}")
            ax.set_xlabel('Alpha Wave (Avg)')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_hrv_correlations.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_dir / 'alpha_hrv_correlations.png'}")
    print(f"✓ Summary saved to {output_dir / 'correlation_summary.csv'}")


if __name__ == '__main__':
    main()
