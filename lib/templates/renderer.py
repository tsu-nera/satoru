"""
レポートテンプレートレンダラー

Jinja2を使用して瞑想分析Markdownレポートを生成
"""

from datetime import datetime
from pathlib import Path
import logging

from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd

logger = logging.getLogger(__name__)


class MeditationReportRenderer:
    """瞑想分析レポートのテンプレートレンダラー"""

    def __init__(self, template_dir=None):
        """
        Parameters
        ----------
        template_dir : str or Path, optional
            テンプレートディレクトリのパス
            Noneの場合はプロジェクトルート/templatesを使用
        """
        if template_dir is None:
            # プロジェクトルート/templatesをデフォルトに
            template_dir = Path(__file__).resolve().parents[2] / 'templates'

        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape([]),  # Markdownなのでautoescape無効
            trim_blocks=True,          # ブロックタグの改行を削除
            lstrip_blocks=True         # ブロック前の空白を削除
        )

        # カスタムフィルタを登録
        self._register_filters()

    def _register_filters(self):
        """カスタムフィルタをJinja2環境に登録"""
        from .filters import (
            number_format,
            format_percent,
            format_db,
            format_hz,
            format_timestamp,
            format_duration,
            format_change,
            df_to_markdown,
            format_score,
        )

        self.env.filters['number_format'] = number_format
        self.env.filters['format_percent'] = format_percent
        self.env.filters['format_db'] = format_db
        self.env.filters['format_hz'] = format_hz
        self.env.filters['format_timestamp'] = format_timestamp
        self.env.filters['format_duration'] = format_duration
        self.env.filters['format_change'] = format_change
        self.env.filters['df_to_markdown'] = df_to_markdown
        self.env.filters['format_score'] = format_score

    def render_report(self, context):
        """
        瞑想分析レポートを生成

        Parameters
        ----------
        context : dict
            テンプレートコンテキスト
            必須キー: generated_at, data_file, start_time, end_time, duration_sec

        Returns
        -------
        str
            レンダリングされたMarkdown

        Raises
        ------
        jinja2.TemplateNotFound
            テンプレートファイルが見つからない場合
        jinja2.TemplateSyntaxError
            テンプレート構文エラーがある場合
        """
        try:
            template = self.env.get_template('meditation/report.md.j2')
            return template.render(**context)
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            raise

    def build_context(self, results, data_path):
        """
        results辞書からテンプレート用のcontextを構築

        Parameters
        ----------
        results : dict
            分析結果辞書（generate_report.pyから渡される）
        data_path : Path
            データファイルのパス

        Returns
        -------
        dict
            テンプレート用のcontext辞書
        """
        info = results.get('data_info', {})

        # ヘッダー情報
        context = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_file': data_path.name,
            'start_time': info.get('start_time'),
            'end_time': info.get('end_time'),
            'duration_sec': info.get('duration_sec'),
        }

        # 接続品質 (HSI)
        if 'hsi_stats' in results:
            hsi_data = results['hsi_stats']
            context['hsi'] = {
                'overall_quality': hsi_data.get('overall_quality'),
                'good_ratio': hsi_data.get('good_ratio', 0.0),
                'statistics': hsi_data.get('statistics'),
            }

        # 生データプレビュー
        if 'raw_preview_img' in results:
            context['raw_preview_img'] = results['raw_preview_img']

        # 分析サマリー
        summary = {}
        if 'session_score' in results:
            summary['session_score'] = results['session_score']
        if 'session_score_breakdown' in results:
            summary['score_breakdown'] = results['session_score_breakdown']
        if 'mean_metrics' in results:
            summary['mean_metrics'] = results['mean_metrics']
        if 'best_metrics' in results:
            summary['best_metrics'] = results['best_metrics']
        if 'segment_peak_range' in results:
            summary['peak_time_range'] = results['segment_peak_range']
        if 'segment_peak_score' in results:
            summary['peak_score'] = results['segment_peak_score']

        if summary:
            context['summary'] = summary

        # 周波数帯域分析
        frequency = {}
        if 'band_power_img' in results:
            frequency['band_power_img'] = results['band_power_img']
        if 'psd_img' in results:
            frequency['psd_img'] = results['psd_img']
        if 'spectrogram_img' in results:
            frequency['spectrogram_img'] = results['spectrogram_img']
        if 'harmonics_img' in results:
            frequency['harmonics_img'] = results['harmonics_img']
        if 'harmonics_table' in results:
            harmonics_table = results['harmonics_table']
            # 4Hz倍数アーチファクトを除外
            if not harmonics_table.empty and 'is_4hz_harmonic' in harmonics_table.columns:
                display_table = harmonics_table[~harmonics_table['is_4hz_harmonic']].drop(columns=['is_4hz_harmonic'])
            else:
                display_table = harmonics_table
            frequency['harmonics_table_display'] = display_table

        if frequency:
            context['frequency'] = frequency

        # 特徴的指標
        indicators = {}

        # Frontal Midline Theta
        if 'frontal_theta_img' in results or 'frontal_theta_stats' in results:
            indicators['fmtheta'] = {
                'img': results.get('frontal_theta_img'),
                'stats': results.get('frontal_theta_stats'),
                'increase': results.get('frontal_theta_increase'),
            }

        # SMR
        if 'smr_img' in results or 'smr_stats' in results:
            indicators['smr'] = {
                'img': results.get('smr_img'),
                'stats': results.get('smr_stats'),
                'increase': results.get('smr_increase'),
            }

        # IAF/PAF
        if 'paf_img' in results or 'paf_summary' in results or 'iaf' in results:
            indicators['paf'] = {
                'img': results.get('paf_img'),
                'summary': results.get('paf_summary'),
                'iaf': results.get('iaf'),
            }

        # Alpha Power
        if 'alpha_power_score' in results:
            indicators['alpha_power'] = {
                'score': results.get('alpha_power_score'),
                'db': results.get('alpha_power_db'),
                'stats': results.get('alpha_power_stats'),
            }

        # FAA
        if 'faa_img' in results or 'faa_stats' in results:
            indicators['faa'] = {
                'img': results.get('faa_img'),
                'stats': results.get('faa_stats'),
            }

        # Spectral Entropy
        if 'spectral_entropy_stats' in results:
            indicators['spectral_entropy'] = {
                'stats': results.get('spectral_entropy_stats'),
                'change': results.get('spectral_entropy_change'),
            }

        # Band Ratios
        if 'band_ratios_img' in results:
            indicators['band_ratios'] = {
                'img': results.get('band_ratios_img'),
            }

        if indicators:
            context['indicators'] = indicators

        # fNIRS
        if 'fnirs_stats' in results or 'fnirs_img' in results:
            context['fnirs'] = {
                'img': results.get('fnirs_img'),
                'stats': results.get('fnirs_stats'),
                'laterality': results.get('fnirs_laterality'),
            }

        # ECG/ANS
        ecg = {}
        if 'hrv_img' in results:
            ecg['hrv_img'] = results['hrv_img']
        if 'hrv_freq_img' in results:
            ecg['hrv_freq_img'] = results['hrv_freq_img']

        # hrv_statsを時間領域と周波数領域に分割
        if 'hrv_stats' in results:
            hrv_stats_df = results['hrv_stats']
            if not hrv_stats_df.empty and 'Domain' in hrv_stats_df.columns:
                time_domain_df = hrv_stats_df[hrv_stats_df['Domain'] == 'Time Domain']
                freq_domain_df = hrv_stats_df[hrv_stats_df['Domain'] == 'Frequency Domain']

                # Domain列を除外してテーブル表示
                if not time_domain_df.empty:
                    ecg['hrv_time_stats'] = time_domain_df.drop(columns=['Domain'])
                if not freq_domain_df.empty:
                    ecg['hrv_freq_stats'] = freq_domain_df.drop(columns=['Domain'])

        if 'respiratory_period' in results:
            ecg['respiratory_period'] = results['respiratory_period']
        if 'rbp_stats' in results:
            ecg['rbp_stats'] = results['rbp_stats']

        if ecg:
            context['ecg'] = ecg

        # 姿勢分析
        if 'motion_img' in results or 'posture_summary' in results:
            context['posture'] = {
                'motion_img': results.get('motion_img'),
                'summary_table': results.get('posture_summary'),
                'detail_table': results.get('posture_detail'),
            }

        # 時間セグメント分析
        if 'segment_plot' in results or 'segment_table' in results:
            context['segments'] = {
                'plot_img': results.get('segment_plot'),
                'band_power_table': results.get('band_power_table'),
                'metrics_table': results.get('metrics_table'),
            }

        return context
