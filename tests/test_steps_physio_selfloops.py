"""データ行が0の SelfLoops ファイルを HRV 無しとして扱うことのテスト"""

from pathlib import Path

from lib.loaders.selfloops import has_selfloops_rows
from lib.report.steps_physio import analyze_hrv

EMPTY_SELFLOOPS = '05 11月 2026 07:53:44\ntimestamp,bpm,rr\n'


def write_selfloops(tmp_path: Path, content: str) -> Path:
    path = tmp_path / 'selfloops_2026-11-05--07-53-44.csv'
    path.write_text(content, encoding='utf-8')
    return path


class TestHasSelfloopsRows:
    def test_header_only_file_has_no_rows(self, tmp_path):
        assert has_selfloops_rows(str(write_selfloops(tmp_path, EMPTY_SELFLOOPS))) is False

    def test_trailing_blank_lines_are_not_rows(self, tmp_path):
        assert has_selfloops_rows(str(write_selfloops(tmp_path, EMPTY_SELFLOOPS + '\n\n'))) is False

    def test_file_with_data_row(self, tmp_path):
        path = write_selfloops(tmp_path, EMPTY_SELFLOOPS + '0,60,1000\n')
        assert has_selfloops_rows(str(path)) is True


class TestAnalyzeHrvEmptySelfloops:
    def test_skips_without_error(self, tmp_path, capsys):
        selfloops_path = write_selfloops(tmp_path, EMPTY_SELFLOOPS)
        results: dict = {}

        # analysis_step は例外時に None を返すので、(None, None) なら例外を経ずにスキップできている
        assert analyze_hrv(None, tmp_path, results, selfloops_path, None) == (None, None)

        out = capsys.readouterr().out
        assert '失敗しました' not in out
        assert 'IndexError' not in out
        assert out.count('HRV解析スキップ') == 1
        assert 'hrv_stats' not in results
