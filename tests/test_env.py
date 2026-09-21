"""lib.env.load_env のテスト"""

import os

from lib.env import load_env


def test_load_env_reads_dotenv(tmp_path, monkeypatch):
    dotenv = tmp_path / '.env'
    dotenv.write_text('SATORU_TEST_KEY=from_dotenv\n')
    monkeypatch.delenv('SATORU_TEST_KEY', raising=False)

    assert load_env(dotenv) is True
    assert os.environ['SATORU_TEST_KEY'] == 'from_dotenv'


def test_load_env_does_not_override_shell_env(tmp_path, monkeypatch):
    """シェルで export 済みの値を .env が上書きしない"""
    dotenv = tmp_path / '.env'
    dotenv.write_text('SATORU_TEST_KEY=from_dotenv\n')
    monkeypatch.setenv('SATORU_TEST_KEY', 'from_shell')

    load_env(dotenv)
    assert os.environ['SATORU_TEST_KEY'] == 'from_shell'


def test_load_env_missing_file(tmp_path):
    assert load_env(tmp_path / 'nonexistent.env') is False
