"""
`.env` の読み込み

Pythonスクリプトを `uv run python scripts/xxx.py` で直接起動した場合、シェル
スクリプト（`scripts/download_data.sh` など）を経由しないため `.env` が読まれず、
`os.environ.get(...)` が黙って None を返す。argparse の `default=` は import 時に
評価されるので、**argparse より前**に `load_env()` を呼ぶこと。

既存の環境変数は上書きしない（シェルで export した値が優先される）。
"""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env(dotenv_path: Path | None = None) -> bool:
    """
    リポジトリルートの `.env` を環境変数に読み込む。

    Parameters
    ----------
    dotenv_path : Path, optional
        読み込む `.env` のパス。省略時はリポジトリルートの `.env`。

    Returns
    -------
    bool
        ファイルが存在して読み込まれた場合 True、存在しない場合 False。
    """
    path = dotenv_path if dotenv_path is not None else PROJECT_ROOT / '.env'
    if not path.is_file():
        return False
    return load_dotenv(path, override=False)
