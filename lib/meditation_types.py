"""瞑想の種類の定義（config/meditation_types.yaml）を読む"""

from pathlib import Path
from typing import Dict, Optional

import yaml

TYPES_PATH = Path(__file__).resolve().parents[1] / 'config' / 'meditation_types.yaml'


def load_types(path: Optional[Path] = None) -> Dict[str, dict]:
    """種類 → {'label': 表示名, 'target': 目安の回数 or None}（ファイルの順）"""
    raw = yaml.safe_load(Path(path or TYPES_PATH).read_text(encoding='utf-8'))
    return {key: {'label': spec['label'], 'target': spec.get('target')} for key, spec in raw.items()}
