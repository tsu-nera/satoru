"""
解析ステップ共通のエラーハンドリングヘルパー

`run_full_analysis` の各解析ブロックは、失敗しても後続処理を継続できるよう
`try: ... except Exception as exc: print(f'警告: ...')` という同型のパターンで
書かれている。このモジュールはそのパターンをデコレータに集約する。
"""

import functools
import traceback


def analysis_step(label, *, show_traceback=False, exceptions=(Exception,)):
    """解析ステップをラップし、例外時に警告を出して None を返すデコレータ。

    Parameters
    ----------
    label : str
        警告メッセージに使うステップ名（例: 'HRV解析'）。
        失敗時のメッセージは `警告: {label}に失敗しました ({exc})` に統一される。
    show_traceback : bool, default=False
        True の場合、警告メッセージに加えて traceback.print_exc() を出力する。
    exceptions : tuple[type[BaseException], ...], default=(Exception,)
        捕捉する例外クラス。元コードの捕捉範囲（例: fNIRSブロックの KeyError のみ）
        を変えないために指定する。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as exc:
                print(f'警告: {label}に失敗しました ({exc})')
                if show_traceback:
                    traceback.print_exc()
                return None
        return wrapper
    return decorator
