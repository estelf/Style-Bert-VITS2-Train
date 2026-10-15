import sys

from loguru import logger

# デフォルトのハンドラを全て除去
logger.remove()

# 新しいハンドラを追加（stdout へ。ファイルログは学習・前処理の実行時に logger.add する）
logger.add(
    sys.stdout,
    format="<g>{time:MM-DD HH:mm:ss}</g> |<lvl>{level:^8}</lvl>| {file}:{line} | {message}",
    backtrace=True,
    diagnose=True,
)
