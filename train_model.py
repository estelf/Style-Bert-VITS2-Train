"""学習のトップレベル呼び出し口。前処理済みのデータセットで単体学習・再開したい場合に使う。

前処理込みの一括実行は `preprocess_all.py` を使うこと（内部でこの train/pipeline と preprocess/ を呼ぶ）。
"""

from train.pipeline import run


if __name__ == "__main__":
    run()
