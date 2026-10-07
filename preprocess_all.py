"""前処理の一括実行CLI。データセットの初期化〜特徴量生成（Step 1〜6）をモデル名を明示して実行する。

学習自体は別のコマンド（`train_model.py -m <モデル名>`）で行う。各段階は冪等で、順序や他のコマンドの実行履歴に依存しない。
"""

import argparse
from multiprocessing import cpu_count

from preprocess import preprocess_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", "-m", type=str, help="Model name", required=True
    )
    parser.add_argument("--batch_size", "-b", type=int, help="Batch size", default=4)
    parser.add_argument("--epochs", "-e", type=int, help="Epochs", default=100)
    parser.add_argument(
        "--save_every_steps",
        "-s",
        type=int,
        help="Save every steps",
        default=1000,
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        help="Number of processes",
        default=cpu_count() // 2,
    )
    parser.add_argument(
        "--freeze_JP_bert", action="store_true", help="Freeze JP BERT", default=True
    )
    parser.add_argument(
        "--freeze_style", action="store_true", help="Freeze style vector", default=False
    )
    parser.add_argument(
        "--freeze_decoder", action="store_true", help="Freeze decoder", default=False
    )
    parser.add_argument(
        "--val_per_lang",
        type=int,
        help="Validation per speaker",
        default=0,
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        help="Log interval",
        default=200,
    )
    parser.add_argument(
        "--yomi_error",
        type=str,
        help="Yomi error. Options: raise, skip, use",
        default="raise",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16"],
        help="学習時の計算精度（config.json の train.dtype に書き込まれる。既定 float32。fp16 は学習に使えないため選択不可）",
        default="float32",
    )

    args = parser.parse_args()

    preprocess_all(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_every_steps=args.save_every_steps,
        num_processes=args.num_processes,
        freeze_JP_bert=args.freeze_JP_bert,
        freeze_style=args.freeze_style,
        freeze_decoder=args.freeze_decoder,
        val_per_lang=args.val_per_lang,
        log_interval=args.log_interval,
        yomi_error=args.yomi_error,
        dtype=args.dtype,
    )
