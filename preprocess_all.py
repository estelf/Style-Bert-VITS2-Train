"""前処理の一括実行CLI。データセットの初期化〜特徴量生成（Step 1〜6）をモデル名を明示して実行する。

学習自体は別のコマンド（`train_model.py -m <モデル名>`）で行う。各段階は冪等で、順序や他のコマンドの実行履歴に依存しない。

学習設定の多くは config.json のキーなので、個別の CLI フラグではなく汎用の `--set dotted.path=value`
で上書きする（設定の単一ソースは style_bert_vits2/models/hyper_parameters.py の HyperParameters）。
`--set` は指定したキーだけ、既存 config.json の値より後の最優先で反映される（未指定キーは既存値を維持
するので再実行で手編集が失われない）。値は JSON として解釈する（4→int, true→bool, 0.0001→float,
[0.8,0.99]→list, null→None、それ以外は文字列）。パス・値は HyperParameters で検証される。

例:
    uv run preprocess_all.py -m mymodel \\
        --set train.batch_size=4 \\
        --set train.epochs=1000 \\
        --set train.eval_interval=1000 \\
        --set train.freeze_style=true

epochs は打ち切り上限。早期停止が既定有効のため収束すればこの前に自動停止する。
"""

import argparse
from multiprocessing import cpu_count

from preprocess import preprocess_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", "-m", type=str, help="Model name", required=True
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        dest="set_overrides",
        help="config.json の設定を dotted.path で上書き（複数指定可）。"
        "例: --set train.batch_size=4 --set train.freeze_style=true。"
        "指定したキーだけ既存 config.json より後に反映され、未指定キーは既存値を維持する。"
        "train.epochs / train.eval_interval / train.log_interval / train.freeze_* などもこれで行う"
        "（epochs は打ち切り上限。早期停止既定有効のため収束すればその前で止まる）",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        help="Number of processes",
        # 1コア環境だと cpu_count()//2 が 0 になり ThreadPoolExecutor が ValueError を出すため最低1を保証
        default=max(1, cpu_count() // 2),
    )
    parser.add_argument(
        "--val_per_lang",
        type=int,
        help="検証発話を話者ごとにこの件数へ固定する（既定0）。言語ごとではない（上流の命名互換）。"
        ">0 にすると val_ratio より優先されるので、件数を明示したいときだけ指定する",
        default=0,
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        help="検証発話の比率（既定0.05=5%%）。話者ごとに round(発話数×ratio) 件を検証へ回す。"
        "0 で検証なし（全件学習。早期停止は機能しなくなるので epochs 上限まで走る）",
        default=0.05,
    )
    parser.add_argument(
        "--min_val",
        type=int,
        help="比率ベースの検証件数の下限（既定20。その話者の発話数の1/5を超えない）。"
        "検証損失の評価ノイズは 1/sqrt(件数) なので、少なすぎると早期停止の判定が機能しない",
        default=20,
    )
    parser.add_argument(
        "--max_val_total",
        type=int,
        help="検証発話の総上限（既定100）。超えた分は話者按比例で縮小して学習へ戻る"
        "（縮小時は Step 4 で警告される）",
        default=100,
    )
    parser.add_argument(
        "--yomi_error",
        type=str,
        help="Yomi error. Options: raise, skip, use",
        default="raise",
    )
    parser.add_argument(
        "--reset_models",
        action="store_true",
        help="既存の models/（学習途中のチェックポイント含む）をバックアップし、事前学習モデルでリセットする（既定は config.json のみ更新）",
        default=False,
    )

    args = parser.parse_args()

    preprocess_all(
        model_name=args.model_name,
        set_overrides=args.set_overrides,
        num_processes=args.num_processes,
        val_per_lang=args.val_per_lang,
        val_ratio=args.val_ratio,
        min_val=args.min_val,
        max_val_total=args.max_val_total,
        yomi_error=args.yomi_error,
        reset_models=args.reset_models,
    )
