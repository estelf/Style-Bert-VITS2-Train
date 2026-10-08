"""学習前処理チェーン（初期化 → raw展開 → データセット検証 → テキスト → BERT特徴 → スタイル特徴）をまとめたパッケージ。

トップレベルの呼び出し口は `preprocess_all.py`（一括実行CLI）で、本パッケージの
`preprocess_all()` を呼ぶ。各段階はモデル名だけで一意に決まる冪等設計（順序や状態ファイルに依存しない）。
BERT特徴・スタイル特徴の生成は重いので別プロセス（`python -m preprocess.bert_gen` 等）として実行する。
環境構築（BERT・SLM・JP-Extra事前学習モデルの取得）は手順ではなく下準備なので、`preprocess_all()` が
必要な分だけ自動でダウンロードする（手動で実行したい場合のみ `uv run -m train.initialize` を使う）。
"""

import json

from preprocess.check_dataset import check_dataset as check_wavs
from preprocess.initialize import extract_raw, initialize
from style_bert_vits2.constants import DATASET_ROOT
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.subprocess import run_script_with_log

__all__ = [
    "initialize",
    "extract_raw",
    "check_dataset",
    "preprocess_text",
    "bert_gen",
    "style_gen",
    "preprocess_all",
]


def check_dataset(model_name: str) -> None:
    """Step 3: 学習データが正しい形（ターゲットのサンプリングレート・モノラル）か特徴量生成前に検証する。

    Raises:
        ValueError: 不正な形の音声ファイルが見つかった場合
    """
    logger.info("Step 3: start checking dataset...")
    dataset_path = DATASET_ROOT / model_name
    with open(dataset_path / "config.json", encoding="utf-8") as f:
        sampling_rate = json.load(f)["data"]["sampling_rate"]
    check_wavs(wavs_dir=dataset_path / "wavs", sampling_rate=sampling_rate)
    logger.success("Step 3: dataset check finished.")


def preprocess_text(model_name: str, val_per_lang: int, yomi_error: str) -> None:
    """Step 4: 書き起こしファイル（esd.list）から train.list / val.list を作る。

    Raises:
        RuntimeError: 書き起こしファイルの前処理に失敗した場合
    """
    logger.info("Step 4: start preprocessing text...")
    ok, message = run_script_with_log(
        [
            "-m",
            "preprocess.text",
            "--model_name",
            model_name,
            "--val-per-lang",
            str(val_per_lang),
            "--yomi_error",
            yomi_error,
            "--correct_path",  # 音声ファイルのパスを正しいパスに修正する
        ]
    )
    if not ok:
        raise RuntimeError(f"書き起こしファイルの前処理に失敗しました:\n{message}")


def bert_gen(model_name: str) -> None:
    """Step 5: BERT特徴ファイル（*.bert.pt）を生成する（重いのでプロセス数いじり不可）。

    Raises:
        RuntimeError: BERT特徴ファイルの生成に失敗した場合
    """
    logger.info("Step 5: start bert_gen...")
    ok, message = run_script_with_log(
        ["-m", "preprocess.bert_gen", "--model_name", model_name], ignore_warning=True
    )
    if not ok:
        raise RuntimeError(f"BERT特徴ファイルの生成に失敗しました:\n{message}")


def style_gen(model_name: str, num_processes: int) -> None:
    """Step 6: スタイル特徴を生成し、style2id を config.json に書き込む（学習の後工程）。

    Raises:
        RuntimeError: スタイル特徴ファイルの生成に失敗した場合
    """
    logger.info("Step 6: start style_gen...")
    ok, message = run_script_with_log(
        [
            "-m",
            "preprocess.style_gen",
            "--model_name",
            model_name,
            "--num_processes",
            str(num_processes),
        ],
        ignore_warning=True,
    )
    if not ok:
        raise RuntimeError(f"スタイル特徴ファイルの生成に失敗しました:\n{message}")


def preprocess_all(
    model_name: str,
    batch_size: int = 4,
    epochs: int = 100,
    save_every_steps: int = 1000,
    num_processes: int = 2,
    freeze_JP_bert: bool = True,
    freeze_style: bool = False,
    freeze_decoder: bool = False,
    val_per_lang: int = 4,  # 話者ごとの検証データ数（上流既定の4。0 で無効化）
    log_interval: int = 200,
    yomi_error: str = "raise",
    dtype: str = "float32",
) -> None:
    """前処理チェーン全体（Step 1〜6）を順番に実行する。

    Args:
        model_name (str): データセット名（Data/<モデル名> を使う・作る）
        dtype (str): 学習時の計算精度 ("bfloat16" / "float32")。config.json に書き込まれる
        yomi_error (str): 読み上げエラー時の挙動。Options: raise, skip, use

    Raises:
        ValueError: モデル名が空の場合
        RuntimeError: いずれかの段階が失敗した場合
    """
    if model_name == "":
        raise ValueError("モデル名は空にできません")
    # 環境構築（モデル取得）は前処理の下準備として自動で行う（既存ファイルがあればスキップされる冪等な処理）
    from train.initialize import (
        download_bert_models,
        download_jp_extra_pretrained_models,
        download_slm_model,
    )

    download_bert_models()
    download_slm_model()
    download_jp_extra_pretrained_models()
    initialize(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        save_every_steps=save_every_steps,
        freeze_JP_bert=freeze_JP_bert,
        freeze_style=freeze_style,
        freeze_decoder=freeze_decoder,
        log_interval=log_interval,
        dtype=dtype,
    )
    extract_raw(model_name=model_name)
    check_dataset(model_name=model_name)
    preprocess_text(
        model_name=model_name, val_per_lang=val_per_lang, yomi_error=yomi_error
    )
    bert_gen(model_name=model_name)
    style_gen(model_name=model_name, num_processes=num_processes)
    logger.success("Success: All preprocess finished!")
