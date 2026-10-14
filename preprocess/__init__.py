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
    """Step 3: 学習データが正しい形（サンプリングレート・モノラル・学習可能な尺・esd.list との整合）か特徴量生成前に検証する。

    Raises:
        ValueError: 不正な形の音声ファイルや esd.list と wavs/ の不一致が見つかった場合
    """
    logger.info("Step 3: start checking dataset...")
    dataset_path = DATASET_ROOT / model_name
    with open(dataset_path / "config.json", encoding="utf-8") as f:
        data_config = json.load(f)["data"]
    check_wavs(
        wavs_dir=dataset_path / "wavs",
        sampling_rate=data_config["sampling_rate"],
        hop_length=data_config["hop_length"],
        transcription_path=dataset_path / "esd.list",
    )
    logger.success("Step 3: dataset check finished.")


def preprocess_text(
    model_name: str,
    val_per_lang: int,
    yomi_error: str,
    max_val_total: int = 100,
    val_ratio: float = 0.05,
    min_val: int = 20,
) -> None:
    """Step 4: 書き起こしファイル（esd.list）から train.list / val.list を作る。

    検証件数は既定は比率ベース（val_ratio、話者ごとに round(発話数×ratio)。下限 min_val）。
    val_per_lang > 0 のときだけ件数固定に上書きする。

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
            "--val-ratio",
            str(val_ratio),
            "--min-val",
            str(min_val),
            "--max-val-total",
            str(max_val_total),
            "--yomi_error",
            yomi_error,
            "--correct_path",  # 音声ファイルのパスを正しいパスに修正する
        ]
    )
    if not ok:
        raise RuntimeError(f"書き起こしファイルの前処理に失敗しました:\n{message}")


def bert_gen(model_name: str) -> None:
    """Step 5: BERT特徴ファイル（*.bert.pt）を生成する。

    既存キャッシュには生成根拠（テキスト＋word2ph）のハッシュを添えて保存し、有効な行は
    読み飛ばす。再生成が必要な行だけを GPU バッチ forward で処理する（旧実装は全行を
    1件ずつ forward していた）。辞書更新やリテイクで読みが変わった行は自動再生成される
    （学習側 train/data_utils.py でもハッシュを照合し、不一致なら Step 5 再実行を要求して失敗する）。

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
    """Step 6: 各音声のスタイル特徴（*.wav.npy）を生成する。

    style2id / num_styles の config.json への書き込みはここではなく、学習開始時に
    train/default_style.py（Neutral 1本化）が行う。既存 .npy がある音声はスキップされる
    （音声の中身が変わった場合は Step 2 の extract_raw が古い .npy を削除する）

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
    set_overrides: list[str] | None = None,
    num_processes: int = 2,
    val_per_lang: int = 0,  # >0 で検証件数を実数固定（既定0は比率ベース）
    val_ratio: float = 0.05,  # 検証発話の比率（話者ごとに round(発話数×ratio)。0 で検証なし）
    min_val: int = 20,  # 比率ベースの検証件数の下限（その話者の発話数の1/5を超えない）
    max_val_total: int = 100,  # 検証発話の総上限（超過分は話者按比例で縮小して学習へ戻す）
    yomi_error: str = "raise",
    reset_models: bool = False,
) -> None:
    """前処理チェーン全体（Step 1〜6）を順番に実行する。

    Args:
        model_name (str): データセット名（Data/<モデル名> を使う・作る）
        set_overrides (list[str] | None): config.json の設定上書きを "dotted.path=value" で
            指定する（例: ["train.batch_size=4", "train.freeze_style=true"]）。指定したものだけ、
            既存 config.json の値より後の最優先で反映する（未指定キーは既存値・既定値を尊重し、
            再実行で手編集が失われない）。パス・値は HyperParameters スキーマで検証される。
        reset_models (bool): True の場合のみ models/ をバックアップして事前学習モデルでリセットする。既定では既存チェックポイントを上書きしない
        yomi_error (str): 読み上げエラー時の挙動。Options: raise, skip, use
        val_per_lang (int): 検証発話を話者ごとにこの件数へ固定する（既定0で比率ベース。
            >0 のとき val_ratio より優先される）
        val_ratio (float): 検証発話の比率。話者ごとに round(発話数×ratio) 件を検証へ回す
            （0 で検証なし＝全件学習。検証損失の評価ノイズは 1/sqrt(件数) で減るため、
            早期停止を使うなら最低20件は欲しい）
        min_val (int): 比率ベースの検証件数の下限。ただしその話者の発話数の1/5を超えない
            （小規模話者が学習データを検証に食われすぎないための上限も兼ねる）
        max_val_total (int): 検証発話の総上限。超えた分は話者按比例で縮小して学習へ戻す
            （縮小時は Step 4 で警告される）

    Raises:
        ValueError: モデル名が空の場合、または set_overrides の指定が不正な場合
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
        set_overrides=set_overrides,
        reset_models=reset_models,
    )
    extract_raw(model_name=model_name)
    check_dataset(model_name=model_name)
    preprocess_text(
        model_name=model_name,
        val_per_lang=val_per_lang,
        yomi_error=yomi_error,
        max_val_total=max_val_total,
        val_ratio=val_ratio,
        min_val=min_val,
    )
    bert_gen(model_name=model_name)
    style_gen(model_name=model_name, num_processes=num_processes)
    logger.success("Success: All preprocess finished!")
