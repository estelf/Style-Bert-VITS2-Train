"""前処理チェーンの Step 1〜2（データセット初期化・raw.zip 展開）。

トップレベルの呼び出し口である `preprocess_all.py`（経由の本パッケージの preprocess_all）から
呼ばれるほか、単体でも `uv run -m preprocess.initialize --model_name <モデル名>` で実行できる。
"""

import json
import shutil
import zipfile
from datetime import datetime

from style_bert_vits2.constants import DATASET_ROOT
from style_bert_vits2.logging import logger

_logger_handler = None


def initialize(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    freeze_JP_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    log_interval: int,
    dtype: str = "float32",
) -> None:
    """Step 1: Data/<モデル名>/config.json をテンプレートから生成し、事前学習済みモデルを配置する。

    Args:
        dtype (str): 学習時の計算精度 ("float32" / "bfloat16")。config.json の train.dtype に書き込まれる。fp16 は学習に使えないため選択不可。

    Raises:
        FileNotFoundError: pretrained/jp_extra が存在しない場合など
    """
    global _logger_handler
    dataset_path = DATASET_ROOT / model_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    # 前処理のログをファイルに保存する
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if _logger_handler is not None:
        logger.remove(_logger_handler)
    _logger_handler = logger.add(dataset_path / f"preprocess_{timestamp}.log")

    logger.info(
        f"Step 1: start initialization...\nmodel_name: {model_name}, batch_size: {batch_size}, epochs: {epochs}, save_every_steps: {save_every_steps}, freeze_JP_bert: {freeze_JP_bert}, freeze_style: {freeze_style}, freeze_decoder: {freeze_decoder}, log_interval: {log_interval}, dtype: {dtype}"
    )

    with open("configs/config_jp_extra.json", encoding="utf-8") as f:
        config = json.load(f)
    config["model_name"] = model_name
    config["data"]["training_files"] = str(dataset_path / "train.list")
    config["data"]["validation_files"] = str(dataset_path / "val.list")
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs
    config["train"]["eval_interval"] = save_every_steps
    config["train"]["log_interval"] = log_interval

    config["train"]["freeze_JP_bert"] = freeze_JP_bert
    config["train"]["freeze_style"] = freeze_style
    config["train"]["freeze_decoder"] = freeze_decoder

    config["train"]["dtype"] = dtype  # 学習時の計算精度（float32 / bfloat16。fp16は学習に使えない）

    model_path = dataset_path / "models"
    if model_path.exists():
        logger.warning(
            f"Step 1: {model_path} already exists, so copy it to backup to {model_path}_backup"
        )
        shutil.copytree(src=model_path, dst=dataset_path / "models_backup", dirs_exist_ok=True)
        shutil.rmtree(model_path)
    shutil.copytree(src="pretrained/jp_extra", dst=model_path)

    with open(dataset_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.success("Step 1: initialization finished.")


def extract_raw(model_name: str) -> None:
    """Step 2: 完成済みデータセットの raw.zip を解凍して wavs/ を作る（スライス・正規化等は分離先のデータセット作成リポジトリ側の責任）。

    Raises:
        FileNotFoundError: raw.zip がない場合
    """
    logger.info("Step 2: start extracting raw.zip...")
    dataset_path = DATASET_ROOT / model_name
    raw_zip = dataset_path / "raw.zip"
    if not raw_zip.exists():
        raise FileNotFoundError(f"{raw_zip} が見つかりません。")
    out_dir = dataset_path / "wavs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip) as zf:
        zf.extractall(out_dir)
    logger.success("Step 2: extracting raw.zip finished.")
