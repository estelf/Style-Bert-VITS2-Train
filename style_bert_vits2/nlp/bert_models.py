"""
Style-Bert-VITS2 の学習・推論に必要な日本語 BERT モデルをロード/取得するためのモジュール。

オリジナルの Bert-VITS2 では BERT モデルが初回インポート時にハードコードされたパスから「暗黙的に」ロードされているが、
場合によっては多重にロードされて非効率なほか、BERT モデルのロード元のパスがハードコードされているためライブラリ化ができない。

そこで、ライブラリの利用前に、音声合成に利用する BERT モデルだけを「明示的に」ロードできるようにした。
一度 load_model/tokenizer() でモデルがロードされていれば、ライブラリ内部のどこからでもロード済みのモデル/トークナイザーを取得できる。
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Optional

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATH
from style_bert_vits2.logging import logger


if TYPE_CHECKING:
    import torch


# ロード済みの日本語 BERT モデル
__loaded_model: Optional[PreTrainedModel] = None

# ロード済みの日本語 BERT トークナイザー
__loaded_tokenizer: Optional[
    "PreTrainedTokenizer | PreTrainedTokenizerFast"
] = None


def load_model(
    pretrained_model_name_or_path: Optional[str] = None,
    device_map: Optional[
        "str | dict[str, int | torch.device] | int | torch.device"
    ] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
) -> "PreTrainedModel":
    """
    日本語 BERT モデルをロードし、ロード済みの BERT モデルを返す。
    一度ロードされていれば、ロード済みの BERT モデルを即座に返す。
    ライブラリ利用時は常に必ず pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある。
    ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき。
    device_map は既にモデルがロードされている場合は効果がない。
    cache_dir と revision は pretrain_model_name_or_path がリポジトリ名の場合のみ有効。

    Args:
        pretrained_model_name_or_path (Optional[str]): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)
        device_map (Optional[str]): accelerate を使用して高速にデバイスにロードするためのデバイスマップ。
            指定しない場合は通常のモデルロード処理になる (デフォルト: None)
            ref: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
        cache_dir (Optional[str]): モデルのキャッシュディレクトリ。指定しない場合はデフォルトのキャッシュディレクトリが利用される (デフォルト: None)
        revision (str): モデルの Hugging Face 上の Git リビジョン。指定しない場合は最新の main ブランチの内容が利用される (デフォルト: None)

    Returns:
        PreTrainedModel: ロード済みの BERT モデル
    """

    global __loaded_model

    # すでにロード済みの場合はそのまま返す
    if __loaded_model is not None:
        return __loaded_model

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        assert DEFAULT_BERT_MODEL_PATH.exists(), (
            "The default JP BERT model does not exist on the file system. Please specify the path to the pre-trained model."
        )
        pretrained_model_name_or_path = str(DEFAULT_BERT_MODEL_PATH)

    # BERT モデルをロードし、格納して返す
    import torch

    start_time = time.time()
    # dtype=float32 を明示的に指定する（transformers>=5 はデフォルトが "auto" で、
    # チェックポイントの config.json にある torch_dtype: float16 を尊重してしまうため）
    __loaded_model = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path,
        device_map=device_map,
        cache_dir=cache_dir,
        revision=revision,
        dtype=torch.float32,
    )
    logger.info(
        f"Loaded the JP BERT model from {pretrained_model_name_or_path} ({time.time() - start_time:.2f}s)"
    )

    return __loaded_model


def load_tokenizer(
    pretrained_model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
) -> "PreTrainedTokenizer | PreTrainedTokenizerFast":
    """
    日本語 BERT トークナイザーをロードし、ロード済みの BERT トークナイザーを返す。
    一度ロードされていれば、ロード済みの BERT トークナイザーを即座に返す。
    ライブラリ利用時は常に必ず pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある。
    cache_dir と revision は pretrain_model_name_or_path がリポジトリ名の場合のみ有効。

    Args:
        pretrained_model_name_or_path (Optional[str]): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)
        cache_dir (Optional[str]): モデルのキャッシュディレクトリ。指定しない場合はデフォルトのキャッシュディレクトリが利用される (デフォルト: None)
        revision (str): モデルの Hugging Face 上の Git リビジョン。指定しない場合は最新の main ブランチの内容が利用される (デフォルト: None)

    Returns:
        PreTrainedTokenizer | PreTrainedTokenizerFast: ロード済みの BERT トークナイザー
    """

    global __loaded_tokenizer

    # すでにロード済みの場合はそのまま返す
    if __loaded_tokenizer is not None:
        return __loaded_tokenizer

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        assert DEFAULT_BERT_MODEL_PATH.exists(), (
            "The default JP BERT tokenizer does not exist on the file system. Please specify the path to the pre-trained model."
        )
        pretrained_model_name_or_path = str(DEFAULT_BERT_MODEL_PATH)

    # BERT トークナイザーをロードし、格納して返す
    __loaded_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
        use_fast=True,  # デフォルトで True だが念のため明示的に指定
    )
    logger.info(
        f"Loaded the JP BERT tokenizer from {pretrained_model_name_or_path}"
    )

    return __loaded_tokenizer


def transfer_model(device: str) -> None:
    """
    日本語 BERT モデルを、指定されたデバイスに移動する。
    モデルのロード後に推論デバイスを変更したい場合に利用する。
    既に指定されたデバイスにモデルがロードされている場合は何も行われない。

    Args:
        device (str): モデルを移動するデバイス
    """

    if __loaded_model is None:
        raise ValueError("BERT model is not loaded.")

    # 既に指定されたデバイスにモデルがロードされている場合は何もしない
    # ex: current_device="cuda:0", device="cuda" → 何もしない
    # ex: current_device="cuda:0", device="cpu" → モデルを CPU に移動
    current_device = str(__loaded_model.device)
    if current_device.startswith(device):
        return

    __loaded_model.to(device)  # type: ignore
    logger.info(
        f"Transferred the JP BERT model from {current_device} to {device}"
    )


def is_model_loaded() -> bool:
    """
    日本語 BERT モデルがロード済みかどうかを返す。
    """

    return __loaded_model is not None


def is_tokenizer_loaded() -> bool:
    """
    日本語 BERT トークナイザーがロード済みかどうかを返す。
    """

    return __loaded_tokenizer is not None


def unload_model() -> None:
    """
    日本語 BERT モデルをアンロードする。
    """

    global __loaded_model

    if __loaded_model is not None:
        import torch

        del __loaded_model
        __loaded_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Unloaded the JP BERT model")


def unload_tokenizer() -> None:
    """
    日本語 BERT トークナイザーをアンロードする。
    """

    global __loaded_tokenizer

    if __loaded_tokenizer is not None:
        del __loaded_tokenizer
        __loaded_tokenizer = None
        gc.collect()
        logger.info("Unloaded the JP BERT tokenizer")
