"""話者埋め込み（スタイルベクトル抽出）モデルのロードを一元化する。

pyannote/wespeaker-voxceleb-resnet34-LM は前処理 Step6（*.wav.npy の生成）、参照音声からの
スタイル推論（TTSModel.infer の reference_audio_path）、学習後の話者類似度評価
（train/select_best.py --spk-sim）の3箇所が使う。以前は各所で `Model.from_pretrained(repo_id)` を
直接呼んでいたため HF キャッシュに依存し、「pretrained/ が揃っていればオフラインで通る」契約の
外側に出ていた。ここでは必ず `style_bert_vits2.pretrained` 経由で `pretrained/` 配下に取得した
ローカルディレクトリを渡す。

pyannote.audio は scikit-learn 等の重量級ライブラリを巻き込むため、実際にスタイルベクトルが
必要になるまで遅延 import する（`--help` だけで落ちないための従来の方針を維持）。
"""

from __future__ import annotations

from typing import Any

from style_bert_vits2.logging import logger
from style_bert_vits2.pretrained import PRETRAINED_REPOS, ensure_pretrained

__all__ = ["load_speaker_inference", "load_speaker_model"]

_SPEAKER_REPO = PRETRAINED_REPOS["speaker"]


def _import_pyannote() -> Any:
    try:
        import pyannote.audio
    except ImportError as e:
        raise ImportError(
            "pyannote.audio is required to infer style vector from audio "
            "（`uv sync` で train グループも含めてインストールされているか確認してください）"
        ) from e
    return pyannote.audio


def load_speaker_model() -> Any:
    """話者埋め込みモデル（pyannote の Lightning モジュール）を pretrained/ のローカルからロードする"""
    ensure_pretrained(("speaker",))
    pyannote_audio = _import_pyannote()
    # ディレクトリ指定では pyannote が AssetFileName.Model（pytorch_model.bin）を直接読む。
    # 上流の gated 化はないのでトークンも利用同意も不要（未取得なら ensure_pretrained が止まる）
    model = pyannote_audio.Model.from_pretrained(str(_SPEAKER_REPO.local_dir))
    if model is None:
        raise RuntimeError(
            f"話者埋め込みモデルを {_SPEAKER_REPO.local_dir} から読み込めませんでした。"
            "`uv run -m style_bert_vits2.pretrained --only speaker --force` で再取得してください"
        )
    return model


def load_speaker_inference(device: str | None = None) -> Any:
    """音声からスタイルベクトルを抽出する pyannote Inference を返す。

    Args:
        device (str | None): "cuda" / "cpu" 等。None なら CUDA があれば GPU

    Returns:
        Any: pyannote.audio.Inference（呼び出すと音声から話者埋め込みが返る）
    """
    import torch

    pyannote_audio = _import_pyannote()
    model = load_speaker_model()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = pyannote_audio.Inference(model=model, window="whole")
    inference = inference.to(torch.device(device))
    logger.debug(f"Loaded the speaker embedding model on {device}")
    return inference
