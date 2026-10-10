import glob
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from style_bert_vits2.logging import logger
from style_bert_vits2.models.utils import (
    checkpoints,  # type: ignore # noqa: F401
    safetensors,  # type: ignore # noqa: F401
)

if TYPE_CHECKING:
    # tensorboard はライブラリとしてインストールされている場合は依存関係に含まれないため、型チェック時のみインポートする
    from torch.utils.tensorboard import SummaryWriter


def summarize(
    writer: "SummaryWriter",
    global_step: int,
    scalars: dict[str, float] = {},
    histograms: dict[str, Any] = {},
    images: dict[str, Any] = {},
    audios: dict[str, Any] = {},
    audio_sampling_rate: int = 22050,
) -> None:
    """
    指定されたデータを TensorBoard にまとめて追加する

    Args:
        writer (SummaryWriter): TensorBoard への書き込みを行うオブジェクト
        global_step (int): グローバルステップ数
        scalars (dict[str, float]): スカラー値の辞書
        histograms (dict[str, Any]): ヒストグラムの辞書
        images (dict[str, Any]): 画像データの辞書
        audios (dict[str, Any]): 音声データの辞書
        audio_sampling_rate (int): 音声データのサンプリングレート
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def is_resuming(dir_path: str | Path) -> bool:
    """
    指定されたディレクトリパスに再開可能なモデルが存在するかどうかを返す

    Args:
        dir_path: チェックするディレクトリのパス

    Returns:
        bool: 再開可能なモデルが存在するかどうか
    """
    # JP-ExtraバージョンではDURがなくWDがあったり変わるため、Gのみで判断する
    g_list = glob.glob(os.path.join(dir_path, "G_*.pth"))
    # d_list = glob.glob(os.path.join(dir_path, "D_*.pth"))
    # dur_list = glob.glob(os.path.join(dir_path, "DUR_*.pth"))
    return len(g_list) > 0


def load_wav_to_torch(full_path: str | Path) -> tuple[torch.FloatTensor, int]:
    """
    指定された音声ファイルを読み込み、PyTorch のテンソルに変換して返す

    Args:
        full_path (Union[str, Path]): 音声ファイルのパス

    Returns:
        tuple[torch.FloatTensor, int]: 音声データのテンソルとサンプリングレート
            テンソルは soundfile により ±1 に正規化済み（scipy.io.wavfile のように生サンプルではないことに注意）
    """

    # この関数は学習時以外使われないため、ライブラリとしての style_bert_vits2 が
    # 重量級ライブラリに依存しないように遅延 import する
    # scipy.io.wavfile は RIFF/WAV しか読めないが、soundfile は flac などにも対応している
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required to load audio file")

    data, sampling_rate = sf.read(full_path, dtype="float32")
    return torch.from_numpy(np.ascontiguousarray(data)), sampling_rate


def load_filepaths_and_text(filename: str | Path, split: str = "|") -> list[list[str]]:
    """
    指定されたファイルからファイルパスとテキストを読み込む

    Args:
        filename (Union[str, Path]): ファイルのパス
        split (str): ファイルの区切り文字 (デフォルト: "|")

    Returns:
        list[list[str]]: ファイルパスとテキストのリスト
    """

    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_steps(model_path: str | Path) -> int | None:
    """
    モデルのパスからイテレーション回数を取得する

    Args:
        model_path (Union[str, Path]): モデルのパス

    Returns:
        Optional[int]: イテレーション回数
    """

    matches = re.findall(r"\d+", model_path)  # type: ignore
    return int(matches[-1]) if matches else None


def check_git_hash(model_dir_path: str | Path) -> None:
    """
    リポジトリの git ハッシュ値をモデルディレクトリの githash と比較する（git 管理外なら黙ってスキップ。毎回警告出していても気づけない）

    Args:
        model_dir_path (Union[str, Path]): モデルのディレクトリのパス
    """

    # git はカレントディレクトリから上位を辿ってリポジトリを探すため、パッケージ内の .git を探す必要はない
    cur_hash = subprocess.getoutput("git rev-parse HEAD")
    if "fatal:" in cur_hash or not cur_hash.strip():
        logger.debug(
            "The current directory is not a git repository, so hash value comparison will be ignored."
        )
        return

    path = os.path.join(model_dir_path, "githash")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved_hash = f.read()
        if saved_hash != cur_hash:
            logger.warning(
                f"git hash values are different. {saved_hash[:8]}(saved) != {cur_hash[:8]}(current)"
            )
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(cur_hash)
