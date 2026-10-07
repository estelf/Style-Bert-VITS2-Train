"""前処理チェーンの最初に走るデータセット検証。

学習対象の音声ファイルが正しい形（読める・ターゲットのサンプリングレート・モノラル）かを
特徴量生成（BERT特徴・スタイル特徴）の前にチェックして、不正なファイルがあれば早期に失敗させる。
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from style_bert_vits2.logging import logger


# soundfile (libsndfile) でメタデータを読める拡張子
SUPPORTED_SUFFIXES = {".flac", ".wav", ".ogg", ".mp3"}


def check_dataset(wavs_dir: Path, sampling_rate: int) -> None:
    """wavs_dir 以下の音声ファイルを検証する（サンプリングレート・チャンネル数）。

    Args:
        wavs_dir (Path): 音声ファイルが展開されたディレクトリ
        sampling_rate (int): config.json の data.sampling_rate にあるターゲットのサンプリングレート

    Raises:
        ValueError: 検証対象の音声がない、または不正な形の音声ファイルが見つかった場合
    """
    problems: list[str] = []
    total = 0
    for path in sorted(wavs_dir.glob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        total += 1
        try:
            with sf.SoundFile(path) as f:
                file_sr, channels = f.samplerate, f.channels
        except Exception as e:
            problems.append(f"{path.name}: 音声ファイルとして読み込めません ({e})")
            continue
        if file_sr != sampling_rate:
            problems.append(
                f"{path.name}: サンプリングレートが一致しません ({file_sr} != {sampling_rate})"
            )
        if channels != 1:
            problems.append(f"{path.name}: モノラルではありません ({channels} channels)")

    if total == 0:
        raise ValueError(f"{wavs_dir} に検証対象の音声ファイルがありません。")

    if problems:
        detail = "\n".join(problems[:10])
        logger.error(
            f"Dataset check failed: {len(problems)}/{total} files are invalid.\n{detail}"
        )
        raise ValueError(
            f"{len(problems)}/{total} 件の音声ファイルが学習データの形式と異なります:\n{detail}"
        )

    logger.info(
        f"Dataset check passed: {total} audio files are {sampling_rate} Hz mono."
    )
