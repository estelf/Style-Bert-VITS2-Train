"""前処理チェーンの最初に走るデータセット検証。

学習対象の音声ファイルが正しい形（読める・ターゲットのサンプリングレート・モノラル）か、また学習に使える尺の範囲内かを
特徴量生成（BERT特徴・スタイル特徴）の前にチェックし、不正なファイルがあれば早期に失敗させる（尺の範囲外は警告のみ）。
esd.list と wavs/ の突き合わせ（欠損・重複・未参照）もこの段階で行い、1 回の実行で全問題を列挙する。
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from style_bert_vits2.logging import logger

# soundfile (libsndfile) でメタデータを読める拡張子
SUPPORTED_SUFFIXES = {".flac", ".wav", ".ogg", ".mp3"}

# DistributedBucketSampler のバケット境界（train/pipeline.py）と一致させる尺の範囲。
# この外の音声は学習時に黙って破棄されるため、前処理段階で警告として通知する（エラーにはしない。
#   既定値なら 32 フレーム ≒ 0.37秒未満、1000 フレーム ≒ 11.6秒超 で除外対象。
#   変更時は train/pipeline.py の boundaries 側も必ず同じ値に揃えること）
MIN_SPEC_FRAMES = 32
MAX_SPEC_FRAMES = 1000


def check_dataset(
    wavs_dir: Path,
    sampling_rate: int,
    hop_length: int,
    transcription_path: Path | None = None,
) -> None:
    """wavs_dir 以下の音声ファイルを検証する（サンプリングレート・チャンネル数・esd.list との突き合わせはエラー、尺の範囲外は警告のみ）。

    Args:
        wavs_dir (Path): 音声ファイルが展開されたディレクトリ
        sampling_rate (int): config.json の data.sampling_rate にあるターゲットのサンプリングレート
        hop_length (int): config.json の data.hop_length（フレーム数→秒換算に使う）
        transcription_path (Optional[Path]): esd.list のパス。指定時は wavs/ と突き合わせて欠損・重複を列挙する

    Raises:
        ValueError: 検証対象の音声がない、または不正な形の音声ファイル・突き合わせエラーが見つかった場合
    """
    min_seconds = MIN_SPEC_FRAMES * hop_length / sampling_rate
    max_seconds = MAX_SPEC_FRAMES * hop_length / sampling_rate
    problems: list[str] = []
    out_of_range: list[str] = []
    total = 0
    listed_utts: list[str] = []

    # esd.list に書かれた音声パスの集合。Step 4 前なので raw のままの 1 行目フィールドを読む。
    # 1行目は wavs/ からの相対パス（サブディレクトリを含む場合がある）なので、
    # ベース名ではなく相対パスのまま正規化して照合する（別ディレクトリの同名ファイルを誤検出しない）
    if transcription_path is not None and transcription_path.is_file():
        for line in transcription_path.read_text(encoding="utf-8").splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 1 and fields[0].strip():
                listed_utts.append(Path(fields[0].strip()).as_posix())

    listed_set = set(listed_utts)
    found_audio: set[str] = set()

    # extract_raw は zip のディレクトリ構造をそのまま展開するため、サブディレクトリ込みで再帰探索する
    # （非再帰 glob だとフォルダ分けデータセットで検出 0 件になり Step 3 が誤って失敗する）
    for path in sorted(wavs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel = path.relative_to(wavs_dir).as_posix()
        total += 1
        found_audio.add(rel)
        try:
            with sf.SoundFile(path) as f:
                file_sr, channels, frames = f.samplerate, f.channels, f.frames
        except Exception as e:
            problems.append(f"{rel}: 音声ファイルとして読み込めません ({e})")
            continue
        if file_sr != sampling_rate:
            problems.append(
                f"{rel}: サンプリングレートが一致しません ({file_sr} != {sampling_rate})"
            )
        if channels != 1:
            problems.append(f"{rel}: モノラルではありません ({channels} channels)")
        # 尺の範囲検証（ DistributedBucketSampler のバケット境界外の音声は学習時に黙って破棄されるため、
        #   前処理段階で参考情報として警告だけ出す。エラーにはしない ）
        duration = frames / file_sr
        if not min_seconds <= duration <= max_seconds:
            out_of_range.append(f"{rel} ({duration:.2f}s)")

    if out_of_range:
        logger.warning(
            f"長さがバケット範囲 [{min_seconds:.2f}s, {max_seconds:.2f}s] 外の音声が {len(out_of_range)} 件あります（学習時に除外されます）: {out_of_range[:10]}"
        )

    if total == 0:
        raise ValueError(f"{wavs_dir} に検証対象の音声ファイルがありません。")

    # esd.list ↔ wavs/ の突き合わせ（欠損・重複は Step 4 で黙って捨てられるためここで落とす）
    if listed_utts:
        missing = sorted({u for u in listed_utts if u not in found_audio})
        problems.extend(f"esd.list にあるが音声が見つかりません: {u}" for u in missing)
        seen: set[str] = set()
        dup = sorted({u for u in listed_utts if u in seen or seen.add(u)})
        problems.extend(
            f"esd.list で同じ音声ファイルが重複しています: {u}" for u in dup
        )
        unlisted = sorted(found_audio - listed_set)
        if unlisted:
            logger.warning(
                f"wavs/ にあるのに esd.list から参照されていない音声ファイルが {len(unlisted)} 件あります（学習に使われません）: {unlisted[:10]}"
            )

    if problems:
        detail = "\n".join(problems[:10])
        logger.error(
            f"Dataset check failed: {len(problems)}/{total} files are invalid.\n{detail}"
        )
        raise ValueError(
            f"{len(problems)}/{total} 件のデータセットの問題が見つかりました:\n{detail}"
        )

    logger.info(
        f"Dataset check passed: {total} audio files are {sampling_rate} Hz mono and within the usable length range."
    )
