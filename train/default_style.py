"""学習時スタイル生成（学習の後工程）。

既定は Neutral 1本のみ（実際に学習へ投入された発話＝train.list / val.list の対象のトリム平均）。
Neutral だけでも十分高い精度が出るため、サブディレクトリごとのスタイル生成は `--styles_by_dirs` を指定したときのみのオプション。
"""

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from style_bert_vits2.constants import DEFAULT_STYLE
from style_bert_vits2.logging import logger


def load_utterance_vectors(
    list_paths: Sequence[Path | str], wav_dir: Path | str
) -> dict[str, list[np.ndarray]]:
    """train.list / val.list の行から、実際に学習へ投入された発話のスタイル特徴を読み込む。

    wavs/ 直下のサブディレクトリごとにグループ化した dict を返す（サブディレクトリなしの平坦な発話は DEFAULT_STYLE グループ）。
    """
    wav_dir = Path(wav_dir).resolve()
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for list_path in list_paths:
        with open(list_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                audiopath = Path(line.split("|")[0].strip())
                # 音声パスにそのまま ".npy" を足した名前（例: <utt>.flac.npy。data_utils と同じ規約）
                npy_path = (audiopath.parent / (audiopath.name + ".npy")).resolve()
                try:
                    vec = np.load(npy_path)
                except FileNotFoundError:
                    raise ValueError(
                        f"スタイル特徴 {npy_path} が見つかりません（{list_path} の行: {audiopath}）"
                    ) from None
                rel = npy_path.relative_to(wav_dir)
                # サブディレクトリ名をスタイル名として扱う（平坦なら DEFAULT_STYLE グループ）
                key = rel.parts[0] if len(rel.parts) > 1 else DEFAULT_STYLE
                groups[key].append(vec)
    return groups


def trimmed_mean(
    vectors: Sequence[np.ndarray], threshold: float = 3.0, max_iter: int = 5
) -> np.ndarray:
    """MAD（中央値絶対偏差）基準の反復トリム平均で代表ベクトルを算出する。

    1. 現在の中央ベクトルからの距離を各発話について計算
    2. 距離が中央値から threshold * MAD 以上離れている外れ値を除外
    3. 外れ値がいなくなる（か残り数が少数）まで反復し、最後に算術平均
    """
    x = np.stack(vectors)
    for _ in range(max_iter):
        median = np.median(x, axis=0)
        dist = np.linalg.norm(x - median, axis=1)
        dist_median = np.median(dist)
        # 1.4826 は正規分布で標準偏差と MAD が一致するための補正係数
        mad = np.median(np.abs(dist - dist_median)) * 1.4826
        if mad <= 0:
            break
        keep = np.abs(dist - dist_median) <= threshold * mad
        if keep.all() or x[keep].shape[0] < 3:
            break  # 外れ値なし、またはこれ以上削れない
        logger.info(
            f"Trimming {int((~keep).sum())} outlier style vector(s) of "
            f"{x.shape[0]} for style representation"
        )
        x = x[keep]
    return x.mean(axis=0)


def save_neutral_vector(
    list_paths: Sequence[Path | str],
    wav_dir: Path | str,
    output_dir: Path | str,
    config_path: Path | str,
    config_output_path: Path | str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = load_utterance_vectors(list_paths, wav_dir)
    embs = [vec for vecs in groups.values() for vec in vecs]
    mean = trimmed_mean(embs)  # (256,)
    only_mean = np.stack([mean])  # (1, 256)
    np.save(output_dir / "style_vectors.npy", only_mean)
    logger.info(
        f"Saved trimmed-mean style vector ({len(embs)} utterances) to {output_dir}"
    )

    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = 1
    json_dict["data"]["style2id"] = {DEFAULT_STYLE: 0}
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved style config to {config_output_path}")


def save_styles_by_dirs(
    list_paths: Sequence[Path | str],
    wav_dir: Path | str,
    output_dir: Path | str,
    config_path: Path | str,
    config_output_path: Path | str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_output_path = Path(config_output_path)

    groups = load_utterance_vectors(list_paths, wav_dir)
    style_dirs = sorted(k for k in groups if k != DEFAULT_STYLE)
    if len(style_dirs) <= 1:
        # サブディレクトリが0〜1個ではスタイル分けの意味がないため Neutral 1本で作り直す
        save_neutral_vector(
            list_paths, wav_dir, output_dir, config_path, config_output_path
        )
        return

    # まず全発話のトリム平均で Neutral を作る（元の順序を保持して全グループ対象）
    style_vectors = [trimmed_mean([vec for vecs in groups.values() for vec in vecs])]
    names = [DEFAULT_STYLE]
    for name in style_dirs:
        style_vectors.append(trimmed_mean(groups[name]))
        names.append(name)

    # Stack them to make (num_styles, 256)
    style_vectors_npy = np.stack(style_vectors, axis=0)
    np.save(output_dir / "style_vectors.npy", style_vectors_npy)
    logger.info(f"Saved style vectors to {output_dir / 'style_vectors.npy'}")

    # Save style2id config to json
    style2id = {name: i for i, name in enumerate(names)}
    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(names)
    json_dict["data"]["style2id"] = style2id
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved style config to {config_output_path}")


def sync_inference_config(
    config_path: Path | str,
    config_output_path: Path | str,
) -> bool:
    """再開時にスキップされるスタイル生成の代わりに、推論資産の config.json へ話者情報を反映する。

    前処理（preprocess/text.py）は Data/<モデル名>/config.json の spk2id / n_speakers を毎回更新するが、
    推論資産 model_assets/<モデル名>/config.json はスタイル生成をスキップすると更新されず、
    話者構成を変えて前処理し直すと推論が古い話者マップで動いてしまう。
    style2id / num_styles は既存のスタイルベクトルと対応させるため Data 側で上書きしない。

    Returns:
        bool: 更新があった場合 True
    """
    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    with open(config_output_path, encoding="utf-8") as f:
        out_dict = json.load(f)

    updated = False
    for key in ("spk2id", "n_speakers"):
        new_value = json_dict["data"].get(key)
        if key in json_dict["data"] and out_dict["data"].get(key) != new_value:
            out_dict["data"][key] = new_value
            updated = True

    if updated:
        with open(config_output_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Updated speaker info (spk2id / n_speakers) in {config_output_path} to match the preprocessed dataset."
        )
    return updated


def save_style_vectors(
    list_paths: Sequence[Path | str],
    wav_dir: Path | str,
    output_dir: Path | str,
    config_path: Path | str,
    config_output_path: Path | str,
    styles_by_dirs: bool = False,
):
    """スタイルベクトルを生成して config.json に style2id を書き込む。

    算出対象は train.list / val.list に載った発話（＝実際に学習へ投入されたもの）のみ。
    デフォルト（styles_by_dirs=False）はサブディレクトリ有無に関係なく Neutral 1本のみ。
    サブディレクトリごとのマルチスタイル生成はオプション（styles_by_dirs=True のときだけ）。
    """
    if styles_by_dirs:
        save_styles_by_dirs(
            list_paths, wav_dir, output_dir, config_path, config_output_path
        )
    else:
        save_neutral_vector(
            list_paths, wav_dir, output_dir, config_path, config_output_path
        )
