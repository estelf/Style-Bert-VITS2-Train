"""Step 5: BERT特徴ファイル（*.bert.pt）を生成する。

特徴には生成根拠（テキスト＋word2ph）のハッシュを添えて保存し、esd.list のリテイクや
dict_data/ 辞書更新で読みが変わった古い特徴を黙って再利用しない。学習側
（train/data_utils.py）でも同じハッシュを照合し、不一致なら Step 5 再実行を要求して
失敗する（style_bert_vits2.utils.bert_cache 参照）。
再生成が必要な行だけを GPU バッチ forward で処理する（旧実装は全行を1件ずつ forward し、
既存キャッシュがあっても毎回 torch.load するだけで重かった）。
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from style_bert_vits2.constants import DATASET_ROOT
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp import cleaned_text_to_sequence
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature_batch
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.utils.bert_cache import (
    bert_feature_key,
    load_bert_feature,
    save_bert_feature,
)

# extract_bert_feature_batch 内の text_to_sep_kata が pyopenjtalk を使うためワーカーを
# 初期化する（dict_data/ 辞書は 読み→word2ph→特徴 の整合に効くのでここでも適用する。
# 特徴 forward 自体はデコーダモデルのみなので、並列化の障害だったワーカー RPC は
# テキスト正規化の直列部分にしか関与しない）
pyopenjtalk_worker.initialize_worker()
update_dict()

# GPUがあればGPU、無ければCPUで特徴量を抽出する
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_line(line: str, add_blank: bool):
    """train.list / val.list の行から (wav_path, text, 展開用 word2ph, 期待音素長, key) を作る"""
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    # ハッシュは list に書かれた生の word2ph（intersperse 前）から作る。
    # train/data_utils.py の get_text と必ず同じ規約であること
    key = bert_feature_key(text, word2ph)
    phone, tone, language = cleaned_text_to_sequence(phone, tone)

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    return wav_path, text, word2ph, len(phone), key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, required=True, help="モデル名")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="BERT forward のバッチサイズ（再生成が必要な行のみ対象。VRAM に応じて調整）",
    )
    args, _ = parser.parse_known_args()
    config_path = DATASET_ROOT / args.model_name / "config.json"
    hps = HyperParameters.load_from_json(config_path)
    lines: list[str] = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    add_blank = hps.data.add_blank

    # まず全行のキャッシュ有効性をチェックし、再生成が必要な行だけ集める
    pending: list[tuple[str, str, list[int], int, str]] = []
    stale_count = 0
    for line in tqdm(lines, desc="checking cache", file=sys.stdout, dynamic_ncols=True):
        wav_path, text, word2ph, phone_len, key = parse_line(line, add_blank)
        bert_path = str(Path(wav_path).with_suffix(".bert.pt"))
        cached, stale = load_bert_feature(bert_path, key)
        if cached is not None and not stale and cached.shape[-1] == phone_len:
            continue  # 有効なキャッシュ。読み飛ばす
        if stale:
            stale_count += 1
        pending.append((wav_path, text, word2ph, phone_len, key))

    if stale_count > 0:
        logger.warning(
            f"{stale_count} 件の .bert.pt はテキストまたはユーザー辞書が変更されているため再生成します"
        )

    if len(pending) != 0:
        logger.info(
            f"{len(pending)} 件の .bert.pt を生成します（forward バッチサイズ {args.batch_size}）"
        )
        features = extract_bert_feature_batch(
            [text for _, text, _, _, _ in pending],
            [word2ph for _, _, word2ph, _, _ in pending],
            DEVICE,
            batch_size=args.batch_size,
        )
        for (wav_path, _, _, phone_len, key), feature in zip(pending, features):
            assert feature.shape[-1] == phone_len, wav_path
            save_bert_feature(str(Path(wav_path).with_suffix(".bert.pt")), feature, key)

    logger.info(f"bert.pt is generated! total: {len(lines)} bert.pt files.")
