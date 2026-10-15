import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from style_bert_vits2.constants import DATASET_ROOT
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import clean_text
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict


def _init_pyopenjtalk_worker() -> None:
    """pyopenjtalk ワーカーを起動し、dict_data/ 以下のユーザー辞書を適用する。

    モジュール import 時に走らせると `import preprocess.text` しただけでワーカープロセスが
    起動するため（分割ロジックのユニットテストが重くなる）、CLI エントリポイントからのみ
    呼ぶ関数に分離した。`python -m preprocess.text` では __main__ が必ず呼ぶ。
    """
    # このプロセスからはワーカーを起動して辞書を使いたいのでここで初期化
    pyopenjtalk_worker.initialize_worker()
    # dict_data/ 以下の辞書データを pyopenjtalk に適用
    update_dict()


# tqdm 進捗表示のための行数カウント
def count_lines(file_path: Path):
    with file_path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def write_error_log(error_log_path: Path, line: str, error: Exception):
    with error_log_path.open("a", encoding="utf-8") as error_log:
        error_log.write(f"{line.strip()}\n{error}\n\n")


def process_line(
    line: str,
    transcription_path: Path,
    correct_path: bool,
    yomi_error: str,
):
    splitted_line = line.strip().split("|")
    if len(splitted_line) != 4:
        raise ValueError(f"Invalid line format: {line.strip()}")
    utt, spk, language, text = splitted_line
    norm_text, phones, tones, word2ph = clean_text(
        text=text,
        raise_yomi_error=(yomi_error != "use"),
    )
    if correct_path:
        utt = str(transcription_path.parent / "wavs" / utt)

    return "{}|{}|{}|{}|{}|{}|{}\n".format(
        utt,
        spk,
        language,
        norm_text,
        " ".join(phones),
        " ".join([str(i) for i in tones]),
        " ".join([str(i) for i in word2ph]),
    )


def speaker_val_count(
    n_utts: int, val_per_lang: int, val_ratio: float, min_val: int
) -> int:
    """話者1名あたりの検証発話数を決める（学習側には必ず1件残す）。

    - val_per_lang > 0: 明示指定として件数で固定する（旧来動作。比率より優先）
    - val_per_lang == 0: 比率ベース。round(n × val_ratio) 件を下限 min_val・上限 n-1 でクランプする。
      下限 min_val はその話者の発話数の 1/5 を超えないよう抑える（10発話の話者に下限20を
      適用して学習データを8割も検証に回さないため。小規模話者は検証件数を増やしても
      検証損失自体がノイズ主体で精度が出ないため、学習データのほうを優先する）
    - val_ratio <= 0: 検証なし（全件学習）

    Returns:
        int: 検証に回す件数（0 で検証なし。必ず n_utts-1 以下）
    """
    if n_utts < 2:
        return 0
    if val_per_lang > 0:
        return min(val_per_lang, n_utts - 1)
    if val_ratio <= 0:
        return 0
    floor = min(min_val, n_utts // 5)
    return min(max(round(n_utts * val_ratio), floor), n_utts - 1)


def split_train_val(
    spk_utt_map: dict[str, list[str]],
    val_per_lang: int,
    val_ratio: float,
    min_val: int,
    max_val_total: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """話者ごとの発話リストから (train_list, val_list) へ決定論的に分割する。

    - 検証件数は話者ごとに決める（speaker_val_count 参照。既定は val_ratio による比率ベース）。
      話者単位で取ることで、特定の話者が検証から完全に欠けるのを防ぐ（検証損失は全話者混合の
      平均なので、欠けた話者の出来不出来が測れない）
    - ただし必ず1件は学習側へ残す（検証が全件を奪ってその話者の学習データが0件になるのを防ぐ。
      発話1件の話者は丸ごと学習側になる）
    - 検証合計が max_val_total を超える場合は話者按比例に縮小する。以前は val_list の先頭から
      切り詰めていたため、話者数が多いと先頭の話者ばかり検証に残る偏りがあった

    Returns:
        tuple[list[str], list[str]]: (train_list, val_list)。元の行順序は保持する
    """
    speakers = list(spk_utt_map)
    counts = [
        speaker_val_count(len(spk_utt_map[s]), val_per_lang, val_ratio, min_val)
        for s in speakers
    ]
    # 比率・件数のどちらの指定でも、発話数が足りなくて削れた話者は後でまとめて警告する
    desired = [
        val_per_lang if val_per_lang > 0 else round(len(spk_utt_map[s]) * val_ratio)
        for s in speakers
    ]

    # max_val_total による縮小は別の警告を出すので、件数不足の話者の判定は縮小前で行う
    pre_cap = list(counts)
    total = sum(counts)
    if 0 <= max_val_total < total:
        orig = list(counts)
        holders = [i for i, c in enumerate(orig) if c > 0]
        # 元々検証を持っていた話者を0件へ落とさない（話者数自体が max_val_total を
        # 超える場合はそれ以上配れないので諦める）。元々0の話者（発話1件など）には配らない
        floor = 1 if len(holders) <= max_val_total else 0
        scaled = [c * max_val_total / total for c in orig]
        counts = [
            (max(floor, int(scaled[i])) if orig[i] > 0 else 0) for i in range(len(orig))
        ]
        # 切り捨て端数は割合の大きかった話者から順に1件ずつ配る（検証を持っていた話者のみ）
        rest = max_val_total - sum(counts)
        order = sorted(holders, key=lambda i: scaled[i] - counts[i], reverse=True)
        for i in order[: max(rest, 0)]:
            counts[i] += 1
        logger.warning(
            f"検証発話が {total} 件で max_val_total({max_val_total}) を超えたため、"
            f"話者按比例で {sum(counts)} 件へ縮小しました。検証を増やしたい場合は "
            "--max_val_total を引き上げてください"
            + (
                ""
                if sum(counts) <= max_val_total
                else f"（話者1件以上の保証のため上限を {sum(counts) - max_val_total} 件超過しています）"
            )
        )

    train_list: list[str] = []
    val_list: list[str] = []
    thin_speakers: list[str] = []
    for index, spk in enumerate(speakers):
        utts = spk_utt_map[spk]
        n_val = counts[index]
        if pre_cap[index] < desired[index]:
            thin_speakers.append(f"{spk}({len(utts)}件)")
        val_indices = set(rng.sample(range(len(utts)), n_val))
        # 元の順序を保ちながらリストに分割
        for i, utt in enumerate(utts):
            if i in val_indices:
                val_list.append(utt)
            else:
                train_list.append(utt)
    if thin_speakers:
        basis = (
            f"val_per_lang({val_per_lang})"
            if val_per_lang > 0
            else f"比率({val_ratio})"
        )
        logger.warning(
            f"発話数が少なく検証件を減らす話者 {len(thin_speakers)} 件（{basis} に対し、"
            f"学習側に最低1件残す・発話数の1/5を上限とする等の調整が入りました）: {thin_speakers[:10]}"
        )
    return train_list, val_list


def preprocess(
    transcription_path: Path,
    cleaned_path: Path | None,
    train_path: Path,
    val_path: Path,
    config_path: Path,
    val_per_lang: int,
    max_val_total: int,
    # clean: bool,
    yomi_error: str,
    correct_path: bool,
    val_ratio: float = 0.05,
    min_val: int = 20,
):
    assert yomi_error in ["raise", "skip", "use"]
    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path.with_name(
            transcription_path.name + ".cleaned"
        )

    error_log_path = transcription_path.parent / "text_error.log"
    if error_log_path.exists():
        error_log_path.unlink()
    error_count = 0

    total_lines = count_lines(transcription_path)

    # transcription_path から 1行ずつ読み込んで文章処理して cleaned_path に書き込む
    with (
        transcription_path.open("r", encoding="utf-8") as trans_file,
        cleaned_path.open("w", encoding="utf-8") as out_file,
    ):
        for line in tqdm(
            trans_file, file=sys.stdout, total=total_lines, dynamic_ncols=True
        ):
            try:
                processed_line = process_line(
                    line,
                    transcription_path,
                    correct_path,
                    yomi_error,
                )
                out_file.write(processed_line)
            except Exception as e:
                # loguru の logger.error に kwargs（encoding 等）を渡すと message に
                # str.format() が走り、セリフ中の { が KeyError を呼ぶ。第1引数のみで渡す
                logger.error(f"An error occurred at line:\n{line.strip()}\n{e}")
                write_error_log(error_log_path, line, e)
                error_count += 1

    transcription_path = cleaned_path

    # 各話者ごとのlineの辞書
    spk_utt_map: dict[str, list[str]] = defaultdict(list)

    # 話者からIDへの写像
    spk_id_map: dict[str, int] = {}

    # 話者ID
    current_sid: int = 0

    # 音源ファイルのチェックや、spk_id_mapの作成
    with transcription_path.open("r", encoding="utf-8") as f:
        audio_paths: set[str] = set()
        count_same = 0
        count_not_found = 0
        # 音声NotFound・重複行はここで黙って捨てられる（データ全体の問題は Step 3 の check_dataset が esd.list ↔ wavs/ を突き合わせて先に検出する）
        for line in f.readlines():
            utt, spk = line.strip().split("|")[:2]
            if utt in audio_paths:
                logger.warning(f"Same audio file appears multiple times: {utt}")
                count_same += 1
                continue
            if not Path(utt).is_file():
                logger.warning(f"Audio not found: {utt}")
                count_not_found += 1
                continue
            audio_paths.add(utt)
            spk_utt_map[spk].append(line)

            # 新しい話者が出てきたら話者IDを割り当て、current_sidを1増やす
            if spk not in spk_id_map:
                spk_id_map[spk] = current_sid
                current_sid += 1
        if count_same > 0 or count_not_found > 0:
            logger.warning(
                f"Total repeated audios: {count_same}, Total number of audio not found: {count_not_found}"
            )

    # 検証・学習分割は決定論的でなければならないため、config.json の train.seed（既定42）でシードする
    with config_path.open("r", encoding="utf-8") as f:
        json_config = json.load(f)
    rng = random.Random(json_config["train"]["seed"])

    train_list, val_list = split_train_val(
        spk_utt_map, val_per_lang, val_ratio, min_val, max_val_total, rng
    )
    n_all = len(train_list) + len(val_list)
    if n_all > 0:
        basis = (
            f"val_per_lang={val_per_lang}（件数固定）"
            if val_per_lang > 0
            else f"val_ratio={val_ratio}（下限 min_val={min_val}、話者数の1/5まで）"
        )
        logger.info(
            f"train/val 分割: 学習 {len(train_list)} 件 / 検証 {len(val_list)} 件"
            f"（全体 {n_all} 件の {100 * len(val_list) / n_all:.1f}%、{basis}、"
            f"上限 max_val_total={max_val_total}、シード train.seed={json_config['train']['seed']}）"
        )
        if 0 < len(val_list) < 20:
            # 検証損失の評価ノイズは 1/sqrt(件数) で減る。20件を切ると収束幅よりノイズが
            # 大きく、早期停止の best 判定がくじ引きになる（test_h 実測: 4件で評価間 1σ≒0.19
            # あり、9000step の実収束幅より大きかった）ため、件数が足りないときは目立たせる
            logger.warning(
                f"検証発話が {len(val_list)} 件しかありません。検証損失の評価ノイズが大きく"
                "早期停止の判定が信頼できません（--val_ratio を上げるか --min_val を上げて "
                "20件以上にすることを推奨します）"
            )

    with train_path.open("w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with val_path.open("w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    # json_config は分割前に読み込み済みのものを使う（train.seed でシードするため）
    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=2, ensure_ascii=False)
    if error_count > 0:
        if yomi_error == "skip":
            logger.warning(
                f"An error occurred in {error_count} lines. Proceed with lines without errors. Please check {error_log_path} for details."
            )
        else:
            # yom_error == "raise"と"use"の場合。
            # "use"の場合は、そもそもyomi_error = Falseで処理しているので、
            # ここが実行されるのは他の例外のときなので、エラーをraiseする。
            logger.error(
                f"An error occurred in {error_count} lines. Please check {error_log_path} for details."
            )
            raise Exception(
                f"An error occurred in {error_count} lines. Please check `Data/you_model_name/text_error.log` file for details."
            )
            # 何故か{error_log_path}をraiseすると文字コードエラーが起きるので上のように書いている
    else:
        logger.info(
            "Training set and validation set generation from texts is complete!"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, required=True, help="モデル名")
    # 「話者ごと」のバリデーションデータ数、言語ごとではない！
    # 元のコードや設定ファイルでval_per_langとなっていたので名前をそのままにしている
    parser.add_argument(
        "--val-per-lang",
        type=int,
        default=0,
        help="検証発話を話者ごとにこの件数へ固定する（0 で val_ratio による比率ベースへ切り替え）。"
        "言語ごとではない（上流の命名互換）",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="検証発話の比率（話者ごとに round(発話数×ratio)。val_per_lang > 0 のときは無視）。"
        "0 で検証なし（全件学習）",
    )
    parser.add_argument(
        "--min-val",
        type=int,
        default=20,
        help="比率ベースの検証件数の下限（ただしその話者の発話数の1/5を超えない）。"
        "検証損失の評価ノイズは 1/sqrt(件数) なので、少なすぎると早期停止が機能しない",
    )
    parser.add_argument(
        "--max-val-total",
        type=int,
        default=100,
        help="検証発話の総上限。超えた分は話者按比例で縮小して学習へ戻す",
    )
    parser.add_argument("--yomi_error", default="raise")
    parser.add_argument("--correct_path", action="store_true")

    args = parser.parse_args()

    # ワーカー起動・辞書適用は CLI から呼ぶときだけ（import 副作用にしない。関数 docstring 参照）
    _init_pyopenjtalk_worker()

    # データセット内のファイル名は契約として固定（Data/<モデル名>/ 配下に配置される）
    dataset_path = DATASET_ROOT / args.model_name
    transcription_path = dataset_path / "esd.list"
    train_path = dataset_path / "train.list"
    val_path = dataset_path / "val.list"
    config_path = dataset_path / "config.json"

    preprocess(
        transcription_path=transcription_path,
        cleaned_path=None,
        train_path=train_path,
        val_path=val_path,
        config_path=config_path,
        val_per_lang=args.val_per_lang,
        max_val_total=args.max_val_total,
        yomi_error=args.yomi_error,
        correct_path=args.correct_path,
        val_ratio=args.val_ratio,
        min_val=args.min_val,
    )
