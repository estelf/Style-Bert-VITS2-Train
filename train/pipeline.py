import argparse
import datetime
import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import statistics
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler

# logging.getLogger("numba").setLevel(logging.WARNING)
from style_bert_vits2.constants import (
    ASSETS_ROOT,
    BASE_DIR,
    BUCKET_BOUNDARIES,
    DATASET_ROOT,
    MAX_SPEC_FRAMES,
    MIN_SPEC_FRAMES,
    TRAIN_ENV_DEFAULTS,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models_jp_extra import (
    DurationDiscriminator,
    MultiPeriodDiscriminator,
    SynthesizerTrn,
    WavLMDiscriminator,
)
from style_bert_vits2.nlp.symbols import SYMBOLS
from train import default_style
from train.data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from train.losses import (
    WavLMLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # 学習が壊れる（音質ノイズ・NaN 等）場合は TF32 を無効化して確認すること
)
torch.set_num_threads(1)
torch.set_float32_matmul_precision("medium")
# torch.backends.cuda.sdp_kernel("flash") は deprecated（sdp_kernel 自体が廃止予定）。下 enable_* で同等の設定を明示する
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# 決定論モード（デフォルト OFF）。回帰テストのゴールデン音声比較時のみ環境変数 SBV2_DETERMINISTIC=1 で有効化する。
# 本番の長時間学習では cuDNN の非決定的最適カーネルが使えるほうが速いため、常時 ON にしない。
# CUBLAS_WORKSPACE_CONFIG は cuBLAS ハンドル生成時に読まれるため、CUDA 呼び出しより前のモジュール先頭で設定する。
# warn_only=True: 決定論カーネルが無い op は警告だけ出して従来カーネルで走る。torch バージョン更新で
# 新規 op が非決定扱いになるとき、学習・テスト自体が RuntimeError で止まらないため（回帰テストは
# 相関ベースの差分検知なので、警告レベルの非決定性までは要求しない）
DETERMINISTIC = os.environ.get("SBV2_DETERMINISTIC", "0") == "1"
if DETERMINISTIC:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# global_step はオプティマイザ更新数ではなくマイクロバッチ（forward-backward）回数を数える。
# accumulation_steps > 1 の実行では eval_interval / early_stopping_min_steps などのステップ基準も
# すべてマイクロバッチ単位になる（実オプティマイザ更新数は global_step / accumulation_steps）
global_step = 0

# 検証損失の履歴（eval_history.json と同期。早期停止の判定に使う。global_step と同じく
# モジュールレベルの状態として扱うのがこのファイルの既存パターン）
_eval_history: list[dict] = []

# 今回の val.list のハッシュ（早期停止履歴の検証セット同一性判定用。run() で設定）
_val_hash: str | None = None

# 検証損失の測定方法のバージョン（eval_history.json の metric_version に書き込む）。
# v1 = segment_size スライス1点を測定、v2 = 復号音声の全文を測定。
# 絶対値も評価ノイズ量も別物なので、旧バージョンの履歴を同じ best 判定に混ぜると
# 「旧: スライス / 新: 全文」の取り比べになり早期停止が壊れる。run() の読み込みで除外する
_VAL_METRIC_VERSION = 2

# 生成器の EMA 状態（パラメータ名 → float32 テンソル）。ema_decay > 0 のとき rank 0 だけが保持する
_ema_state: dict[str, torch.Tensor] | None = None

# 非有限損失（NaN/inf）の連続スキップ回数。閾値超過で学習停止（全ランク同期で判定する）
_nan_skip_count = 0

# best.safetensors へコピー済みの best step（同じ best のたびに ~120MB コピーし直さないため）
_best_copied_step: int | None = None

# early_stopping_action="reduce_lr" の学習率削減状態（run() で初期化・履歴から復元し、
# train_and_evaluate の評価ステップで更新する）。_lr_reduced_step は直近の削減 step で、
# 削減後の評価だけを patience/プラトー判定に使うための境界になる
_lr_reduce_count = 0
_lr_reduced_step: int | None = None


def _ipv6_available() -> bool:
    """IPv6 ソケットを作れるか（ループバック ::1 が使える環境かの判定）"""
    try:
        socket.socket(socket.AF_INET6, socket.SOCK_STREAM).close()
        return True
    except OSError:  # IPv6 無効環境
        return False


def _free_port() -> str:
    """空きポートを選ぶ（既定ポート固定だと同時実行する2本目が衝突するため）"""
    if _ipv6_available():
        family, bind_addr = socket.AF_INET6, "::1"
    else:  # IPv6 無効環境では IPv4 ループバックで代替
        family, bind_addr = socket.AF_INET, "127.0.0.1"
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.bind((bind_addr, 0))
        return str(s.getsockname()[1])


def _val_list_hash(path: str) -> str | None:
    """val.list の内容ハッシュ。中身が変わったら早期停止履歴の比較可能性が失われることの検出に使う"""
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _validate_train_config(hps) -> None:
    """学習開始前に config.json の train 設定を検証する（実行途中で ZeroDivisionError 等になるのを防ぐ）"""
    t = hps.train
    if t.log_interval <= 0:
        raise ValueError(
            f"train.log_interval は 1 以上が必要です（現在: {t.log_interval}）"
        )
    if t.eval_interval <= 0:
        raise ValueError(
            f"train.eval_interval は 1 以上が必要です（現在: {t.eval_interval}）"
        )
    if t.accumulation_steps < 1:
        raise ValueError(
            f"train.accumulation_steps は 1 以上が必要です（現在: {t.accumulation_steps}）"
        )
    if t.warmup_epochs < 0:
        raise ValueError(
            f"train.warmup_epochs は 0 以上が必要です（現在: {t.warmup_epochs}）"
        )
    if not (t.ema_decay == 0 or 0 < t.ema_decay < 1):
        raise ValueError(
            f"train.ema_decay は 0（無効）または 0〜1 の実数が必要です（現在: {t.ema_decay}）"
        )
    if t.batch_size < 1:
        raise ValueError(
            f"train.batch_size は 1 以上が必要です（現在: {t.batch_size}）"
        )
    if t.epochs < 1:
        raise ValueError(f"train.epochs は 1 以上が必要です（現在: {t.epochs}）")
    if t.num_workers < 0:
        raise ValueError(
            f"train.num_workers は 0 以上が必要です（現在: {t.num_workers}。0 でメインプロセス読み込み）"
        )
    if t.learning_rate <= 0:
        raise ValueError(
            f"train.learning_rate は 0 より大きい値が必要です（現在: {t.learning_rate}）"
        )
    if hps.data.hop_length <= 0 or t.segment_size % hps.data.hop_length != 0:
        raise ValueError(
            f"train.segment_size は data.hop_length（{hps.data.hop_length}）の整数倍が必要です"
            f"（現在: {t.segment_size}）"
        )
    if t.early_stopping_patience < 1:
        raise ValueError(
            f"train.early_stopping_patience は 1 以上が必要です（現在: {t.early_stopping_patience}）"
        )
    if t.early_stopping_min_steps < -1:
        raise ValueError(
            "train.early_stopping_min_steps は -1（自動: max(1000, 5×1エポックのstep数)、"
            f"学習開始時に解決）、0（全履歴対象）、正の固定step数のいずれかが必要です"
            f"（現在: {t.early_stopping_min_steps}）"
        )
    if t.early_stopping_min_delta < 0 or t.early_stopping_min_delta_rel < 0:
        raise ValueError(
            "train.early_stopping_min_delta / early_stopping_min_delta_rel は"
            f"0 以上が必要です（現在: {t.early_stopping_min_delta}, {t.early_stopping_min_delta_rel}）"
        )
    if t.early_stopping_plateau_window != 0 and t.early_stopping_plateau_window < 3:
        raise ValueError(
            "train.early_stopping_plateau_window は 0（無効）または 3 以上が必要です"
            f"（現在: {t.early_stopping_plateau_window}。回帰傾きの推定に3点以上必要）"
        )
    if t.early_stopping_action not in ("stop", "reduce_lr"):
        raise ValueError(
            f"train.early_stopping_action は 'stop' または 'reduce_lr' のいずれかが必要です"
            f"（現在: {t.early_stopping_action!r}）"
        )
    if not (0 < t.early_stopping_reduce_lr_factor < 1):
        raise ValueError(
            f"train.early_stopping_reduce_lr_factor は 0〜1 の実数が必要です（現在: {t.early_stopping_reduce_lr_factor}）"
        )
    if t.early_stopping_reduce_lr_times < 0:
        raise ValueError(
            f"train.early_stopping_reduce_lr_times は 0 以上が必要です（現在: {t.early_stopping_reduce_lr_times}）"
        )
    if t.early_stopping_monitor not in ("mel", "dur", "kl", "lm", "total"):
        logger.warning(
            f"early_stopping_monitor='{t.early_stopping_monitor}' は不明です。'mel' に正規化します"
        )
        t.early_stopping_monitor = "mel"


def _all_ranks_finite(losses: list[torch.Tensor], device) -> bool:
    """損失が全ランク・全要素で有限かを判定する。

    NaN ガードの分岐はランク間で必ず同じ判定にしないと、あるランクだけ backward を
    飛ばして DDP の集合通信が不一致になりハングする。そのため GPU 上のフラグを
    all_reduce で集約してから .item() で同期する（1ステップ1回の同期は許容）
    """
    bad = torch.zeros((), dtype=torch.int32, device=device)
    for loss in losses:
        bad += (~torch.isfinite(loss.detach())).to(torch.int32).sum()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(bad)
    return bad.item() == 0


def _zero_loss(loss: torch.Tensor) -> torch.Tensor:
    """非有限損失を backward 可能なゼロに置き換える。

    単純に backward を飛ばすと DDP の Reducer が「前回の backward が未完」で次回
    forward に失敗するため、計算グラフは保ったまま値と勾配だけがゼロになるよう
    nan_to_num 後に 0 倍する（detach しないこと。グラフは必要）。
    forward 中間値自体が NaN の場合 backward で NaN 勾配が再発しうるが、
    optim.step() を呼ばず zero_grad で破棄するため重みは汚染されない
    """
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0) * 0.0


def _safetensors_step(path: Path) -> int:
    """<モデル名>_e<epoch>_s<step>[_ema].safetensors から step を取り出す（取れなければ -1）"""
    m = re.search(r"_s(\d+)(?:_ema)?\.safetensors$", path.name)
    return int(m.group(1)) if m else -1


def _clean_safetensors(
    assets_dir: str | Path, keep: int, best_step: int | None = None
) -> None:
    """model_assets/ 配下の .safetensors を step 単位で keep 個に絞る（0 で全残し）。

    best.safetensors（推論に使うべき1点のコピー）・best_step のファイル・step を解釈
    できないファイルは削除対象から除外する。早期停止の監視 best とは別に、合成品質評価
    select_best.py の候補として全残しにしてきたが、eval_interval=1000 の長期学習で数百個
    溜まり disk を圧迫するため、保持上限を設定できるようにした
    """
    if keep <= 0:
        return
    files = [
        p
        for p in Path(assets_dir).glob("*.safetensors")
        # best.safetensors（推論用1点コピー）と step を解釈できないファイルは削除しない
        if p.name != "best.safetensors" and _safetensors_step(p) >= 0
    ]
    steps = sorted({_safetensors_step(p) for p in files}, reverse=True)
    keep_steps = set(steps[:keep])
    if best_step is not None:
        keep_steps.add(best_step)
    for p in files:
        if _safetensors_step(p) not in keep_steps:
            try:
                p.unlink()
                logger.info(f"Free up space by deleting safetensors {p.name}")
            except OSError as e:
                logger.warning(f"{p.name} の削除に失敗しました: {e}")


def _filter_eval_dataset(eval_dataset, min_frames: int, max_frames: int) -> int:
    """評価データセットから (min_frames, max_frames]（spec フレーム数。下限排他的）外のサンプルを in-place で除外し、件数を返す。

    学習のバケット除外（DistributedBucketSampler の _bisect は「境界 < 長さ」の strict 比較で
    MIN_SPEC_FRAMES < 長さ <= MAX_SPEC_FRAMES を保持）と同じ基準。ちょうど下限フレーム数の
    サンプルも学習では使われないので評価からも外す（基準を揃えないと、学習で一切使われない
    サンプルの損失が早期停止の判定に混ざる）。長尺サンプルが評価に残っていると evaluate() の
    infer(max_len=1000) で生成音声だけが約 11.6 秒に打ち切られ、フル長の gt/audio_ と比較できない
    （TensorBoard の音声プレビューが意味をなさない）
    """
    keep_entries, keep_lengths = [], []
    for entry, length in zip(eval_dataset.audiopaths_sid_text, eval_dataset.lengths):
        if min_frames < length <= max_frames:
            keep_entries.append(entry)
            keep_lengths.append(length)
    dropped = len(eval_dataset.audiopaths_sid_text) - len(keep_entries)
    if dropped > 0:
        eval_dataset.audiopaths_sid_text = keep_entries
        eval_dataset.lengths = keep_lengths
    return dropped


def _ema_update(net_g, decay: float) -> None:
    """生成器パラメータの EMA を更新する（rank 0 のみが _ema_state を持つ）"""
    if _ema_state is None:
        return
    module = getattr(net_g, "module", net_g)
    with torch.no_grad():
        for name, p in module.named_parameters():
            _ema_state[name].mul_(decay).add_(p.detach().float(), alpha=1.0 - decay)


def _save_ema_safetensors(net_g, epoch: int, path: str) -> None:
    """EMA 重みを一時ロードして _ema.safetensors として保存し、直後に元重みへ戻す。

    学習継続に EMA の影響を与えないため、保存の一瞬だけ重みを差し替える。
    推論・select_best.py 評価では通常の .safetensors と並ぶ候補になる
    """
    if _ema_state is None:
        return
    module = getattr(net_g, "module", net_g)
    backup = {n: p.detach().clone() for n, p in module.named_parameters()}
    try:
        with torch.no_grad():
            for n, p in module.named_parameters():
                p.copy_(_ema_state[n].to(p.dtype))
        utils.safetensors.save_safetensors(module, epoch, path, for_infer=True)
    finally:
        with torch.no_grad():
            for n, p in module.named_parameters():
                p.copy_(backup[n])


def _reduce_lr(schedulers: list, factor: float) -> None:
    """プラトー時の学習率削減（ReduceLROnPlateau 相当）。

    LambdaLR は scheduler.step() ごとに lr = base_lrs × λ(epoch) を再計算するため、
    param_groups["lr"] だけ掛けると次回エポック境界で元のカードルに巻き戻る。
    base_lrs 自体をスケールして削減をエポックをまたいで永続化し、現在の lr も
    即時に下げる（全ランクが同一の factor で呼ぶこと。ランク間で LR がズレると
    DDP 学習が壊れるため、呼出は必ず broadcast 後の全ランク同期点から行う）。

    再開時の復元（run() 側）でも同じく両方へ factor^prior を掛ける。torch の
    LRScheduler は last_epoch>=0 で構築されたとき param_groups["lr"] を
    base_lrs × λ から再計算して上書きする（公式 docstring の警告どおり）。
    pipeline は optimizer 読込後にスケジューラを構築するため、optimizer 状態の
    削減済み lr は構築で失われ「削減前のカードル」に落ちる。base_lrs の出典である
    initial_lr は _reduce_lr で変更しないので、構築後の base_lrs・lr ともに
    未削減（=config 値基準）であり、両方へ factor^prior を掛けることで正確に
    復元される（torch 2.14 で実測確認済み）
    """
    for scheduler in schedulers:
        if scheduler is None:
            continue
        scheduler.base_lrs = [base * factor for base in scheduler.base_lrs]
        for pg in scheduler.optimizer.param_groups:
            pg["lr"] *= factor


def _apply_lr_reduction(hps, schedulers: list) -> None:
    """broadcast 到達後、全ランクで学習率削減を実行し削減状態も更新する。

    _lr_reduced_step（直近の削減 step）は patience/プラトー判定の窓の開始点になる。
    削減評価に書き込まれた eval_history エントリには削減「前」の回数が残るため、
    再開時の復元（run() 側）は「カウントが prior-1 だった最後の評価 step」を
    削減発生点とみなす
    """
    global _lr_reduce_count, _lr_reduced_step
    _reduce_lr(schedulers, hps.train.early_stopping_reduce_lr_factor)
    _lr_reduce_count += 1
    _lr_reduced_step = global_step


def run():
    # 学習対象は --model_name で明示する（順序や状態ファイルに依存しない冪等設計）
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
        help="データセット名（DATASET_ROOT 配下。デフォルトは Data/、環境変数 SBV2_DATASET_ROOT で差し替え可）。設定は <データセット>/config.json から読まれる",
    )
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Do not show the progress bar while training.",
    )
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="Speed up training by disabling logging and evaluation.",
    )
    parser.add_argument(
        "--not_use_custom_batch_sampler",
        help="Don't use custom batch sampler for training, which was used in the version < 2.5",
        action="store_true",
    )
    parser.add_argument(
        "--styles_by_dirs",
        action="store_true",
        help="オプション: wavs/ のサブディレクトリごとにスタイルベクトルを生成する（既定は Neutral 1本のみ。フォルダ分けしていても使われない）。Neutral でも十分高い精度が出るため通常は不要",
    )
    parser.add_argument(
        "--allow_scratch",
        action="store_true",
        help="オプション: 再開時のチェックポイント読み込み失敗（破損・config 不一致など）でもスクラッチ学習へフォールバックする。既定は失敗時に停止して学習成果物の消失を防ぐ",
    )
    args = parser.parse_args()

    # 学習対象のパスはモデル名から一意に定まる（状態管理なし・冪等）
    model_name = args.model_name
    dataset_path = str(DATASET_ROOT / model_name)
    config_path = str(DATASET_ROOT / model_name / "config.json")
    out_dir = str(ASSETS_ROOT / model_name)

    # -m の打ち間違いで存在しないモデル名を渡されたら、ログ出力より前に候補を挙げて失敗する。
    # このチェックがないと loguru が Data/<打ち間違い名>/ を自動作成し、空のゴミディレクトリと
    # train_*.log が残って後から片付けるはめになる（前処理は preprocess_all.py が作る）
    if not Path(config_path).is_file():
        available = (
            sorted(
                p.name
                for p in DATASET_ROOT.iterdir()
                if p.is_dir() and (p / "config.json").is_file()
            )
            if DATASET_ROOT.is_dir()
            else []
        )
        sys.exit(
            f"{config_path} が見つかりません。モデル名（-m {model_name!r}）を確認してください。"
            f"学習可能なデータセット: {available or 'なし（先に preprocess_all.py -m <モデル名> を実行してください）'}"
        )

    # ログファイルの設定
    model_dir = os.path.join(dataset_path, "models")
    # マルチGPUでは全ランクが同じログファイルを掴んで混線するため、rank 0 だけがファイルへ書く
    # （他ランクはコンソール出力のみ。torchrun 経由ならコンソールがまとめて残る）。
    # ファイル名の日時は loguru の {time} マクロ（Python 側で整形した文字列だと loguru が
    # パターンを認識できず retention が効かない）。実行のたびに増え続けないよう直近10本だけ保持
    rank_for_log = int(os.environ.get("RANK", TRAIN_ENV_DEFAULTS["RANK"]))
    if rank_for_log == 0:
        logger.add(
            os.path.join(dataset_path, "train_{time:YYYYMMDD_HHmmss}.log"),
            retention=10,
        )

    # 環境変数の読み込み（デフォルトは定数。環境変数が既に設定されている場合はそれを優先）
    envs = dict(TRAIN_ENV_DEFAULTS)
    # 既定ポートの固定値は同時実行する2本目が衝突するため、未設定のときだけ空きポートを選ぶ
    # （torchrun 経由なら MASTER_PORT はランチャー側で設定されるのでここは通らない）
    picked_port = "MASTER_PORT" not in os.environ
    if picked_port:
        envs["MASTER_PORT"] = _free_port()
    # MASTER_ADDR の既定 "::1" は WSL2 mirrored mode 対策のリテラルだが、IPv6 自体が使えない
    # 環境では待ち受けに失敗する。_free_port は IPv4 にフォールバックするのにアドレスが ::1 のままだと
    # 不整合になるため、IPv6 不可のときだけ 127.0.0.1 に落とす（明示的な環境変数があればそれを尊重）
    if "MASTER_ADDR" not in os.environ and not _ipv6_available():
        envs["MASTER_ADDR"] = "127.0.0.1"
        logger.warning(
            "IPv6 ループバック (::1) が使えないため MASTER_ADDR を 127.0.0.1 にフォールバックします"
        )
    for env_name, env_value in envs.items():
        if env_name not in os.environ.keys():
            logger.info(f"Loading configuration from config {env_value!s}")
            os.environ[env_name] = str(env_value)
    logger.info(
        "Loading environment variables \nMASTER_ADDR: {},\nMASTER_PORT: {},\nWORLD_SIZE: {},\nRANK: {},\nLOCAL_RANK: {}".format(
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
            os.environ["LOCAL_RANK"],
        )
    )

    backend = "nccl"
    if platform.system() == "Windows":
        backend = "gloo"  # Windows では gloo バックエンドへ切り替える
    # timeout: 評価ステップでは rank 0 だけが検証音声の生成＋チェックポイント4点保存を行い、
    # 他ランクは次の集合通信で待つ。検証文が多い・ディスクが遅い環境では 300 秒では足りず、
    # 学習中盤で NCCL タイムアウト落ちするため 30 分に拡大する
    world_size_env = int(os.environ.get("WORLD_SIZE", TRAIN_ENV_DEFAULTS["WORLD_SIZE"]))
    for attempt in range(3):
        try:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=datetime.timedelta(seconds=1800),
            )  # 分散起動は torchrun 経由（mp.spawn は使わない）
            break
        except Exception as e:
            # 空きポート選択には close してから bind までの TOCTOU の窓がある。
            # 自前でポートを選んだ単一プロセス起動でのみ再試行して解消する
            # （torchrun 経由や複数ランクでは他ランクが旧ポートを待ち続けるので再試行できない）
            if not (picked_port and world_size_env == 1) or attempt == 2:
                raise
            logger.warning(
                f"MASTER_PORT {os.environ['MASTER_PORT']} の待ち受けに失敗しました ({e})。空きポートで再試行します"
            )
            os.environ["MASTER_PORT"] = _free_port()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    n_gpus = dist.get_world_size()
    # NCCL の最初の集合通信（スタイル生成後の barrier）より前にデバイスを確定させる。
    # 後回しだと全ランクがデフォルトの cuda:0 で communicator を作ろうとして失敗・ハングする
    torch.cuda.set_device(local_rank)

    hps = HyperParameters.load_from_json(config_path)
    # train_and_evaluate() へ渡す必要があるため、実行ごとの値をここで hps に設定する
    hps.model_name = model_name
    hps.model_dir = model_dir
    hps.out_dir = out_dir
    hps.dataset_path = dataset_path
    hps.speedup = args.speedup
    _validate_train_config(hps)
    os.makedirs(out_dir, exist_ok=True)

    # val.list の内容ハッシュ（早期停止履歴が別の検証セット上の損失と混ざらないようにする）
    global _val_hash
    _val_hash = _val_list_hash(hps.data.validation_files)

    # 再開時は推論資産（config.json + style_vectors.npy）が既に揃っているのでスタイル生成は自動でスキップする（CLI指定不要）
    # マルチGPUでは全ランクが同じ style_vectors.npy / config.json を同時に書かないよう rank 0 のみ実行し、他ランクは完了を待つ
    if rank == 0:
        if (Path(out_dir) / "config.json").exists() and (
            Path(out_dir) / "style_vectors.npy"
        ).exists():
            logger.info(
                f"Style assets already exist in {out_dir}, so style generation is skipped automatically (resuming)."
            )
            # スキップしても Data 側の config.json は前処理のたびに更新されているため、話者情報だけを同期する
            default_style.sync_inference_config(
                config_path=config_path,
                config_output_path=os.path.join(out_dir, "config.json"),
            )
            # --styles_by_dirs はスタイル資産が存在すれば再生成されず黙って無視される。
            # フラグの意味が実行ごとに変わっていても気づけないため、既存資産のスタイル数と
            # 照合して食い違いだけを警告する（一致していれば何も出さない）
            try:
                out_config = json.loads(
                    (Path(out_dir) / "config.json").read_text(encoding="utf-8")
                )
                num_styles = int(out_config.get("data", {}).get("num_styles", 1))
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                num_styles = 1
            if args.styles_by_dirs and num_styles <= 1:
                logger.warning(
                    "--styles_by_dirs を指定しましたが、既存のスタイル資産（Neutral 1本）が"
                    "そのまま使われます（再開時はスタイル生成がスキップされるため）。反映するには "
                    f"{out_dir}/style_vectors.npy を削除して再実行してください（既存 .safetensors と"
                    "スタイル ID がずれないよう、スタイル構成を変えたら学習のやり直しを推奨します）"
                )
        else:
            # 既定は Neutral 1本のみ（train.list / val.list の発話が算出対象）。サブディレクトリごとのスタイル生成は --styles_by_dirs のときだけ
            default_style.save_style_vectors(
                [hps.data.training_files, hps.data.validation_files],
                os.path.join(dataset_path, "wavs"),
                out_dir,
                config_path=config_path,
                config_output_path=os.path.join(out_dir, "config.json"),
                styles_by_dirs=args.styles_by_dirs,
            )
    if n_gpus > 1:
        dist.barrier()

    torch.manual_seed(hps.train.seed)
    # シード固定だけでは消せない演算カーネルレベルの非決定性を潰す（回帰テストのゴールデン比較の前提）。
    # 決定論カーネルは遅いため、環境変数 SBV2_DETERMINISTIC=1 のときのみ有効化する（本番学習は既定 OFF）。
    # warn_only=True にし、決定論カーネルが無い op でも例外で止めない（torch 更新耐性。前述の通り
    # 回帰テストは相関ベースの差分検知なのでビット単位一致は要求しない）
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True, warn_only=True)

    global global_step
    writer = None
    writer_eval = None
    if rank == 0 and not args.speedup:
        utils.check_git_hash(model_dir)
        writer = SummaryWriter(log_dir=model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))
    train_dataset = TextAudioSpeakerLoader(
        hps.data.training_files, hps.data, spec_cache=hps.train.spec_cache
    )
    collate_fn = TextAudioSpeakerCollate()
    if not args.not_use_custom_batch_sampler:
        # バケット境界は constants の単一ソースを参照。DistributedBucketSampler は空バケット
        # 除去で受け取ったリストを pop で破壊するため、必ずコピーを渡す（定数本体が汚染されると
        # 同一プロセスで複数回学習するテストや再開時に境界が欠ける）
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size,
            list(BUCKET_BOUNDARIES),
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            # num_workers は config の train.num_workers（既定2）。.spec.pt キャッシュが無い
            # 初回エポックの STFT が CPU 並列になり、メモリが厳しい環境では 1 に下げられる
            num_workers=hps.train.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            # batch_size=hps.train.batch_size,
            # persistent_workers は num_workers>0 のときだけ有効にできる（0 で True にすると
            # DataLoader 構築時に ValueError）。num_workers=0 はメインプロセス読み込み
            persistent_workers=hps.train.num_workers > 0,
            # これもメモリ消費量を減らそうとしてコメントアウト
            # prefetch_factor=6,
        )
    else:
        # この経路は長さによる除外をしないため、segment_size 未満（既定 32 フレーム）の音声だと
        # commons.slice_segments の torch.gather がインデックス範囲外でクラッシュする。
        # check_dataset（Step 3）は範囲外を警告のみで通すため、ここで明示的に除外する
        # （カスタムバッチサンプラー経路ではバケット境界外のサンプルは黙って除外される）
        min_frames = hps.train.segment_size // hps.data.hop_length
        short_indices = {
            i for i, length in enumerate(train_dataset.lengths) if length < min_frames
        }
        if short_indices:
            logger.warning(
                f"{len(short_indices)} samples are shorter than segment_size ({min_frames} spec frames) "
                "and would crash slice_segments. They are excluded from training "
                "(--not_use_custom_batch_sampler path)."
            )
            train_dataset.audiopaths_sid_text = [
                x
                for i, x in enumerate(train_dataset.audiopaths_sid_text)
                if i not in short_indices
            ]
            train_dataset.lengths = [
                length
                for i, length in enumerate(train_dataset.lengths)
                if i not in short_indices
            ]
        train_sampler = DistributedLengthGroupedSampler(
            dataset=train_dataset,
            batch_size=hps.train.batch_size,
            num_replicas=n_gpus,
            rank=rank,
            lengths=train_dataset.lengths,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            # num_workers は config の train.num_workers（既定2。上のカスタムバッチサンプラー側と同じ）
            num_workers=hps.train.num_workers,
            # shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
            batch_size=hps.train.batch_size,
            # persistent_workers は num_workers>0 のときだけ（上のバケットサンプラー側と同じ理由）
            persistent_workers=hps.train.num_workers > 0,
            # これもメモリ消費量を減らそうとしてコメントアウト
            # prefetch_factor=6,
        )
        # この経路は長さによる除外をしないため、32フレーム未満の音声だと commons.slice_segments の
        # torch.gather がインデックス範囲外になる。上の除外フィルタで防ぐ
        logger.info("Using DistributedLengthGroupedSampler for training.")
        logger.debug(f"len(train_dataset): {len(train_dataset)}")
        logger.debug(f"len(train_loader): {len(train_loader)}")

    eval_dataset = None
    eval_loader = None
    if rank == 0 and not args.speedup:
        eval_dataset = TextAudioSpeakerLoader(
            hps.data.validation_files, hps.data, spec_cache=hps.train.spec_cache
        )
        # 評価も学習バケットと同じ基準（MIN < フレーム数 <= MAX。下限排他はサンプラーの
        # strict 比較と揃えた）で除外する。長尺サンプルが残ると TensorBoard の音声プレビューで
        # 生成だけが max_len=1000 に打ち切られ、フル長の gt と比較できない。
        # 早期停止の val 損失から学習で一切使われないサンプルを外す意味もある
        n_eval_dropped = _filter_eval_dataset(
            eval_dataset, MIN_SPEC_FRAMES, MAX_SPEC_FRAMES
        )
        if n_eval_dropped > 0:
            logger.warning(
                f"val.list のうち {n_eval_dropped} 件が評価バケット範囲 ({MIN_SPEC_FRAMES}, {MAX_SPEC_FRAMES}] "
                "フレーム外のため評価から除外します（生成音声が max_len=1000 で打ち切られ gt と比較できないため）"
            )
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    # 早期停止は val.list の検証損失で判定するため、検証データが無い・評価を無効化している
    # 状態では黙って機能せず、epochs 上限（既定 1000）まで走る。暴走に見えないよう開始時に警告する
    if rank == 0 and hps.train.early_stopping_enabled:
        if args.speedup:
            logger.warning(
                "--speedup のため評価を行わず、早期停止は無効になります（epochs 上限まで学習します）"
            )
        elif eval_dataset is None or len(eval_dataset) == 0:
            logger.warning(
                f"val.list が空のため早期停止は機能しません（epochs 上限 {hps.train.epochs} まで学習します）。"
                "検証データを置くか、early_stopping_enabled: false を明示してください"
            )
    # early_stopping_min_steps=-1（自動）をデータセット規模から concrete な step 数へここで解決し、
    # hps へ書き戻す（以降の _early_stop_check / _best_eval_entry は解決済み値だけを参照する。
    # Data 側の config.json はユーザー指定のまま書き換えない）。各ランクが同一データセットから
    # 同じ値を計算するので DDP でも不一致は起きない
    if hps.train.early_stopping_min_steps < 0:
        resolved_min_steps = _resolve_early_stopping_min_steps(hps, len(train_loader))
        if rank == 0 and hps.train.early_stopping_enabled:
            logger.info(
                f"early_stopping_min_steps=auto: 学習 {len(train_dataset)} 文・"
                f"{len(train_loader)} step/エポック → 経過 {resolved_min_steps} step 以降の評価から判定開始"
            )
    if hps.model.use_noise_scaled_mas is True:
        logger.info("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        logger.info("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    if hps.model.use_duration_discriminator is True:
        logger.info("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(local_rank)
    else:
        net_dur_disc = None
    if hps.model.use_wavlm_discriminator is True:
        net_wd = WavLMDiscriminator(
            hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
        ).cuda(local_rank)
    else:
        net_wd = None
    if hps.model.use_spk_conditioned_encoder is True:
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        logger.info("Using normal encoder for VITS1")

    net_g = SynthesizerTrn(
        len(SYMBOLS),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        # hps.model 以下のすべての値を引数に渡す
        use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
        use_duration_discriminator=hps.model.use_duration_discriminator,
        use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_layers_q=hps.model.n_layers_q,
        use_spectral_norm=hps.model.use_spectral_norm,
        gin_channels=hps.model.gin_channels,
        slm=hps.model.slm,
    ).cuda(local_rank)
    if getattr(hps.train, "freeze_JP_bert", False):
        logger.info("Freezing (JP) bert encoder !!!")
        for param in net_g.enc_p.bert_proj.parameters():
            param.requires_grad = False
    if getattr(hps.train, "freeze_style", False):
        logger.info("Freezing style encoder !!!")
        for param in net_g.enc_p.style_proj.parameters():
            param.requires_grad = False

    if getattr(hps.train, "freeze_decoder", False):
        logger.info("Freezing decoder !!!")
        for param in net_g.dec.parameters():
            param.requires_grad = False

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(local_rank)
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None
    if net_wd is not None:
        optim_wd = torch.optim.AdamW(
            net_wd.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_wd = None
    # gradient_as_bucket_view: 勾配を allreduce バッファの実体（ビュー）に共有させ、
    # バケットコピー分の VRAM を節約する。ビューが壊れるので zero_grad は必ず
    # set_to_none=False（ゼロ埋め）で呼ぶこと（train_and_evaluate 側参照。torch の
    # DDP ドキュメントも detach_ 不可の代替として set_to_none=False を案内している。
    # 本リポジトリに detach_() の呼び出しはないことは確認済み）
    net_g = DDP(
        net_g,
        device_ids=[local_rank],
        gradient_as_bucket_view=True,
        # bucket_cap_mb=512
    )
    net_d = DDP(
        net_d,
        device_ids=[local_rank],
        gradient_as_bucket_view=True,
        # bucket_cap_mb=512
    )
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc,
            device_ids=[local_rank],
            gradient_as_bucket_view=True,
            # bucket_cap_mb=512,
        )
    if net_wd is not None:
        net_wd = DDP(
            net_wd,
            device_ids=[local_rank],
            gradient_as_bucket_view=True,
            #  bucket_cap_mb=512
        )

    if utils.is_resuming(model_dir):
        if net_dur_disc is not None:
            try:
                _, _, dur_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                    utils.checkpoints.get_latest_checkpoint_path(
                        model_dir, "DUR_*.pth"
                    ),
                    net_dur_disc,
                    optim_dur_disc,
                    skip_optimizer=hps.train.skip_optimizer,
                )
                if not optim_dur_disc.param_groups[0].get("initial_lr"):
                    optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
            except Exception as e:
                # 再開失敗時はまだ dur_resume_lr が束縛されていないので参照しない（本当の例外を隠さない）
                if not optim_dur_disc.param_groups[0].get("initial_lr"):
                    optim_dur_disc.param_groups[0][
                        "initial_lr"
                    ] = hps.train.learning_rate
                logger.warning(f"Initialize dur_disc: {e}")
        if net_wd is not None:
            try:
                _, optim_wd, wd_resume_lr, epoch_str = (
                    utils.checkpoints.load_checkpoint(
                        utils.checkpoints.get_latest_checkpoint_path(
                            model_dir, "WD_*.pth"
                        ),
                        net_wd,
                        optim_wd,
                        skip_optimizer=hps.train.skip_optimizer,
                    )
                )
                if not optim_wd.param_groups[0].get("initial_lr"):
                    optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
            except Exception as e:
                # 再開失敗時はまだ wd_resume_lr が束縛されていないので参照しない（本当の例外を隠さない）
                if not optim_wd.param_groups[0].get("initial_lr"):
                    optim_wd.param_groups[0]["initial_lr"] = hps.train.learning_rate
                logger.warning(f"Initialize wavlm: {e}")

        try:
            _, optim_g, g_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth"),
                net_g,
                optim_g,
                skip_optimizer=hps.train.skip_optimizer,
            )
            _, optim_d, d_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "D_*.pth"),
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer,
            )
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr

            epoch_str = max(epoch_str, 1)
            # global_step = (epoch_str - 1) * len(train_loader)
            global_step = (
                utils.get_steps(
                    utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth")
                )
                or 0
            )
            logger.info(
                f"******************Found the model. Current epoch is {epoch_str}, gloabl step is {global_step}*********************"
            )
        except Exception as e:
            # 再開失敗を握り潰してスクラッチ続行すると、keep_ckpts=1（既定）では次の保存後の
            # clean_checkpoints が旧チェックポイントを削除し、学習の進捗が失われる。
            # checkpoints.py の MAX_MISMATCHED_KEYS ガードの意図（大規模不一致を黙って通さない）
            # を機能させるため、既定では停止し、明示的な --allow_scratch のときだけフォールバックする
            if not args.allow_scratch:
                raise RuntimeError(
                    f"チェックポイントからの再開に失敗したため学習を開始できません: {e}。"
                    "config.json（n_speakers / freeze_* など）と既存モデルの組み合わせを確認するか、"
                    "意図的にスクラッチ学習へ切り替える場合は --allow_scratch を指定してください"
                    "（続行すると旧チェックポイントが削除されるおそれがあります）"
                ) from e
            logger.warning(e)
            logger.warning(
                "--allow_scratch が指定されているため、スクラッチ学習にフォールバックします。"
            )
            epoch_str = 1
            global_step = 0
    else:
        try:
            _ = utils.safetensors.load_safetensors(
                os.path.join(model_dir, "G_0.safetensors"), net_g
            )
            _ = utils.safetensors.load_safetensors(
                os.path.join(model_dir, "D_0.safetensors"), net_d
            )
            if net_dur_disc is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(model_dir, "DUR_0.safetensors"), net_dur_disc
                )
            if net_wd is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(model_dir, "WD_0.safetensors"), net_wd
                )
            logger.info("Loaded the pretrained models.")
        except Exception as e:
            if Path(model_dir, "G_0.safetensors").is_file() and not args.allow_scratch:
                # 事前学習モデルは存在するのに読み込みに失敗した（破損・不整合など）。
                # ランダム初期化で黙って続行すると、壊れた土台で学習が進み、
                # そのことに誰も気づかないままチェックポイントが上書きされていく
                raise RuntimeError(
                    f"事前学習モデル {os.path.join(model_dir, 'G_0.safetensors')} の読み込みに失敗しました: {e}。"
                    "ファイルを復元するか、意図的にランダム初期化で学習する場合は --allow_scratch を指定してください"
                ) from e
            logger.warning(e)
            logger.warning(
                "It seems that you are not using the pretrained models, so we will train from scratch."
            )
        finally:
            epoch_str = 1
            global_step = 0

    def lr_lambda(epoch):
        """
        Learning rate scheduler for warmup and (near-)constant LR.
        - During the warmup period, the learning rate increases linearly (epoch は 0 ベースなので +1 する。さないと初回エポックの LR が厳密に 0 になる).
        - After the warmup period, an epoch-unit factor of lr_decay**epochs is applied. With the default 0.99996 it decays only ~0.4% over 100 epochs,
          i.e. effectively a constant LR (上流のステップ単位減衰とは異なる点に注意).
        """
        if epoch < hps.train.warmup_epochs:
            return float(epoch + 1) / float(max(1, hps.train.warmup_epochs))
        else:
            return hps.train.lr_decay ** (epoch - hps.train.warmup_epochs)

    scheduler_last_epoch = epoch_str - 2
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optim_g, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optim_d, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.LambdaLR(
            optim_dur_disc, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
        )
    else:
        scheduler_dur_disc = None
    if net_wd is not None:
        scheduler_wd = torch.optim.lr_scheduler.LambdaLR(
            optim_wd, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
        )
        # slm.model は config テンプレートでは相対パス（"./pretrained/slm/..."）。
        # CWD 依存を避けるため、相対パスで CWD に無い場合は BASE_DIR 基準に解決する
        slm_model = hps.model.slm.model
        slm_path = Path(slm_model)
        if not slm_path.is_absolute() and not slm_path.exists():
            slm_model = str(BASE_DIR / slm_model)
        wl = WavLMLoss(
            slm_model,
            net_wd,
            hps.data.sampling_rate,
            hps.model.slm.sr,
        ).to(local_rank)
    else:
        scheduler_wd = None
        wl = None
    logger.info("Start training.")

    diff = abs(
        epoch_str * len(train_loader) - (hps.train.epochs + 1) * len(train_loader)
    )
    # 進捗バーは rank 0 のみ（他ランクで同じバーを出しても意味がない）。
    # 早期停止が有効な場合 epochs は到達保証のない上限にすぎないので、バーは総数なし
    # （経過step・レートのみ）にして、%/上限で収束を誤解させない
    pbar = None
    if rank == 0 and not args.no_progress_bar:
        pbar = tqdm(
            total=None if hps.train.early_stopping_enabled else global_step + diff,
            initial=global_step,
            smoothing=0.05,
            file=sys.stdout,
            dynamic_ncols=True,
        )
    initial_step = global_step

    # 検証損失履歴を読み込む（再開時も履歴をまたいで早期停止を判定するため。書き込むのは rank 0 のみ）。
    # val.list の内容ハッシュ付きで保存しており、別検証セット上の損失を best 判定に混ぜない
    _eval_history.clear()
    history_path = Path(model_dir) / "eval_history.json"
    if rank == 0 and history_path.is_file():
        try:
            entries = json.loads(history_path.read_text(encoding="utf-8"))
            kept, dropped, legacy, stale = [], 0, 0, 0
            for entry in entries:
                # 測定方法のバージョンが違う履歴は絶対値もノイズ量も別物（v1=スライス1点、
                # v2=全文）。旧履歴を best 判定に混ぜると測定方法の取り比べになるため除外する
                if entry.get("metric_version") != _VAL_METRIC_VERSION:
                    stale += 1
                    continue
                entry_hash = entry.get("val_hash")
                if entry_hash is None:
                    # ハッシュ導入以前の履歴。検証セットが変わったか判定できないため
                    # 一応残す（best 情報を黙って失うほうが害が大きい）が、変更していれば再学習推奨
                    legacy += 1
                    kept.append(entry)
                elif entry_hash == _val_hash:
                    kept.append(entry)
                else:
                    dropped += 1
            _eval_history.extend(kept)
            if stale > 0:
                logger.warning(
                    f"検証損失の測定方法が変わったため、旧測定（metric_version の違う履歴）{stale} 件を"
                    f"早期停止判定から除外しました（現行 v{_VAL_METRIC_VERSION}。"
                    "学習を継続する場合は models/eval_history.json を消すか、"
                    "新測定での評価が十分溜まるまで早期停止が発火しません）"
                )
            if dropped > 0:
                logger.warning(
                    f"val.list が変更されているため、別検証セット上の損失である履歴 {dropped} 件を早期停止判定から除外しました"
                )
            if legacy > 0:
                logger.warning(
                    f"検証セット情報のない旧形式の履歴 {legacy} 件はそのまま採用します。val.list を変更した覚えがあれば eval_history.json を削除してください"
                )
            logger.info(
                f"Loaded {len(_eval_history)} validation history entries from {history_path}"
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"eval_history.json の読み込みに失敗したため空履歴で開始します: {e}"
            )

    # 生成器の EMA（ema_decay > 0 のときのみ）。パラメータは DDP でランク間同期されるため、
    # 更新・保持は rank 0 だけでよい（保存も rank 0 のみ）
    global _ema_state
    _ema_state = None
    if hps.train.ema_decay > 0 and rank == 0:
        _ema_state = {
            name: p.detach().float().clone()
            for name, p in net_g.module.named_parameters()
        }
        logger.info(
            f"EMA を有効化しました (decay={hps.train.ema_decay})。評価保存時に _ema.safetensors も併記します"
        )

    # early_stopping_action="reduce_lr" の削減状態を履歴から復元する。LambdaLR は毎エポック
    # base_lrs から LR を組み立てるため、再開時に過去の削減を再適用しないと黙って元の LR カードルへ
    # 戻る。削減回数は eval_history.json の lr_reductions に記録済み。判定状態は rank 0 のみが
    # 履歴を持つので、他ランクへ broadcast で揃える（LR はランク間で必ず同一でなければならない）
    global _lr_reduce_count, _lr_reduced_step
    _lr_reduce_count = 0
    _lr_reduced_step = None
    if rank == 0:
        prior = max((int(e.get("lr_reductions", 0)) for e in _eval_history), default=0)
        reduced_step = None
        if prior > 0:
            before = [
                e["step"]
                for e in _eval_history
                if int(e.get("lr_reductions", 0)) == prior - 1
            ]
            reduced_step = max(before, default=None)
        lr_state = [prior, reduced_step]
    else:
        lr_state = [None, None]
    if n_gpus > 1:
        dist.broadcast_object_list(lr_state, src=0)
    _lr_reduce_count, _lr_reduced_step = lr_state
    if _lr_reduce_count > 0:
        restore_factor = hps.train.early_stopping_reduce_lr_factor**_lr_reduce_count
        # base_lrs・現在の lr の両方へ factor^prior を掛ける。スケジューラ構築が
        # optimizer 状態の削減済み lr を「削減前のカードル」で上書きしているため、
        # ここで両方揃えないと最初の scheduler.step() まで削減が効かない
        # （_reduce_lr の docstring 参照。torch 2.14 で挙動を実測確認済み）
        _reduce_lr(
            [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
            restore_factor,
        )
        logger.info(
            f"eval_history から過去の学習率削減 {_lr_reduce_count} 回分"
            f"（LR ×{restore_factor:.4g}）を復元しました"
        )

    # 学習中に例外で抜けてもプロセスグループと進捗バーを確実に閉じる
    try:
        for epoch in range(epoch_str, hps.train.epochs + 1):
            # エポックごとにバッチ構成・順序を変える（self.epoch は set_epoch で初めて反映される）
            train_sampler.set_epoch(epoch)
            if rank == 0:
                should_stop = train_and_evaluate(
                    rank,
                    local_rank,
                    epoch,
                    hps,
                    [net_g, net_d, net_dur_disc, net_wd, wl],
                    [optim_g, optim_d, optim_dur_disc, optim_wd],
                    [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
                    [train_loader, eval_loader],
                    logger,
                    [writer, writer_eval],
                    pbar,
                    initial_step,
                )
            else:
                should_stop = train_and_evaluate(
                    rank,
                    local_rank,
                    epoch,
                    hps,
                    [net_g, net_d, net_dur_disc, net_wd, wl],
                    [optim_g, optim_d, optim_dur_disc, optim_wd],
                    [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
                    [train_loader, None],
                    None,
                    None,
                    pbar,
                    initial_step,
                )
            scheduler_g.step()
            scheduler_d.step()
            if net_dur_disc is not None:
                scheduler_dur_disc.step()
            if net_wd is not None:
                scheduler_wd.step()
            if should_stop:
                # 早期停止発動。チェックポイント・safetensors は評価ステップで保存済み。
                # 残る .pth は停止step（再開用）なので、推論に使うべき best を再掲する
                best = _best_eval_entry(_eval_history, hps, initial_step)
                logger.info(
                    f"====> Early stopping: epoch {epoch} / global_step {global_step} で学習を停止します"
                    + ("" if best is None else _best_summary(hps, best[0], best[1]))
                )
                break
            if epoch == hps.train.epochs:
                # 最終モデルの保存（保存は rank 0 のみ。他ランクが同時に同じ .pth へ
                # 書くと競合して壊れる。評価ステップの保存は元々 rank 0 ガード付き）
                if rank == 0:
                    assert optim_g is not None
                    utils.checkpoints.save_checkpoint(
                        net_g,
                        optim_g,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(model_dir, f"G_{global_step}.pth"),
                    )
                    assert optim_d is not None
                    utils.checkpoints.save_checkpoint(
                        net_d,
                        optim_d,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(model_dir, f"D_{global_step}.pth"),
                    )
                    if net_dur_disc is not None:
                        assert optim_dur_disc is not None
                        utils.checkpoints.save_checkpoint(
                            net_dur_disc,
                            optim_dur_disc,
                            hps.train.learning_rate,
                            epoch,
                            os.path.join(model_dir, f"DUR_{global_step}.pth"),
                        )
                    if net_wd is not None:
                        assert optim_wd is not None
                        utils.checkpoints.save_checkpoint(
                            net_wd,
                            optim_wd,
                            hps.train.learning_rate,
                            epoch,
                            os.path.join(model_dir, f"WD_{global_step}.pth"),
                        )
                    final_safetensors = os.path.join(
                        out_dir,
                        f"{model_name}_e{epoch}_s{global_step}.safetensors",
                    )
                    utils.safetensors.save_safetensors(
                        net_g, epoch, final_safetensors, for_infer=True
                    )
                    if _ema_state is not None:
                        _save_ema_safetensors(
                            net_g, epoch, _ema_sibling(final_safetensors)
                        )
                    # 最終保存でも best 記録を最新化してから、keep_safetensors 上限で
                    # 古い eval 資産を整理する（best の step は必ず残す）
                    best = _best_eval_entry(_eval_history, hps, initial_step)
                    if best is not None:
                        _save_best_checkpoint(hps, best[0], best[1])
                    _clean_safetensors(
                        out_dir,
                        hps.train.keep_safetensors,
                        best_step=None if best is None else best[0]["step"],
                    )
    finally:
        if pbar is not None:
            pbar.close()
        # TensorBoard のイベントファイルは close するまでバッファに残いうるため、
        # 正常終了・例外終了を問わず閉じる（学習ログの末尾欠落防止）
        if writer is not None:
            writer.close()
            writer_eval.close()
        dist.destroy_process_group()


def train_and_evaluate(
    rank,
    local_rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    loaders,
    logger,
    writers,
    pbar: tqdm | None,
    initial_step: int,
):
    net_g, net_d, net_dur_disc, net_wd, wl = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    global global_step, _nan_skip_count

    # 早期停止の判定結果（rank 0 が評価ステップで設定し、他ランクへブロードキャストで同期する）。
    # lr_reduce_requested も同じ経路で同期する（学習率は全ランクで必ず一致させる）
    early_stop_requested = False
    lr_reduce_requested = False

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    if net_wd is not None:
        net_wd.train()
    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speakers,
        tone,
        language,
        bert,
        style_vec,
    ) in enumerate(train_loader):
        # 勾配累積（accumulation_steps > 1 のとき有効）。更新マイクロバッチ以外は
        # DDP の勾配 all_reduce を no_sync で省略し、is_opt_step の backward だけで
        # 同期する（中間ステップの集合通信コストを払わない）。既定 1 では常に
        # is_opt_step=True で従来動作と完全に一致する
        accumulation_steps = hps.train.accumulation_steps
        is_opt_step = (batch_idx + 1) % accumulation_steps == 0 or (
            batch_idx + 1 == len(train_loader)  # エポック末尾の端数も必ず更新する
        )
        if batch_idx % accumulation_steps == 0:
            # 累積ウィンドウの先頭で勾配をリセット（ウィンドウ内の backward が合計される）。
            # set_to_none=False は必須: DDP を gradient_as_bucket_view=True でラップしているため、
            # 勾配テンソルを None にして作り直すとバケットビューの対応が壊れる
            # （torch 側も毎イテレーションでビューが破棄され節約効果が消えると明記）。
            # ゼロ埋めならバケット実体そのものをゼロ化するのでビューは有効なまま
            optim_g.zero_grad(set_to_none=False)
            optim_d.zero_grad(set_to_none=False)
            if optim_dur_disc is not None:
                optim_dur_disc.zero_grad(set_to_none=False)
            if optim_wd is not None:
                optim_wd.zero_grad(set_to_none=False)
        with ExitStack() as stack:
            if not is_opt_step:
                stack.enter_context(net_g.no_sync())
                stack.enter_context(net_d.no_sync())
                if net_dur_disc is not None:
                    stack.enter_context(net_dur_disc.no_sync())
                if net_wd is not None:
                    stack.enter_context(net_wd.no_sync())
            if net_g.module.use_noise_scaled_mas:
                current_mas_noise_scale = (
                    net_g.module.mas_noise_scale_initial
                    - net_g.module.noise_scale_delta * global_step
                )
                net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
            x, x_lengths = x.cuda(local_rank, non_blocking=True), x_lengths.cuda(
                local_rank, non_blocking=True
            )
            spec, spec_lengths = spec.cuda(
                local_rank, non_blocking=True
            ), spec_lengths.cuda(local_rank, non_blocking=True)
            y, y_lengths = y.cuda(local_rank, non_blocking=True), y_lengths.cuda(
                local_rank, non_blocking=True
            )
            speakers = speakers.cuda(local_rank, non_blocking=True)
            tone = tone.cuda(local_rank, non_blocking=True)
            language = language.cuda(local_rank, non_blocking=True)
            bert = bert.cuda(local_rank, non_blocking=True)
            style_vec = style_vec.cuda(local_rank, non_blocking=True)

            # 勾配ノルムはログ出力ステップ（rank 0 のみ）だけ計算する（clip_grad_value_(..., None) はクリップせず毎回全パラメータを走査するだけ）
            is_log_step = (
                rank == 0
                and global_step % hps.train.log_interval == 0
                and not hps.speedup
            )

            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),  # , logw_sdp),
                g,
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
                style_vec,
            )
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator（3つの判別器損失を先に計算し、NaN ガードの判定を
            # 1回の集合通信で全ランクへ揃えてから backward / step する。
            # ランク間で backward の実行・不実行が分かれると DDP の集合通信が
            # 不一致になってハングするため、判定は必ず all_reduce 経由で共有する）
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw.detach(),
                    g.detach(),
                )
                # TODO: I think need to mean using the mask, but for now, just mean all
                (
                    loss_dur_disc,
                    losses_dur_disc_r,
                    losses_dur_disc_g,
                ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                loss_dur_disc_all = loss_dur_disc
            if net_wd is not None:
                # logger.debug(f"y.shape: {y.shape}, y_hat.shape: {y_hat.shape}")
                # shape: (batch, 1, time)
                # WavLM の forward は 1ステップで実音声1回（no_grad）・生成音声1回（勾配あり）に統合する。
                # 旧実装は forward()/generator()/discriminator() がそれぞれ独立に forward していて倍計算だった。
                y_embeddings = wl.encode(y.detach().squeeze(1), requires_grad=False)
                y_rec_embeddings = wl.encode(
                    y_hat.detach().squeeze(1), requires_grad=False
                )
                loss_slm = wl.discriminator(y_embeddings, y_rec_embeddings).mean()

            disc_losses = [loss_disc_all]
            if net_dur_disc is not None:
                disc_losses.append(loss_dur_disc_all)
            if net_wd is not None:
                disc_losses.append(loss_slm)
            disc_ok = _all_ranks_finite(disc_losses, y.device)
            if not disc_ok:
                # 非有限損失は backward しても勾配がゴミなので、グラフを保ったまま
                # 値だけゼロにしてから backward する（DDP Reducer の整合性維持）
                loss_disc_all = _zero_loss(loss_disc_all)
                if net_dur_disc is not None:
                    loss_dur_disc_all = _zero_loss(loss_dur_disc_all)
                if net_wd is not None:
                    loss_slm = _zero_loss(loss_slm)

            if net_dur_disc is not None:
                loss_dur_disc_all.backward()
                # 勾配ノルムはログに出力しないので計算しない（旧実装は毎回全パラメータを走査するだけだった）
                if is_opt_step and disc_ok:
                    optim_dur_disc.step()
            if net_wd is not None:
                loss_slm.backward()
                if is_log_step:
                    grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None)
                if is_opt_step and disc_ok:
                    optim_wd.step()

            loss_disc_all.backward()
            if is_log_step:
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            if is_opt_step and disc_ok:
                optim_d.step()
            if not disc_ok:
                # ゴミ勾配を次の累積ウィンドウへ引きずらない（set_to_none=False は上と同じ理由）
                optim_d.zero_grad(set_to_none=False)
                if optim_dur_disc is not None:
                    optim_dur_disc.zero_grad(set_to_none=False)
                if optim_wd is not None:
                    optim_wd.zero_grad(set_to_none=False)

            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                _, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw, g)
            if net_wd is not None:
                # 実音声側（y）の WavLM forward は判別器ステップで共有した no_grad 結果を使い、
                # 生成音声側（y_hat）は勾配を流すため勾配ありで再度 forward する（no_grad 版とは別計算）
                y_rec_embeddings_grad = wl.encode(y_hat.squeeze(1), requires_grad=True)
                loss_lm = wl(y_embeddings, y_rec_embeddings_grad).mean()
                loss_lm_gen = wl.generator(y_rec_embeddings_grad)
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)

            # 敵対的損失のウォームアップ（adversarial_warmup_steps > 0 のとき、今回の
            # 学習開始 step から線形に 0→1 でスケール）。判別器が未熟な初期段階で
            # 生成器が無茶な敵対勾配に引っ張られるのを防ぐ。既定 0 は無効（従来動作）
            adv_scale = 1.0
            if hps.train.adversarial_warmup_steps > 0:
                adv_scale = min(
                    1.0,
                    (global_step - initial_step + 1)
                    / hps.train.adversarial_warmup_steps,
                )

            loss_gen_all = (
                (loss_gen + loss_fm) * adv_scale + loss_mel + loss_dur + loss_kl
            )
            if net_dur_disc is not None:
                loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                loss_gen_all += loss_dur_gen * adv_scale
            # SLM(WavLM) の敵対学習損失は duration 判別器の有無と無関係に必ず生成器へ加算する
            # （JP-Extra 既定は use_duration_discriminator: false なので、内側にネストすると常に未加算になっていた）
            if net_wd is not None:
                loss_gen_all += (loss_lm + loss_lm_gen) * adv_scale

            # 生成器側も NaN ガード（判定は全ランク同期）。disc 側と同じ理由で
            # backward はグラフごとゼロ化して通す
            gen_ok = _all_ranks_finite([loss_gen_all], y.device)
            if not gen_ok:
                loss_gen_all = _zero_loss(loss_gen_all)
            loss_gen_all.backward()
            # 実際のクリップは更新ステップのみ（累積中は勾配が途中経過なので
            # 中間ステップでクリップすると合計勾配が過小になる）。戻り値はログ用に使う
            grad_norm_g = None
            if is_opt_step:
                grad_norm_g = torch.nn.utils.clip_grad_norm_(
                    parameters=net_g.parameters(), max_norm=500
                )
            if is_opt_step and gen_ok:
                optim_g.step()
                _ema_update(net_g, hps.train.ema_decay)
            if not gen_ok:
                optim_g.zero_grad(set_to_none=False)

            # 非有限損失（NaN/inf）ガード。判定は全ランク同期済みなので、
            # スキップも停止も全ランクで同時に起きる（ハングしない）
            if disc_ok and gen_ok:
                _nan_skip_count = 0
            else:
                _nan_skip_count += 1
                if rank == 0:
                    logger.warning(
                        f"step {global_step}: 非有限損失（NaN/inf）を検出したため optimizer の更新を"
                        f"スキップしました（連続 {_nan_skip_count} 回目）"
                    )
                if (
                    hps.train.nan_guard_max_skips > 0
                    and _nan_skip_count >= hps.train.nan_guard_max_skips
                ):
                    raise RuntimeError(
                        f"非有限損失が {_nan_skip_count} ステップ連続で発生したため学習を停止します。"
                        "学習率の高すぎ・データの破損を点検してください"
                        "（nan_guard_max_skips: 0 で無効化可能）"
                    )

            if rank == 0:
                if is_log_step:
                    lr = optim_g.param_groups[0]["lr"]
                    # logger.info(
                    #     "Train Epoch: {} [{:.0f}%]".format(
                    #         epoch, 100.0 * batch_idx / len(train_loader)
                    #     )
                    # )
                    # logger.info([loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl] + [global_step, lr])

                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc_all,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                            "loss/g/dur": loss_dur,
                            "loss/g/kl": loss_kl,
                        }
                    )
                    scalar_dict.update(
                        {f"loss/g/{i}": v for i, v in enumerate(losses_gen)}
                    )
                    scalar_dict.update(
                        {f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)}
                    )
                    scalar_dict.update(
                        {f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)}
                    )

                    if net_dur_disc is not None:
                        scalar_dict.update({"loss/dur_disc/total": loss_dur_disc_all})

                        scalar_dict.update(
                            {
                                f"loss/dur_disc_g/{i}": v
                                for i, v in enumerate(losses_dur_disc_g)
                            }
                        )
                        scalar_dict.update(
                            {
                                f"loss/dur_disc_r/{i}": v
                                for i, v in enumerate(losses_dur_disc_r)
                            }
                        )

                        scalar_dict.update({"loss/g/dur_gen": loss_dur_gen})
                        scalar_dict.update(
                            {
                                f"loss/g/dur_gen_{i}": v
                                for i, v in enumerate(losses_dur_gen)
                            }
                        )

                    if net_wd is not None:
                        scalar_dict.update(
                            {
                                "loss/wd/total": loss_slm,
                                "grad_norm_wd": grad_norm_wd,
                                "loss/g/lm": loss_lm,
                                "loss/g/lm_gen": loss_lm_gen,
                            }
                        )
                    # 勾配累積の中間マイクロバッチ（is_opt_step でないステップ）では
                    # 勾配クリップしないため grad_norm_g は None。TensorBoard に
                    # None を渡さないよう除外する
                    scalar_dict = {
                        k: v for k, v in scalar_dict.items() if v is not None
                    }
                    # 画像ログ（スペクトログラム・アライメントの TensorBoard 出力）は
                    # matplotlib の削除済み API に依存していたため撤去した（utils.plot_* も削除済み）
                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        scalars=scalar_dict,
                    )

                if (
                    global_step % hps.train.eval_interval == 0
                    and global_step != 0
                    and initial_step != global_step
                ):
                    if not hps.speedup:
                        evaluate(hps, net_g, eval_loader, writer_eval)
                        val_metrics = validate_loss(hps, net_g, eval_loader, writer, wl)
                        if val_metrics:
                            _eval_history.append(
                                {
                                    "step": global_step,
                                    "epoch": epoch,
                                    "val": val_metrics,
                                    # 測定方法のバージョン（v2=全文復号）。旧バージョンの履歴は
                                    # 読み込み時に早期停止判定から除外される
                                    "metric_version": _VAL_METRIC_VERSION,
                                    # 検証セットの識別子。val.list を変更したら旧履歴の
                                    # 損失は比較不能になるので読み込み時に除外する
                                    "val_hash": _val_hash,
                                    # 現在の学習率（reduce_lr の効果を目視確認するため）と
                                    # 時点の削減回数。再開時の LR 削減復元はここから行う（run() 側）
                                    "lr": optim_g.param_groups[0]["lr"],
                                    "lr_reductions": _lr_reduce_count,
                                }
                            )
                            _save_eval_history(hps)
                            if hps.train.early_stopping_enabled:
                                # since_step: reduce_lr 済みならその削減以降の評価だけで判定する
                                # （削減後に再改善した学習を、削減前の古い横ばいで即停止させない）
                                stop, reason = _early_stop_check(
                                    _eval_history,
                                    hps,
                                    initial_step,
                                    since_step=_lr_reduced_step,
                                )
                                if stop:
                                    if (
                                        hps.train.early_stopping_action == "reduce_lr"
                                        and _lr_reduce_count
                                        < hps.train.early_stopping_reduce_lr_times
                                    ):
                                        # まだ停止せず学習率を半減して続行。実際の削減は
                                        # broadcast 後に全ランクへ適用する（LR はランク間同期が必須）。
                                        # 削減評価の履歴エントリには削減「前」の回数が残る
                                        lr_reduce_requested = True
                                        logger.info(
                                            f"停止条件に到達したが early_stopping_action=reduce_lr のため"
                                            f"学習率を ×{hps.train.early_stopping_reduce_lr_factor} に削減して続行します"
                                            f"（{_lr_reduce_count + 1}/{hps.train.early_stopping_reduce_lr_times} 回目、"
                                            f"step {global_step}: {reason}）"
                                        )
                                    else:
                                        early_stop_requested = True
                                        best = _best_eval_entry(
                                            _eval_history, hps, initial_step
                                        )
                                        logger.info(
                                            f"Early stopping condition met at step {global_step}: {reason}"
                                            + (
                                                ""
                                                if best is None
                                                else _best_summary(
                                                    hps, best[0], best[1]
                                                )
                                            )
                                        )
                    utils.checkpoints.save_checkpoint(
                        net_g,
                        optim_g,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, f"G_{global_step}.pth"),
                    )
                    utils.checkpoints.save_checkpoint(
                        net_d,
                        optim_d,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, f"D_{global_step}.pth"),
                    )
                    if net_dur_disc is not None:
                        utils.checkpoints.save_checkpoint(
                            net_dur_disc,
                            optim_dur_disc,
                            hps.train.learning_rate,
                            epoch,
                            os.path.join(hps.model_dir, f"DUR_{global_step}.pth"),
                        )
                    if net_wd is not None:
                        utils.checkpoints.save_checkpoint(
                            net_wd,
                            optim_wd,
                            hps.train.learning_rate,
                            epoch,
                            os.path.join(hps.model_dir, f"WD_{global_step}.pth"),
                        )
                    keep_ckpts = hps.train.keep_ckpts
                    if keep_ckpts > 0:
                        utils.checkpoints.clean_checkpoints(
                            model_dir_path=hps.model_dir,
                            n_ckpts_to_keep=keep_ckpts,
                            sort_by_time=True,
                        )
                    # 推論用 safetensors を model_assets/{model_name} へ保存
                    saved_safetensors = os.path.join(
                        hps.out_dir,
                        f"{hps.model_name}_e{epoch}_s{global_step}.safetensors",
                    )
                    utils.safetensors.save_safetensors(
                        net_g, epoch, saved_safetensors, for_infer=True
                    )
                    if _ema_state is not None:
                        _save_ema_safetensors(
                            net_g, epoch, _ema_sibling(saved_safetensors)
                        )
                    # 監視指標上の best を毎回更新して記録する。停止を待たずとも、どの
                    # .safetensors を推論に使うべきかが常に参照できる（手動中断でも失われない）。
                    # safetensors 保存「後」に置くことで best が現 step でも実在する状態になる
                    best = _best_eval_entry(_eval_history, hps, initial_step)
                    if best is not None:
                        _save_best_checkpoint(hps, best[0], best[1])
                    # keep_safetensors 上限で古い eval 資産を整理（best の step は必ず残す）
                    _clean_safetensors(
                        hps.out_dir,
                        hps.train.keep_safetensors,
                        best_step=None if best is None else best[0]["step"],
                    )

            # 停止・学習率削減の判定を全ランクで同期する。ブロードキャスト条件は評価ブロックの
            # 条件と同一にしておく（global_step は全ランク同一値なので同じ反復で合流できる）。
            # nccl は全ランクが同じステップで集合通信を抜けなければハングするため
            if (
                global_step % hps.train.eval_interval == 0
                and global_step != 0
                and initial_step != global_step
                and dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            ):
                obj = [early_stop_requested, lr_reduce_requested]
                dist.broadcast_object_list(obj, src=0)
                early_stop_requested, lr_reduce_requested = obj[0], obj[1]
            if lr_reduce_requested:
                # 判定は rank 0 だけが行うが、削減自体は全ランクで同一に適用する
                # （単一プロセスでは broadcast を経ずに rank 0 のフラグがそのまま届く）
                _apply_lr_reduction(hps, schedulers)
                lr_reduce_requested = False
            if early_stop_requested:
                # global_step を増やす前に抜ける（評価ステップで保存済みのチェックポイントと一致させる）
                break

            global_step += 1
            if pbar is not None:
                pct = 100.0 * batch_idx / len(train_loader)
                if hps.train.early_stopping_enabled:
                    # 収束で打ち切り上限より前に止まりうるので、上限は接頭辞を付け、
                    # 早期停止の判定単位である絶対stepも常時表示する
                    desc = (
                        f"Epoch {epoch}({pct:.0f}%)/上限{hps.train.epochs}"
                        f" step {global_step}"
                    )
                else:
                    desc = f"Epoch {epoch}({pct:.0f}%)/{hps.train.epochs}"
                pbar.set_description(desc)
                pbar.update()

    # caching allocator が空きを再利用するため、エポックごとの empty_cache は不要（削除）
    gc.collect()
    if pbar is None and rank == 0:
        logger.info(f"====> Epoch: {epoch}, step: {global_step}")
    return early_stop_requested


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    audio_dict = {}
    logger.info("Evaluating ...")
    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
            bert,
            style_vec,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            style_vec = style_vec.cuda()
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    style_vec,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
                # 画像ログ（スペクトログラムの TensorBoard 出力）は matplotlib の削除済み API に
                # 依存していたため撤去した（utils.plot_* も削除済み。音声ログのみ残す）
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


def validate_loss(hps, net_g, eval_loader, writer, wl=None):
    """検証データ（val.list）で非敵対損失（mel/dur/kl + 必要なら lm）を**全文**で計算し TensorBoard の val/* に書き込む。

    早期停止の監視指標。学習ステップと同じ forward（MAS アライメント）を使うが、評価間で値を
    比較できるようシードを固定し MAS ノイズを 0 にする。mel・kl・lm は復号音声の全文で測る
    （学習のような segment_size スライス1点にすると測定音が少なく、評価ノイズが収束幅より
    大きくなって best 判定がコイントスになるため。実測値は docs/DECISIONS.md §9.2 参照）。
    検証セットの件数自体は前処理の --val_ratio（既定5%）で決まる。
    敵対損失（fm/gen/wd 系）は判別器の状態に依存して単調でなくなるため監視に使わない。
    lm（WavLM feature matching 損失）は敵対的ではない「実音声と生成音声の WavLM 特徴の距離」で、
    知覚的な近さを mel のフレーム対応とは別軸から測る指標になる（wl を渡したときだけ計算）。
    DDP ラッパを介さず module を直接呼ぶ（rank 0 だけの forward で broadcast_buffers による
    集合通信が発生すると他ランクが待っておらずハングするため）。

    測定方法は eval_history.json の metric_version で識別する（v1=スライス1点、v2=全文。
    絶対値もノイズ量も別物なので、旧履歴と混ぜると best 判定が壊れる。run() の読み込みで除外）。

    Args:
        hps: ハイパーパラメータ
        net_g: 生成器（DDP ラップ可）
        eval_loader: val.list のデータローダ（batch_size=1）
        writer: TensorBoard writer
        wl: WavLMLoss（None で lm 計算をスキップ。use_wavlm_discriminator: false の設定相当）

    Returns:
        dict: 平均検証損失 {"mel":..,"dur":..,"kl":..,"total":..(,"lm":..)}。val.list が空なら {}
    """
    module = getattr(net_g, "module", net_g)
    if eval_loader is None or len(eval_loader) == 0:
        return {}

    # 学習ループの乱数系列を乱さないよう RNG 状態を退避し、決定論的な検証損失を作る
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    saved_mas_noise = getattr(module, "current_mas_noise_scale", None)

    totals = {"mel": 0.0, "dur": 0.0, "kl": 0.0}
    if wl is not None:
        totals["lm"] = 0.0
    n_batches = 0
    module.eval()
    with torch.no_grad():
        torch.manual_seed(hps.train.seed)
        if (
            getattr(module, "use_noise_scaled_mas", False)
            and saved_mas_noise is not None
        ):
            module.current_mas_noise_scale = 0.0
        for (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
            bert,
            style_vec,
        ) in eval_loader:
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y = (
                y.cuda()
            )  # lm 計算で正解音声のスライスが必要（mel/dur/kl は spec 側だけ使う）
            speakers = speakers.cuda()
            tone = tone.cuda()
            language = language.cuda()
            bert = bert.cuda()
            style_vec = style_vec.cuda()
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                g,
            ) = module(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
                style_vec,
            )
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            # 検証は全文で測る（v2）。学習は segment_size（既定 0.37秒）のランダムスライスだが、
            # 検証もスライス1点にすると評価ノイズが収束幅より大きくなることを実測で確認した
            # （docs/DECISIONS.md §9.2）。全文復号は推論時の挙動そのものでもあり、スライス損失より
            # 合成品質との対応が良い。forward は z_slice（スライス）しか復号しないので、
            # 全文の後標本 z を復号し直す。no_grad・batch_size=1 なので長文でも VRAM は問題にならない。
            # ※ forward が返す y_hat / ids_slice（スライス側）は使わない
            y_hat_full = module.dec(z, g=g)
            y_hat_mel = mel_spectrogram_torch(
                y_hat_full.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            # z のフレーム数は spec のそれと一致するので通常は同形だが、短い側へ揃えておく
            # （mel_spectrogram_torch は reflect パディング込みのフレーム数、正解側は ceil 枠）
            n_mel_frames = min(mel.size(2), y_hat_mel.size(2))
            totals["mel"] += (
                F.l1_loss(
                    y_hat_mel[:, :, :n_mel_frames], mel[:, :, :n_mel_frames]
                ).item()
                * hps.train.c_mel
            )
            totals["kl"] += (
                kl_loss(z_p, logs_q, m_p, logs_p, z_mask).item() * hps.train.c_kl
            )
            totals["dur"] += torch.sum(l_length.float()).item()
            if wl is not None:
                # lm も全文で測る（kl・dur は forward が全文で返す値なので元から全文基準）。
                # wl.forward は層ごとに |er - eg| を取るため、実音声と生成音声のフレーム数が
                # 1つでもずれるとブロードキャストに失敗する。生成音声は末尾 hop 未満の端数分
                # 長くなることがあるので、両者を短い側のサンプル数へ揃えて WavLM へ入れる
                n_wav = min(y.size(2), y_hat_full.size(2))
                er = wl.encode(y[:, :, :n_wav].squeeze(1), requires_grad=False)
                eg = wl.encode(y_hat_full[:, :, :n_wav].squeeze(1), requires_grad=False)
                totals["lm"] += wl(er, eg).item()
            n_batches += 1
    module.train()

    if rng_cuda is not None:
        torch.cuda.set_rng_state(rng_cuda)
    torch.set_rng_state(rng_cpu)
    if saved_mas_noise is not None:
        module.current_mas_noise_scale = saved_mas_noise

    avg = {k: v / n_batches for k, v in totals.items()}
    # total は mel + dur + kl の合計（lm は含めない）。lm を total に入れると、lm 計算が
    # 有効な実行と無効な実行（use_wavlm_discriminator: false 等）で total の意味が変わり、
    # eval_history をまたいだ早期停止判定が壊れるため、lm は単独で監視する（monitor: "lm"）。
    # mel は c_mel（既定45）倍で桁が大きく、dur は音素数で正規化済みの MSE（batch_size=1 の
    # 評価では1発話あたり値）なので、total は実質 mel 支配になる。dur 変化だけを追いたい場合は
    # monitor: "dur" を使うこと
    avg["total"] = avg["mel"] + avg["dur"] + avg["kl"]
    utils.summarize(
        writer=writer,
        global_step=global_step,
        scalars={f"val/{k}": v for k, v in avg.items()},
    )
    logger.info(
        f"Validation loss at step {global_step}: "
        + ", ".join(f"{k}={v:.4f}" for k, v in avg.items())
    )
    return avg


def _save_eval_history(hps) -> None:
    """検証損失履歴を models/eval_history.json に保存する（再開時も早期停止判定へ引き継ぐ）"""
    path = Path(hps.model_dir) / "eval_history.json"
    try:
        path.write_text(
            json.dumps(_eval_history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as e:
        logger.warning(f"eval_history.json の保存に失敗しました: {e}")


def _best_eval_entry(
    history: list[dict], hps, initial_step: int
) -> tuple[dict, str] | None:
    """監視指標で見た最良の評価を (履歴entry, 監視キー) で返す。評価が無ければ None。

    判定窓は _early_stop_check と揃える（min_steps 超の評価のみ）。窓を揃えないと
    「停止理由が指す best」と「残す best」が食い違うため。min_steps 未達で窓が空に
    なる間は全履歴から選ぶ（best 自体は早期停止の有無に関係なく有益なので出す）。

    best 選定は early_stopping_best_smooth（既定5）の窓内中央値で行う。検証損失は
    評価ノイズで数%揺れるため、単発の最小値だと「偶発的に低く出た評価」を best に
    選ぶくじ引きになる。中央値スムージングで近傍ごと安定して低い step を選び、
    同スコアのときは生値・step 昇順（早い方）で決める。1 で従来動作（単発最小値）。
    min_steps の負値（-1=自動。run() で解決済みなため通常届かない）は 0 と同じく全履歴対象
    """
    key = hps.train.early_stopping_monitor
    if key not in ("mel", "dur", "kl", "lm", "total"):
        key = "mel"
    judged = history
    min_steps = hps.train.early_stopping_min_steps
    if hps.train.early_stopping_enabled and min_steps > 0:
        threshold = initial_step + min_steps
        filtered = [entry for entry in history if entry["step"] > threshold]
        if filtered:
            judged = filtered
    scored = [entry for entry in judged if key in entry.get("val", {})]
    if not scored:
        return None
    window = max(1, hps.train.early_stopping_best_smooth)
    if window == 1:
        return min(scored, key=lambda entry: entry["val"][key]), key
    vals = [entry["val"][key] for entry in scored]
    half = window // 2

    def smoothed(i: int) -> float:
        # 自身を中心に窓を取り、端はクリップ（2点の中央値は平均になり対称に効く）
        return statistics.median(vals[max(0, i - half) : min(len(vals), i + half + 1)])

    best_i = min(
        range(len(scored)),
        key=lambda i: (smoothed(i), vals[i], scored[i]["step"]),
    )
    return scored[best_i], key


def _best_summary(hps, entry: dict, key: str) -> str:
    """best 評価を人間可読にする（早期停止ログで「どのファイルを使うべきか」を明示するため）"""
    step = entry["step"]
    epoch = entry.get("epoch")
    name = f"{hps.model_name}_e{epoch}_s{step}.safetensors"
    if hps.train.best_use_ema:
        # 推論用の1点は EMA 重みのコピー（best_use_ema 既定 true）。検証損失を実測して
        # best step を選んだ通常重み側は _ema 併記ファイルとして残ることを添える
        use = f"best.safetensors（EMA 重み。実体 {_ema_sibling(name)}）"
    else:
        use = name
    return (
        f" / 監視指標 {key} の best は step {step}（{entry['val'][key]:.4f}）。"
        f"残った .pth は停止step（再開用）なので、推論には "
        f"{os.path.join(str(hps.out_dir), use)} を使用してください"
    )


def _ema_sibling(name: str) -> str:
    """通常重みの .safetensors 名から対応する EMA 重み名を作る（sibling ファイル）"""
    suffix = ".safetensors"
    stem = name[: -len(suffix)] if name.endswith(suffix) else name
    return f"{stem}_ema{suffix}"


def _save_best_checkpoint(hps, entry: dict, key: str) -> None:
    """監視指標上の best を models/best_checkpoint.json へ書き出す。

    keep_ckpts=1 では停止stepの .pth だけが残り、best の .pth は clean_checkpoints で
    削除済み。.pth は再開用（optim 状態つき）で巻き戻せるものではないため、推論に使う
    べき資産は別にあることを機械可読に残す。eval ごとの .safetensors は keep_safetensors
    の上限で間引かれるので、best の実体は best.safetensors として1点固定名で保持する
    （3点セットの config.json / style_vectors.npy は 1 ラン内で共通なので、コピーは本体だけでよい）
    """
    step = entry["step"]
    epoch = entry.get("epoch")
    name = f"{hps.model_name}_e{epoch}_s{step}.safetensors"
    ema_name = _ema_sibling(name)
    # best.safetensors にコピーする実体を選ぶ。best step の選定は通常重みで検証損失を
    # 実測して行うが、同一 step 比較では EMA 重みの方が合成品質に優位だった（実測run 実測:
    # 8/8 の候補で UTMOS 優位・平均 +0.27、mel 相関 7/8 優位）。ユーザーが普通に読むのは
    # best.safetensors なので、書き出す1点は EMA 側を既定にする。EMA 実体が無い step
    # （ema_decay=0・EMA 導入前の古い資産）は通常重みへフォールバックする
    use_ema = hps.train.best_use_ema and (Path(hps.out_dir) / ema_name).is_file()
    payload = {
        "model_name": hps.model_name,
        "step": step,
        "epoch": epoch,
        "monitor": key,
        "value": entry["val"][key],
        "safetensors": name,
        "safetensors_ema": ema_name,
        "weights": "ema" if use_ema else "normal",  # best.safetensors に入った実体
        "assets_dir": str(hps.out_dir),
        "note": "早期停止の監視指標で最良の評価。models/ の .pth は停止stepのもので再開用",
    }
    path = Path(hps.model_dir) / "best_checkpoint.json"
    try:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as e:
        logger.warning(f"best_checkpoint.json の保存に失敗しました: {e}")

    # best の .safetensors を "best.safetensors" としても1点コピーする。
    # keep_safetensors で古い eval 資産が削除されるようになり、best が古い step だと
    # 実体が消える事態がありうるため、推論に使うべき1点は名前で固定参照できる実体を持つ。
    # 3点セットの config.json / style_vectors.npy は 1 ラン内で共通なのでコピーは本体だけでよい。
    # コピー元は best_use_ema が既定（EMA 重み）。同じ best のたびに ~120MB コピーし直さないよう、
    # best step が変わるときだけコピーする
    global _best_copied_step
    src = Path(hps.out_dir) / (ema_name if use_ema else name)
    dst = Path(hps.out_dir) / "best.safetensors"
    if _best_copied_step != step and src.is_file():
        # tmp → os.replace のアトミックコピー。直接 shutil.copyfile だと書き込み途中で
        # 失敗（disk 満杯等）したとき best.safetensors が破損した状態で残る。tmp 経由なら
        # 失敗しても既存 best は前回の正常なまま。_best_copied_step を更新しないので
        # 次の評価ステップで自動再試行される（失敗を warning で済ませ学習は止めないが、
        # 推論直結の資産なので error レベルで目立たせる）
        tmp = dst.with_name(f"best.safetensors.tmp{os.getpid()}")
        try:
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            _best_copied_step = step
            logger.info(
                f"best を best.safetensors へコピーしました"
                f" (step {step}, {payload['weights']} 重み)"
            )
        except OSError as e:
            logger.error(
                f"best.safetensors へのコピーに失敗しました（既存 best が古いうえ次の評価まで"
                f"再試行されません disk を点検してください。実体の best 自体は "
                f"{src.name} に残っています）: {e}"
            )
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _resolve_early_stopping_min_steps(hps, steps_per_epoch: int) -> int:
    """early_stopping_min_steps=-1（自動）を concrete な step 数へ解決して hps へ書き戻す。

    自動値は max(1000, 5×1エポックのstep数)。5エポックとしたのは、回帰傾きがプラトー閾値を
    大きく超える急速改善期を越えつつ、過学習が深まりすぎる前の妥協点であるため。下限 1000 は
    極小データセットで判定窓（プラトーに評価5回分必要）が短くなりすぎるのを防ぐ。
    旧既定の固定 3000 をやめた経緯は docs/DECISIONS.md §9.4 参照。
    0 以上はそのまま返す（0=全履歴対象、>0=固定step数）
    """
    value = hps.train.early_stopping_min_steps
    if value < 0:
        value = max(1000, 5 * max(1, steps_per_epoch))
        hps.train.early_stopping_min_steps = value
    return value


def _early_stop_check(
    history: list[dict], hps, initial_step: int, since_step: int | None = None
) -> tuple[bool, str]:
    """検証損失履歴から早期停止すべきかを判定する。

    - 判定開始点: early_stopping_min_steps が今回の学習開始点（再開時は再開step）から
      経過した後の評価のみを判定に使う（それ以前の履歴は best・patience・プラトー窓に
      一切数えない）。新規学習の早すぎる停止と、履歴を引き継いだ再開直後の即停止を防ぐ。
      0 の場合は全履歴を対象にする。負値（-1=自動）は run() で学習開始前に concrete 値へ
      解決済みなので通常届かないが、届いた場合は 0（全履歴対象）として扱う
    - since_step: ここ（early_stopping_action="reduce_lr" の直近削減 step 等）を
      超えた評価だけで判定する。None で無効
    - patience: best 更新（実効 min_delta 以上の改善）が無い評価が early_stopping_patience
      回続いたら停止。実効 min_delta = max(early_stopping_min_delta,
      best × early_stopping_min_delta_rel)。検証損失は評価ノイズで ±数% 揺れるため、
      絶対値 0.001 だけだとノイズの偶発更新で patience が永遠に発火せず epochs 上限まで
      走り続ける（best に対する割合で見るのが ReduceLROnPlateau 等の慣習）
    - プラトー: 直近 early_stopping_plateau_window 評価の最小二乗回帰傾きの絶対値が
      |平均値| × early_stopping_plateau_slope_eps 以下なら停止（0 で無効）
    - 悪化: 同上の窓の回帰傾きが +閾値 を超えるなら停止。悪化が明らかな学習を続けても
      best は保存済みで得られない。|傾き|≦閾値 のみ停止だった旧実装では、ノイズ帯を
      超えて悪化する run はプラトー判定で止まらず、評価回数の少ない小規模データセットでは
      patience も発火しにくく epochs 上限まで走りうる（経緯は docs/DECISIONS.md §9.4）

    Returns:
        tuple[bool, str]: (停止すべきか, 理由)
    """
    min_steps = hps.train.early_stopping_min_steps
    if min_steps > 0:
        threshold = initial_step + min_steps
        history = [entry for entry in history if entry["step"] > threshold]
    if since_step is not None:
        history = [entry for entry in history if entry["step"] > since_step]

    key = hps.train.early_stopping_monitor
    if key not in ("mel", "dur", "kl", "lm", "total"):
        logger.warning(f"early_stopping_monitor='{key}' は不明です。'mel' で判定します")
        key = "mel"
    vals = [entry["val"][key] for entry in history if key in entry.get("val", {})]
    if len(vals) < 2:
        return False, ""

    # patience 判定（実効 min_delta 付きの best 追跡）
    best = float("inf")
    counter = 0
    for v in vals:
        if not math.isfinite(best):
            # 初回は必ず best 登録（inf に相対閾値を引くと nan になり誤って counter が増える）
            best = v
            counter = 0
            continue
        effective_delta = max(
            hps.train.early_stopping_min_delta,
            best * hps.train.early_stopping_min_delta_rel,
        )
        if v < best - effective_delta:
            best = v
            counter = 0
        else:
            counter += 1
    if counter >= hps.train.early_stopping_patience:
        return (
            True,
            f"patience 超過: best({key}={best:.4f}) 更新なし（実効閾値 {max(hps.train.early_stopping_min_delta, best * hps.train.early_stopping_min_delta_rel):.4f} 未満の改善）が {counter} 評価続いています",
        )

    # プラトー・悪化判定（窓内最小二乗回帰の相対傾き）
    window = hps.train.early_stopping_plateau_window
    if window >= 3 and len(vals) >= window:
        ys = vals[-window:]
        x_mean = (window - 1) / 2
        y_mean = sum(ys) / window
        denom = sum((x - x_mean) ** 2 for x in range(window))
        slope = sum((x - x_mean) * (y - y_mean) for x, y in enumerate(ys)) / denom
        threshold = hps.train.early_stopping_plateau_slope_eps * max(abs(y_mean), 1e-8)
        if slope > threshold:
            # ノイズ帯を超えた悪化。改善（slope < -閾値）でも平坦（|slope|≦閾値）でもない
            # のは損失が上昇に転じたと見るべき状態で、best は保存済みなので続行の益がない
            return (
                True,
                f"悪化検出: 直近 {window} 評価の傾き {slope:+.5f}/評価 が悪化閾値 +{threshold:.5f} を超過"
                "（ノイズ帯を超えて悪化。best は保存済みのため続行の益なし）",
            )
        if abs(slope) <= threshold:
            return (
                True,
                f"プラトー検出: 直近 {window} 評価の傾き {slope:+.5f}/評価 が閾値 ±{threshold:.5f} 内",
            )

    return False, ""


if __name__ == "__main__":
    run()
