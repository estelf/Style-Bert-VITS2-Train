"""前処理チェーンの Step 1〜2（データセット初期化・raw.zip 展開）。

トップレベルの呼び出し口である `preprocess_all.py`（経由の本パッケージの preprocess_all）から
呼ばれるほか、単体でも `uv run -m preprocess.initialize --model_name <モデル名>` で実行できる。
"""

import hashlib
import json
import shutil
import zipfile
from datetime import datetime

from style_bert_vits2.constants import BASE_DIR, DATASET_ROOT, VERSION
from style_bert_vits2.logging import logger

_logger_handler = None


def _deep_merge(base: dict, override: dict) -> dict:
    """dict を再帰的にマージする（override 側を優先）。テンプレートに無いユーザー独自キーも残す"""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def initialize(
    model_name: str,
    batch_size: int | None = None,
    epochs: int | None = None,
    save_every_steps: int | None = None,
    freeze_JP_bert: bool | None = None,
    freeze_style: bool | None = None,
    freeze_decoder: bool | None = None,
    log_interval: int | None = None,
    reset_models: bool = False,
) -> None:
    """Step 1: Data/<モデル名>/config.json をテンプレートから生成し、事前学習済みモデルを配置する。

    既存 config.json がある場合はテンプレートに**既存値を重ねたもの**を base にする。
    以前は毎回テンプレートから作り直していたため、learning_rate や early_stopping_* など
    CLI 引数に対応しないキーの手動編集が黙って失われていた。CLI 引数に対応するキー
    （batch_size / epochs / eval_interval / log_interval / freeze_*）は、引数が**明示された
    ときだけ**上書きする（未指定なら既存値を尊重。freeze フラグは true を明示したときだけ
    立ち上がり、false に戻すには config.json を直接編集する）

    Args:
        reset_models (bool): True の場合のみ既存の models/ をバックアップして事前学習モデルでリセットする。
            既定では既存チェックポイント（学習途中の G_*.pth 等）を尊重し、config.json の更新だけ行う。

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
        f"Step 1: start initialization...\nmodel_name: {model_name}, batch_size: {batch_size}, epochs: {epochs}, save_every_steps: {save_every_steps}, freeze_JP_bert: {freeze_JP_bert}, freeze_style: {freeze_style}, freeze_decoder: {freeze_decoder}, log_interval: {log_interval}, reset_models: {reset_models}"
    )

    # CWD 依存を避けるため BASE_DIR 基準で解決する
    with open(BASE_DIR / "configs/config_jp_extra.json", encoding="utf-8") as f:
        config = json.load(f)
    # 既存 config.json がある場合はテンプレートへ既存値を重ねる（新規キーはテンプレート既定、
    # 既存キーはユーザー値を維持）。pydantic 側の既定値もあるので欠損キーは起動時に補われる
    existing_path = dataset_path / "config.json"
    if existing_path.is_file():
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            config = _deep_merge(config, existing)
            logger.info("Step 1: 既存 config.json の設定値を維持して更新します")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Step 1: 既存 config.json の読み込みに失敗したためテンプレートから再生成します: {e}"
            )
    config["model_name"] = model_name
    # config の version はモデルフォーマットの識別子（推論時の互換判定に使う）。
    # パッケージバージョン（pyproject.toml）から生成して手動同期を無くす
    config["version"] = f"{VERSION}-JP-Extra"
    config["data"]["training_files"] = str(dataset_path / "train.list")
    config["data"]["validation_files"] = str(dataset_path / "val.list")
    # CLI 引数が明示されたときだけ上書きする（None は既存/テンプレート値を尊重）
    if batch_size is not None:
        config["train"]["batch_size"] = batch_size
    if epochs is not None:
        config["train"]["epochs"] = epochs
    if save_every_steps is not None:
        config["train"]["eval_interval"] = save_every_steps
    if log_interval is not None:
        config["train"]["log_interval"] = log_interval
    if freeze_JP_bert is not None:
        config["train"]["freeze_JP_bert"] = freeze_JP_bert
    if freeze_style is not None:
        config["train"]["freeze_style"] = freeze_style
    if freeze_decoder is not None:
        config["train"]["freeze_decoder"] = freeze_decoder

    model_path = dataset_path / "models"
    if model_path.exists() and not reset_models:
        # epochs を変えたいだけで学習がやり直しにならないよう、既存チェックポイントがある場合は config.json だけ更新する
        logger.info(
            f"Step 1: {model_path} already exists, so keeping existing checkpoints and only updating config.json (pass --reset_models to reset)."
        )
    else:
        if model_path.exists():
            logger.warning(
                f"Step 1: {model_path} already exists, so copy it to backup to {model_path}_backup"
            )
            shutil.copytree(
                src=model_path, dst=dataset_path / "models_backup", dirs_exist_ok=True
            )
            shutil.rmtree(model_path)
        shutil.copytree(src=BASE_DIR / "pretrained/jp_extra", dst=model_path)

    with open(dataset_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.success("Step 1: initialization finished.")


def extract_raw(model_name: str) -> None:
    """Step 2: 完成済みデータセットの raw.zip を解凍して wavs/ を作る（スライス・正規化等は分離先のデータセット作成リポジトリ側の責任）。

    ファイルごとにハッシュを記録し、音声の中身が変わった場合のみ .spec.pt / .bert.pt / *.npy 等の派生キャッシュを
    無効化して差し替える（古い派生ファイルが黙って再利用され続けることを防ぐ）。

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
    manifest_path = dataset_path / ".raw_extracted.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    new_manifest = dict(manifest)
    with zipfile.ZipFile(raw_zip) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # path traversal 対策: 絶対パスや ".." を含むエントリは wavs/ の外へ
            # 書き込めるため拒否する（悪意ある zip からのファイル上書きを防ぐ）
            target = (out_dir / info.filename).resolve()
            if not target.is_relative_to(out_dir.resolve()):
                raise ValueError(
                    f"Step 2: raw.zip のエントリ {info.filename!r} は wavs/ の外を指すため展開を中止します"
                )
            data = zf.read(info)
            digest = hashlib.sha256(data).hexdigest()
            if manifest.get(info.filename) == digest and target.exists():
                continue  # 前回と同一。派生キャッシュ（.spec.pt 等）も有効なので何もしない
            # 中身が変わった（または新規）→ 古い派生キャッシュを捨てて上書き展開する
            for suffix in (".spec.pt", ".mel.pt", ".bert.pt", ".npy"):
                stale = target.with_suffix(target.suffix + suffix)
                if stale.exists():
                    logger.info(f"Step 2: deleting stale derived cache {stale.name}")
                    stale.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            new_manifest[info.filename] = digest
    manifest_path.write_text(
        json.dumps(new_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.success("Step 2: extracting raw.zip finished.")
