"""前処理チェーンの Step 1〜2（データセット初期化・raw.zip 展開）。

トップレベルの呼び出し口である `preprocess_all.py`（経由の本パッケージの preprocess_all）から
呼ばれるほか、単体でも `uv run -m preprocess.initialize --model_name <モデル名>` で実行できる。
"""

import hashlib
import json
import shutil
import zipfile

from pydantic import BaseModel, TypeAdapter

from style_bert_vits2.constants import BASE_DIR, DATASET_ROOT, VERSION
from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters

_logger_handler = None

# --set での上書きを禁止するキー。モデル名・データセット派生パス・version は毎回自動決定される
# ため、手動指定と衝突しても自動値が最優先で上書きして紛らわしい（変更不可にはっきり弾く）
_FORCED_CONFIG_KEYS = {
    "model_name",
    "version",
    "data.training_files",
    "data.validation_files",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """dict を再帰的にマージする（override 側を優先）。テンプレートに無いユーザー独自キーも残す"""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_field(path: str):
    """dotted path が HyperParameters の実フィールドを指すか検証し、そのフィールドを返す。

    旧 CLI は設定キーをハードコードしていたため打ち間違いようがなかったが、汎用 --set では
    `--set train.batchsize=4` のような誤ったパスが黙って無視されないよう、スキーマを正として検証する。
    """
    segments = path.split(".")
    model_cls: type[BaseModel] = HyperParameters
    for i, seg in enumerate(segments):
        fields = model_cls.model_fields
        if seg not in fields:
            known = ", ".join(sorted(fields))
            raise ValueError(
                f"--set の設定キー {path!r} は不明です（{seg!r} は {model_cls.__name__} にありません。"
                f"候補: {known}）"
            )
        field = fields[seg]
        if i < len(segments) - 1:
            # 中間セグメントは入れ子の設定ブロック（train/data/model/slm）でなければならない
            if not (
                isinstance(field.annotation, type)
                and issubclass(field.annotation, BaseModel)
            ):
                raise ValueError(
                    f"--set の設定キー {path!r}: {seg!r} の下にさらに設定キーは指定できません"
                )
            model_cls = field.annotation
    return field


def _coerce_override_value(field, raw: str):
    """'--set key=value' の value を JSON として解釈し、フィールド型へ変換・検証する。

    4→int, true→bool, 0.0001→float, [0.8,0.99]→list, null→None。JSON でない語（mel 等）は
    文字列として扱う。resblock のように '1' が数値化してしまう str フィールドは生の文字列へ戻す。
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    if field.annotation is str and not isinstance(parsed, str):
        parsed = raw
    return TypeAdapter(field.annotation).validate_python(parsed)


def apply_config_overrides(config: dict, set_items: list[str]) -> dict:
    """['train.batch_size=4', ...] を config（dict）へ反映する（パス・値は HyperParameters で検証）。

    既存 config.json の手編集より後に適用される＝--set が最優先。未知のパスや型違いの値は
    黙って無視せず ValueError で停止する。モデル名・派生パス・version への変更は拒否する。
    """
    for item in set_items:
        if "=" not in item:
            raise ValueError(
                f"--set は 'dotted.path=value' 形式で指定してください（受け取った値: {item!r}）"
            )
        path, raw = item.split("=", 1)
        path = path.strip()
        if path in _FORCED_CONFIG_KEYS:
            raise ValueError(
                f"{path} はモデル名やデータセットパスから自動設定されるため、--set では変更できません"
            )
        field = _resolve_field(path)
        value = _coerce_override_value(field, raw)
        *parents, leaf = path.split(".")
        node = config
        for seg in parents:
            node = node.setdefault(seg, {})
        node[leaf] = value
    return config


def initialize(
    model_name: str,
    set_overrides: list[str] | None = None,
    reset_models: bool = False,
) -> None:
    """Step 1: Data/<モデル名>/config.json を既定値から生成し、事前学習済みモデルを配置する。

    base は既定値の単一ソースである `HyperParameters.default_config_dict()` から作る（旧
    configs/config_jp_extra.json テンプレートは削除済み）。既存 config.json がある場合は
    **既定値に既存値を重ねたもの**を base にする。以前は毎回テンプレートから作り直していたため、
    learning_rate や early_stopping_* など CLI 対応の手動編集キーが黙って失われていた。

    設定の上書きは汎用 `--set dotted.path=value`（`set_overrides`）に集約した。旧来は
    batch_size / epochs / save_every_steps / log_interval / freeze_* を個別引数として
    preprocess_all → __init__ → initialize と None 透過で引き回していたが、パス・値の検証を
    HyperParameters スキーマに一本化するため廃止した（`apply_config_overrides` 参照）。

    Args:
        set_overrides (list[str] | None): 設定上書きの指定（["train.batch_size=4", ...]）。
            指定されたものだけ、既存値より後の最優先で反映する。None/空なら既存値・既定値を尊重。
        reset_models (bool): True の場合のみ既存の models/ をバックアップして事前学習モデルでリセットする。
            既定では既存チェックポイント（学習途中の G_*.pth 等）を尊重し、config.json の更新だけ行う。

    Raises:
        FileNotFoundError: pretrained/jp_extra が存在しない場合など
        ValueError: set_overrides のパス・値が不正、または変更禁止キーへの変更指定
    """
    global _logger_handler
    dataset_path = DATASET_ROOT / model_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    # 前処理のログをファイルに保存する。ファイル名の日時は loguru の {time} マクロで生成する
    # （Python 側で strftime した文字列を渡すと loguru がファイル名パターンを認識できず、
    # retention（古いログの自動削除）が効かない）。実行のたびに増え続け disk を圧迫しないよう
    # 直近10本だけ保持する
    if _logger_handler is not None:
        logger.remove(_logger_handler)
    _logger_handler = logger.add(
        dataset_path / "preprocess_{time:YYYYMMDD_HHmmss}.log", retention=10
    )

    logger.info(
        f"Step 1: start initialization...\nmodel_name: {model_name}, set: {set_overrides}, reset_models: {reset_models}"
    )

    # 既定値の単一ソース（HyperParameters）から base を作る（旧 config_jp_extra.json 相当）
    config = HyperParameters.default_config_dict()
    # 既存 config.json がある場合は既定値へ既存値を重ねる（新規キーは既定値、
    # 既存キーはユーザー値を維持）。pydantic 側の既定値もあるので欠損キーは起動時に補われる
    existing_path = dataset_path / "config.json"
    if existing_path.is_file():
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            config = _deep_merge(config, existing)
            logger.info("Step 1: 既存 config.json の設定値を維持して更新します")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Step 1: 既存 config.json の読み込みに失敗したため既定値から再生成します: {e}"
            )
    config["model_name"] = model_name
    # config の version はモデルフォーマットの識別子（推論時の互換判定に使う）。
    # パッケージバージョン（pyproject.toml）から生成して手動同期を無くす
    config["version"] = f"{VERSION}-JP-Extra"
    config["data"]["training_files"] = str(dataset_path / "train.list")
    config["data"]["validation_files"] = str(dataset_path / "val.list")
    # --set による明示的な上書き（既存値より後＝最優先。パス・値は HyperParameters で検証）
    if set_overrides:
        apply_config_overrides(config, set_overrides)

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
            # 中身が変わった（または新規）→ 古い派生キャッシュを捨てて上書き展開する。
            # 実ファイル名は音声 X.flac に対して spec=X.spec.pt / mel=X.mel.pt / bert=X.bert.pt
            # （いずれも with_suffix で .flac を置換）、スタイル特徴だけ f"{audiopath}.npy" 規約で
            # X.flac.npy（名前の後ろに追記）。with_suffix の置換系と文字列追記系を混同すると
            # 存在しない X.flac.spec.pt を生成してしまい、古い spec が黙って再利用され続ける
            # （音声リテイク時の誤学習防止のため命名どおりに削除する）
            stale_files = [
                target.with_suffix(".spec.pt"),
                target.with_suffix(".mel.pt"),
                target.with_suffix(".bert.pt"),
                target.parent / (target.name + ".npy"),
            ]
            for stale in stale_files:
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
