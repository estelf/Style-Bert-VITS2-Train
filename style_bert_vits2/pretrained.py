"""事前学習済みモデル（日本語BERT・SLM・JP-Extra・話者埋め込み）の取得を一元管理する。

学習にも推論にも必要な下準備なので、パイプライン側（旧 `train/initialize.py`）ではなくライブラリ側に置く。
取得先とファイル一覧は以下のコード内 manifest 1箇所だけ正とする（`pretrained/` 配下は git に含まれず、
全てここでのダウンロード成果物。軽い config 類も例外ではない＝HF が単一ソース）。

- 完備判定はファイル単位の存在＋非空チェック。**揃っていればネットワークを一切叩かない**
  （hf_hub は revision を "main" のまま使ると存在していても HEAD を叩くため、先に自前で弾く）
- 欠けているときだけ `snapshot_download(local_dir=..., allow_patterns=...)` で対象リポジトリの
  manifest 内のファイルのみを取得する（BERT のように .bin と .safetensors を両方持つリポジトリでは、
  「リポジトリの全ファイル」ではなく「必要な形」に絞るのが download 量の観点に合う）
- 取得後も欠けが残る場合（オフライン・gated・名前変更）は `PretrainedModelMissing` で明確に止まる
- `SBV2_SKIP_PRETRAINED_DOWNLOAD=1` で自動取得を抑止、`HF_HUB_OFFLINE=1` は huggingface_hub 側が尊重する

手動でまとめて取得・確認したいとき（環境構築用のコマンドであり、モデルごとの運用手順には出てこない）:

```bash
uv run -m style_bert_vits2.pretrained --dry-run      # 取得予定と総量を確認
uv run -m style_bert_vits2.pretrained                # 全部取得
uv run -m style_bert_vits2.pretrained --only bert    # 推論に必要な分だけ
uv run -m style_bert_vits2.pretrained --check        # 完備状況の確認（欠ければ exit 1）
```
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from style_bert_vits2.constants import PRETRAINED_ROOT, SKIP_PRETRAINED_DOWNLOAD
from style_bert_vits2.logging import logger

__all__ = [
    "PRETRAINED_REPOS",
    "PRETRAINED_ROOT",
    "PretrainedModelMissing",
    "PretrainedRepo",
    "ensure_pretrained",
    "human_size",
    "missing_files",
    "plan_download",
]


class PretrainedModelMissing(RuntimeError):
    """事前学習モデルが未取得のまま、それを必要とする処理に進もうとした"""


@dataclass(frozen=True)
class PretrainedRepo:
    """1つの Hugging Face リポジトリから `pretrained/` 配下へ取るファイル集合"""

    name: str
    repo_id: str
    subdir: str  # PRETRAINED_ROOT からの相対ディレクトリ
    # snapshot_download の allow_patterns 兼「完備判定の required」列表。
    # 軽い config 類も明記してダウンロード対象に含める（git 同梱による二重ソースを作らない）
    files: tuple[str, ...]

    @property
    def local_dir(self) -> Path:
        return PRETRAINED_ROOT / self.subdir

    def missing_files(self) -> list[str]:
        """存在しない（または 0 バイトの）ファイル名を返す。ネットワークは使わない"""
        missing: list[str] = []
        for file in self.files:
            path = self.local_dir / file
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(file)
        return missing


# 取得先の単一ソース。JP-Extra 一本化なので4件固定（multi-language 対応時の JSON は廃止済み）
PRETRAINED_REPOS: dict[str, PretrainedRepo] = {
    # 日本語 BERT（学習の前処理 Step5 と推論の G2P が使う）。
    # 元リポジトリは pytorch_model.bin と model.safetensors を両方持つので、重みは safetensors 1本に寄せる
    "bert": PretrainedRepo(
        name="bert",
        repo_id="ku-nlp/deberta-v2-large-japanese-char-wwm",
        subdir="bert/deberta-v2-large-japanese-char-wwm",
        files=(
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "model.safetensors",
        ),
    ),
    # 学習の WavLM 損失（SLM）用の判别器モデル。推論は読まない
    "slm": PretrainedRepo(
        name="slm",
        repo_id="microsoft/wavlm-base-plus",
        subdir="slm/wavlm-base-plus",
        files=("config.json", "preprocessor_config.json", "pytorch_model.bin"),
    ),
    # 学習の初期チェックポイント（前処理 Step1 が Data/<モデル名>/models/ にコピーする）
    "jp_extra": PretrainedRepo(
        name="jp_extra",
        repo_id="litagin/Style-Bert-VITS2-2.0-base-JP-Extra",
        subdir="jp_extra",
        files=("G_0.safetensors", "D_0.safetensors", "WD_0.safetensors"),
    ),
    # 話者埋め込み＝スタイルベクトル抽出（前処理 Step6・参照音声からのスタイル推論）。
    # pyannote.audio はディレクトリ指定時に AssetFileName.Model（既定 pytorch_model.bin）を読みに行く
    "speaker": PretrainedRepo(
        name="speaker",
        repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        subdir="pyannote/wespeaker-voxceleb-resnet34-LM",
        files=("pytorch_model.bin", "config.yaml"),
    ),
}


def _resolve(targets: Sequence[str] | None) -> list[PretrainedRepo]:
    """targets を manifest エントリへ解決する（None は全件）"""
    if targets is None:
        return list(PRETRAINED_REPOS.values())
    repos: list[PretrainedRepo] = []
    for target in targets:
        if target not in PRETRAINED_REPOS:
            raise ValueError(
                f"unknown pretrained target: {target} "
                f"(options: {', '.join(PRETRAINED_REPOS)})"
            )
        repos.append(PRETRAINED_REPOS[target])
    return repos


def _gaps(repos: Sequence[PretrainedRepo]) -> dict[str, list[str]]:
    return {repo.name: repo.missing_files() for repo in repos if repo.missing_files()}


def missing_files(targets: Sequence[str] | None = None) -> dict[str, list[str]]:
    """完備状況（{モデル名: 欠けているファイル一覧}）。空 dict なら全て揃っている"""
    return _gaps(_resolve(targets))


def _format_gaps(gaps: dict[str, list[str]]) -> str:
    return "; ".join(f"{name}: {', '.join(files)}" for name, files in gaps.items())


def _missing_error(
    gaps: dict[str, list[str]], *, reason: str
) -> PretrainedModelMissing:
    names = ",".join(gaps)
    return PretrainedModelMissing(
        f"事前学習モデルが揃っていないため処理を進められません（{reason}）。"
        f"欠け: {_format_gaps(gaps)}。"
        f"`uv run -m style_bert_vits2.pretrained --only {names}` を実行して取得してください"
        + (
            "（現在 SBV2_SKIP_PRETRAINED_DOWNLOAD=1 が設定されています）"
            if SKIP_PRETRAINED_DOWNLOAD
            else ""
        )
    )


def plan_download(repo: PretrainedRepo) -> list[tuple[str, int, str]]:
    """取得予定をサーバーに確認する（ネットワーク使用）。[(ファイル名, バイト数, 状態)]

    状態: "exists"（local_dir にある）/ "download"（HTTP 取得）/ "copy"（HF キャッシュからコピーで済む）
    / "update"（既存だが remote が新しい）。
    hf_hub の is_cached・will_download は `~/.cache/huggingface` も見るため「local_dir に有無」とは
    別軸で見る必要がある（1.3GB を「取得不要」と誤報しないために切り分ける）
    """
    infos = snapshot_download(
        repo.repo_id,
        local_dir=str(repo.local_dir),
        allow_patterns=list(repo.files),
        dry_run=True,
    )
    missing = set(repo.missing_files())
    plan: list[tuple[str, int, str]] = []
    for info in infos:
        size = int(info.file_size or 0)
        if info.filename in missing:
            status = (
                "copy" if (info.is_cached and not info.will_download) else "download"
            )
        elif info.will_download:
            status = "update"
        else:
            status = "exists"
        plan.append((info.filename, size, status))
    return plan


def human_size(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f}GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f}MB"
    return f"{num_bytes / 1024:.1f}KB"


def _planned_size(plan: list[tuple[str, int, str]]) -> int:
    """exists（そのまま使える）を除いた、実際に準備が必要なバイト数"""
    return sum(size for _, size, status in plan if status != "exists")


def _log_plan(repos: Sequence[PretrainedRepo]) -> None:
    """取得総量を先に警告ログへ出す（長時間のダウンロードが黙り込んだように見えるのを防ぐ）"""
    total = 0
    for repo in repos:
        try:
            plan = plan_download(repo)
        except Exception as e:  # 総量確認の失敗で取得自体を止めない
            logger.info(f"Skip checking the download size of {repo.name}: {e}")
            continue
        total += _planned_size(plan)
    if total > 0:
        logger.warning(
            f"Preparing {human_size(total)} of pretrained models under {PRETRAINED_ROOT} ..."
        )


def ensure_pretrained(
    targets: Sequence[str] | None = None, *, force: bool = False
) -> None:
    """事前学習モデルが揃っていれば即return、欠けていれば対象リポジトリだけから取得する。

    Args:
        targets (Sequence[str] | None): 対象のモデル名（None で全件）。存在しない名前は ValueError
        force (bool): True で既存ファイルを上書き再取得する

    Raises:
        ValueError: 存在しないモデル名が指定された場合
        PretrainedModelMissing: 取得抑止中、または取得後も欠けが残った場合
    """
    repos = _resolve(targets)
    gaps = _gaps(repos)

    if not gaps and not force:
        logger.debug(f"Pretrained models are complete under {PRETRAINED_ROOT}")
        return
    if SKIP_PRETRAINED_DOWNLOAD:
        if gaps:
            raise _missing_error(gaps, reason="自動ダウンロードは無効化されている")
        raise PretrainedModelMissing(
            "SBV2_SKIP_PRETRAINED_DOWNLOAD が有効なため再取得（force）も行いません。"
            f"解除するか、対象を手动で削除してから実行してください（{PRETRAINED_ROOT}）"
        )

    if gaps:
        logger.info(f"Missing pretrained files -> {_format_gaps(gaps)}")
    else:
        logger.info("force: 揃っているファイルも再取得します")
    need = list(repos) if force else [repo for repo in repos if repo.name in gaps]
    _log_plan(need)

    for repo in need:
        logger.info(f"Downloading {repo.repo_id} -> {repo.local_dir}")
        try:
            snapshot_download(
                repo.repo_id,
                local_dir=str(repo.local_dir),
                allow_patterns=list(repo.files),
                force_download=force,
                max_workers=8,
            )
        except Exception as e:
            # オフライン（HF_HUB_OFFLINE=1）・レート制限・リポジトリ名変更等は、そのまま生の例外を
            # 見せるより「何が取りたかったのか」を添えて止まる（呼び出し側は PretrainedModelMissing だけで済む）
            raise PretrainedModelMissing(
                f"{repo.name}（{repo.repo_id}）の取得に失敗しました: {e}。"
                "ネットワーク接続と HF_HUB_OFFLINE / HF_TOKEN を確認するか、"
                f"`uv run -m style_bert_vits2.pretrained --only {repo.name}` を再実行してください"
            ) from e

    remaining = _gaps(repos)
    if remaining:
        raise _missing_error(remaining, reason="取得後もファイルが見つからない")
    logger.success(f"Pretrained models are ready under {PRETRAINED_ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="事前学習済みモデル（bert/slm/jp_extra/speaker）を pretrained/ に取得する"
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"取得対象をカンマ区切りで絞る（options: {', '.join(PRETRAINED_REPOS)}）",
    )
    parser.add_argument(
        "--force", action="store_true", help="既存ファイルを再取得して上書きする"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="取得せず、取得予定のファイルと総量を表示する",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="完備状況を表示するだけ（欠けがあれば exit 1）",
    )
    args = parser.parse_args()
    targets = (
        [t.strip() for t in args.only.split(",") if t.strip()] if args.only else None
    )
    repos = _resolve(targets)

    if args.check:
        gaps = _gaps(repos)
        for repo in repos:
            missing = gaps.get(repo.name, [])
            status = "OK" if not missing else f"MISSING ({', '.join(missing)})"
            print(f"{repo.name:9s} {status:20s} {repo.local_dir}")
        raise SystemExit(1 if gaps else 0)

    if args.dry_run:
        total = 0
        for repo in repos:
            plan = plan_download(repo)
            total += _planned_size(plan)
            for name, size, status in plan:
                print(f"{repo.name:9s} {status:8s} {human_size(size):>9s} {name}")
        print(f"合計 {human_size(total)} を準備予定（{PRETRAINED_ROOT}）")
        return

    ensure_pretrained(targets, force=args.force)


if __name__ == "__main__":
    main()
