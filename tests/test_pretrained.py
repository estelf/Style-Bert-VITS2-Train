"""事前学習モデル取得（style_bert_vits2.pretrained）の契約テスト（ネットワーク・CUDA 不要）。

カバーする契約:
- manifest（PRETRAINED_REPOS）が取得先の単一ソース。config の slm パス、pyannote が読む
  ファイル名とズレていない（ズレると「取得済みなのに読めない」が静かに起きる）
- 完備判定はファイル単位（0 バイトも欠落とみなす）。**揃っていれば snapshot_download を呼ばない**
- 欠けているときだけ manifest の allow_patterns 全体を要求し、取得後も欠けが残れば
  PretrainedModelMissing で止まる（黙って進めない）
- SBV2_SKIP_PRETRAINED_DOWNLOAD（constants.SKIP_PRETRAINED_DOWNLOAD）で自動取得を抑止する
- TTSModelHolder は既定で推論に必要な bert を確保し、pretrained_targets=() で抑止できる
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

import style_bert_vits2.pretrained as pretrained
from style_bert_vits2.constants import BASE_DIR, DEFAULT_BERT_MODEL_PATH
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.pretrained import (
    PRETRAINED_REPOS,
    PretrainedModelMissing,
    ensure_pretrained,
    missing_files,
)


@dataclass
class FakeFileInfo:
    """snapshot_download(dry_run=True) の戻り値の互換"""

    filename: str
    file_size: int
    is_cached: bool
    will_download: bool


class FakeHub:
    """snapshot_download の偽物。実ダウンロードの呼び出しだけを記録する"""

    def __init__(self, *, provide: bool = True, size: int = 1024) -> None:
        self.provide = provide
        self.size = size
        self.calls: list[dict] = []

    def __call__(self, repo_id: str, **kwargs) -> object:
        local_dir = Path(kwargs["local_dir"])
        files = list(kwargs["allow_patterns"])
        if kwargs.get("dry_run"):
            return [
                FakeFileInfo(
                    filename=name,
                    file_size=self.size,
                    is_cached=False,
                    will_download=True,
                )
                for name in files
            ]

        self.calls.append({"repo_id": repo_id, **kwargs})
        if self.provide:
            for name in files:
                path = local_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
        return str(local_dir)

    @property
    def repo_ids(self) -> list[str]:
        return [call["repo_id"] for call in self.calls]


@pytest.fixture
def pretrained_root(tmp_path: Path, monkeypatch) -> Path:
    """PRETRAINED_ROOT を一時ディレクトリへ差し替える（実 pretrained/ を絶対に触らない）"""
    root = tmp_path / "pretrained"
    monkeypatch.setattr(pretrained, "PRETRAINED_ROOT", root)
    return root


def make_file(name: str, root: Path, files: tuple[str, ...] | None = None) -> None:
    """manifest 通り（files 省略時は全ファイル）のダミーファイルを置く"""
    entry = PRETRAINED_REPOS[name]
    target_dir = root / entry.subdir
    for file in files or entry.files:
        path = target_dir / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")


def make_all(root: Path) -> None:
    for name in PRETRAINED_REPOS:
        make_file(name, root)


# --- manifest が単一ソースであること ---------------------------------------


def test_manifest_paths_match_consumers() -> None:
    """manifest の取得先が、パスを独自解決している消費側と一致する"""
    # BERT: constants.DEFAULT_BERT_MODEL_PATH（bert_models のデフォルト）と同じ場所を指す
    assert PRETRAINED_REPOS["bert"].local_dir == DEFAULT_BERT_MODEL_PATH
    # SLM: config テンプレの相対パス（train/pipeline.py が BASE_DIR 基準で解決する）と同じ場所
    slm_in_config = HyperParameters.default_config_dict()["model"]["slm"]["model"]
    assert BASE_DIR / slm_in_config == PRETRAINED_REPOS["slm"].local_dir
    # 全て PRETRAINED_ROOT（git 管理外のディレクトリ）の下に集約する
    for entry in PRETRAINED_REPOS.values():
        assert entry.local_dir.is_relative_to(pretrained.PRETRAINED_ROOT)


def test_manifest_includes_light_files() -> None:
    """軽い config・トークナイザーファイルもダウンロード対象（git 同梱の二重ソースを作らない）"""
    assert set(PRETRAINED_REPOS["bert"].files) == {
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "model.safetensors",
    }
    assert set(PRETRAINED_REPOS["slm"].files) == {
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
    }
    assert set(PRETRAINED_REPOS["speaker"].files) == {
        "pytorch_model.bin",
        "config.yaml",
    }
    assert set(PRETRAINED_REPOS["jp_extra"].files) == {
        "G_0.safetensors",
        "D_0.safetensors",
        "WD_0.safetensors",
    }
    # BERT リポジトリは pytorch_model.bin も持つが、重みは safetensors 1本に絞る（2倍取得の回避）
    assert "pytorch_model.bin" not in PRETRAINED_REPOS["bert"].files


def test_manifest_speaker_file_matches_pyannote() -> None:
    """pyannote がディレクトリ指定で読むファイル名と manifest が一致する（規約変更の検知）"""
    pytest.importorskip("pyannote.audio")
    from pyannote.audio.utils.hf_hub import AssetFileName

    assert str(AssetFileName.Model) in PRETRAINED_REPOS["speaker"].files


# --- 完備判定 ---------------------------------------------------------------


def test_complete_tree_touches_no_network(pretrained_root: Path, monkeypatch) -> None:
    """全ファイルが揃っていれば missing_files は空、ensure はネットワークを一切使わない"""
    make_all(pretrained_root)
    fake = FakeHub()
    monkeypatch.setattr(pretrained, "snapshot_download", fake)

    assert missing_files() == {}
    ensure_pretrained()
    assert fake.calls == []


def test_zero_byte_file_counts_as_missing(pretrained_root: Path) -> None:
    """0 バイトファイル（ダウンロード中断の残骸）は欠落とみなす"""
    make_all(pretrained_root)
    (pretrained_root / PRETRAINED_REPOS["bert"].subdir / "vocab.txt").write_bytes(b"")

    assert missing_files() == {"bert": ["vocab.txt"]}


def test_missing_files_scopes_to_targets(pretrained_root: Path) -> None:
    """targets を絞ればその対象だけの判定（推論準備に slm の欠落を報告しない）"""
    make_file("bert", pretrained_root)

    assert missing_files(("bert",)) == {}
    assert set(missing_files(("jp_extra",))) == {"jp_extra"}


# --- ensure_pretrained の取得 ------------------------------------------------


def test_ensure_downloads_only_missing_repos(
    pretrained_root: Path, monkeypatch
) -> None:
    """欠けているリポジトリだけ、manifest の allow_patterns 全体を要求して取得する"""
    make_file("jp_extra", pretrained_root)  # 元から揃っている対象
    fake = FakeHub()
    monkeypatch.setattr(pretrained, "snapshot_download", fake)

    ensure_pretrained(("bert", "jp_extra"))

    assert fake.repo_ids == [PRETRAINED_REPOS["bert"].repo_id]
    call = fake.calls[0]
    assert call["local_dir"] == str(pretrained_root / PRETRAINED_REPOS["bert"].subdir)
    assert call["allow_patterns"] == list(PRETRAINED_REPOS["bert"].files)
    assert missing_files(("bert", "jp_extra")) == {}


def test_ensure_raises_when_download_incomplete(
    pretrained_root: Path, monkeypatch
) -> None:
    """取得を試みても欠けが残る（オフライン・ファイル名変更等）なら明示的に停止する"""
    monkeypatch.setattr(pretrained, "snapshot_download", FakeHub(provide=False))

    with pytest.raises(PretrainedModelMissing, match="bert"):
        ensure_pretrained(("bert",))


def test_ensure_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="unknown pretrained target"):
        ensure_pretrained(("nonexistent",))


def test_ensure_wraps_download_error(pretrained_root: Path, monkeypatch) -> None:
    """オフライン・レート制限等のダウンロード失敗も PretrainedModelMissing に集約する"""

    def fail(repo_id: str, **kwargs) -> str:
        raise OSError("outgoing traffic has been disabled")

    monkeypatch.setattr(pretrained, "snapshot_download", fail)

    with pytest.raises(PretrainedModelMissing, match="取得に失敗しました") as excinfo:
        ensure_pretrained(("bert",))
    assert isinstance(excinfo.value.__cause__, OSError)


def test_ensure_honours_skip_flag(pretrained_root: Path, monkeypatch) -> None:
    """SBV2_SKIP_PRETRAINED_DOWNLOAD 有効時は通信せず、案内つきで止まる"""
    fake = FakeHub()
    monkeypatch.setattr(pretrained, "snapshot_download", fake)
    monkeypatch.setattr(pretrained, "SKIP_PRETRAINED_DOWNLOAD", True)

    with pytest.raises(PretrainedModelMissing, match="SBV2_SKIP_PRETRAINED_DOWNLOAD"):
        ensure_pretrained()
    assert fake.calls == []


def test_force_redownloads_completed_repo(pretrained_root: Path, monkeypatch) -> None:
    """完備していても force なら再取得する"""
    make_all(pretrained_root)
    fake = FakeHub()
    monkeypatch.setattr(pretrained, "snapshot_download", fake)

    ensure_pretrained(("slm",), force=True)

    assert fake.repo_ids == [PRETRAINED_REPOS["slm"].repo_id]
    assert fake.calls[0]["force_download"] is True


# --- 取得予定の表示（純粋関数） ----------------------------------------------


def test_planned_size_and_human_size() -> None:
    from style_bert_vits2.pretrained import _planned_size, human_size

    plan = [
        ("config.json", 100, "exists"),
        ("model.safetensors", 5 * 1024**3, "download"),
        ("vocab.txt", 2 * 1024, "copy"),
    ]
    assert _planned_size(plan) == 5 * 1024**3 + 2 * 1024
    assert human_size(5 * 1024**3) == "5.00GB"
    assert human_size(2 * 1024) == "2.0KB"


# --- 推論アプリ（TTSModelHolder）からの確保 ----------------------------------


def _patch_holder(monkeypatch, tts_model) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr(
        tts_model,
        "ensure_pretrained",
        lambda targets=None: calls.append(tuple(targets or ())),
    )
    return calls


def test_tts_model_holder_ensures_bert_by_default(tmp_path: Path, monkeypatch) -> None:
    """app.py が launch 前に通る TTSModelHolder 初期化で bert を確保する"""
    import style_bert_vits2.tts_model as tts_model

    calls = _patch_holder(monkeypatch, tts_model)
    holder = tts_model.TTSModelHolder(tmp_path / "model_assets", "cpu")

    assert calls == [("bert",)]
    assert holder.models_info == []


def test_tts_model_holder_can_skip_pretrained(tmp_path: Path, monkeypatch) -> None:
    """pretrained_targets=() なら取得を試みない（CI・オフラインの起動）"""
    import style_bert_vits2.tts_model as tts_model

    calls = _patch_holder(monkeypatch, tts_model)
    tts_model.TTSModelHolder(tmp_path, "cpu", pretrained_targets=())

    assert calls == []
