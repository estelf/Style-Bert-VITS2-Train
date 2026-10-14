"""設定の単一ソース化（HyperParameters SSOT）と汎用 --set の契約テスト（学習・CUDA 不要）。

カバーする契約:
- HyperParameters.default_config_dict(): 旧 config_jp_extra.json テンプレの代替。既定値が揃い、
  実行時専用キー（model_dir/out_dir/dataset_path/speedup）は含まれず、そのまま model_validate できる
- apply_config_overrides(): 値を JSON 解釈＋フィールド型で検証して反映する（int/bool/float/list/str）
- 未知のパス・型違いの値・変更禁止キー・不正な書式は ValueError で停止する（黙って無視しない）
"""

import pytest

from preprocess.initialize import apply_config_overrides
from style_bert_vits2.models.hyper_parameters import HyperParameters


def test_default_config_dict_shape():
    """default_config_dict がテンプレ代替として必要なキーを持ち、実行時専用キーを含まない"""
    config = HyperParameters.default_config_dict()
    assert set(config) == {"model_name", "version", "train", "data", "model"}
    # 学習時にだけ入るキーは config.json に書き戻さない
    for runtime_key in ("model_dir", "out_dir", "dataset_path", "speedup"):
        assert runtime_key not in config
    # 主要な既定値が SSOT（Pydantic）から正しく供給される
    assert config["train"]["batch_size"] == 2
    assert config["train"]["seed"] == 42
    assert config["train"]["early_stopping_enabled"] is True
    assert config["data"]["sampling_rate"] == 44100
    assert config["data"]["mel_fmax"] is None
    assert config["model"]["gin_channels"] == 512
    assert config["model"]["slm"]["sr"] == 16000
    # mode="json" なので tuple は list になる（JSON シリアライズ可能）
    assert config["train"]["betas"] == [0.8, 0.99]
    # 生成した base はそのまま有効な HyperParameters としてロードできる
    HyperParameters.model_validate(config)


def test_apply_overrides_typed_values():
    """--set の値が JSON 解釈・フィールド型変換されて反映される"""
    config = HyperParameters.default_config_dict()
    apply_config_overrides(
        config,
        [
            "train.batch_size=4",
            "train.freeze_style=true",
            "train.learning_rate=0.0002",
            "train.betas=[0.5, 0.9]",
            "train.early_stopping_monitor=dur",  # JSON でない語は文字列
            "data.mel_fmax=8000",
            "model.resblock=2",  # str フィールドは数値化せず生の文字列へ戻す
            "model.slm.sr=24000",
        ],
    )
    assert config["train"]["batch_size"] == 4
    assert config["train"]["freeze_style"] is True
    assert config["train"]["learning_rate"] == 0.0002
    assert list(config["train"]["betas"]) == [0.5, 0.9]
    assert config["train"]["early_stopping_monitor"] == "dur"
    assert config["data"]["mel_fmax"] == 8000
    assert config["model"]["resblock"] == "2"
    assert config["model"]["slm"]["sr"] == 24000


def test_apply_overrides_win_over_existing():
    """既存 config.json の値より --set が後（最優先）で反映される"""
    config = HyperParameters.default_config_dict()
    config["train"]["batch_size"] = 8  # 既存 config.json の手編集を模擬
    apply_config_overrides(config, ["train.batch_size=4"])
    assert config["train"]["batch_size"] == 4


@pytest.mark.parametrize(
    "item",
    [
        "train.batchsize=4",  # キーの打ち間違い（未知のパス）
        "train.batch_size=abc",  # int フィールドに数値でない値
        "model_name=foo",  # 自動設定キーは変更禁止
        "version=x",
        "data.training_files=/tmp/x",
        "data.spk2id.Foo=0",  # dict フィールドの下に設定キーは置けない
        "train.batch_size",  # '=' が無い不正な書式
    ],
)
def test_apply_overrides_rejects_bad_input(item):
    """未知のパス・型違い・禁止キー・不正書式は ValueError で停止する（黙って無視しない）"""
    config = HyperParameters.default_config_dict()
    with pytest.raises(ValueError):
        apply_config_overrides(config, [item])


def test_apply_overrides_rejects_bad_type_keeps_config():
    """型違いの値を弾いたとき、config は元の既定値のまま（部分適用で壊れない）"""
    config = HyperParameters.default_config_dict()
    with pytest.raises(ValueError):
        apply_config_overrides(config, ["train.batch_size=abc"])
    assert config["train"]["batch_size"] == 2
