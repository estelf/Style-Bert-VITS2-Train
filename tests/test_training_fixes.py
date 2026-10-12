"""学習パイプライン修正箇所のユニットテスト（学習・データセット・CUDA 不要な純ロジック）。

カバーする契約:
- load_filepaths_and_text: 空行を除外する（train.list 途中の空行で unpack 例外を出さない）
- bert_cache: ハッシュ一致で再利用 / 不一致は stale 検出 / 旧形式（テンソルのみ）は信頼
- save_checkpoint / save_safetensors: 原子的保存（一時ファイルを残さない・再読込可能）
- _clean_safetensors: keep 個の step だけ残す / best.safetensors と step 不明は削除しない
- _safetensors_step: e/step 名前の解析（_ema 付きも同じ step）
- _val_list_hash: val.list 内容の変更検出
- _validate_train_config: log_interval / eval_interval 等の不正値を開始時に弾く
- extract_raw: path traversal を持つ zip を拒否する
- _all_ranks_finite / _zero_loss: NaN ガードの単一プロセス動作とゼロ勾配
- get_text: dataset 保持の word2ph を破壊しない（重複サンプル2回目で誤 stale 検出しない）
- _filter_eval_dataset: 評価からバケット範囲外サンプルを除外する（gt/生成音声の比較可能性）
- _pyannote_file: pyannote-audio 4.x のメモリ内音声フォーマット互換（select_best --spk-sim）
"""

import zipfile

import numpy as np
import pytest
import torch

from style_bert_vits2.models import utils as model_utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.utils import checkpoints as ckpt_utils
from style_bert_vits2.utils.bert_cache import (
    bert_feature_key,
    load_bert_feature,
    save_bert_feature,
)
from train.pipeline import (
    _all_ranks_finite,
    _clean_safetensors,
    _filter_eval_dataset,
    _safetensors_step,
    _val_list_hash,
    _validate_train_config,
    _zero_loss,
)


def test_load_filepaths_and_text_skips_empty_lines(tmp_path):
    list_file = tmp_path / "train.list"
    list_file.write_text(
        "a.wav|spk|JP|あ|a ı|0 1|1 1\n\n   \nb.wav|spk|JP|い|i|0|1\n",
        encoding="utf-8",
    )
    rows = model_utils.load_filepaths_and_text(str(list_file))
    assert [r[0] for r in rows] == ["a.wav", "b.wav"]


def test_bert_cache_roundtrip_and_stale(tmp_path):
    path = tmp_path / "a.wav.bert.pt"
    feature = torch.zeros(3, 5)
    key = bert_feature_key("あ", [1, 1])
    # 未保存 → 再生成対象
    assert load_bert_feature(path, key) == (None, False)
    save_bert_feature(path, feature, key)
    loaded, stale = load_bert_feature(path, key)
    assert not stale and loaded is not None and loaded.shape == feature.shape
    # テキストが変わったら stale（黙って再利用しない）
    loaded, stale = load_bert_feature(path, bert_feature_key("い", [1, 1]))
    assert loaded is None and stale


def test_bert_cache_old_format_trusted(tmp_path):
    path = tmp_path / "a.wav.bert.pt"
    torch.save(torch.zeros(3, 5), path)  # ハッシュ導入以前の旧形式
    loaded, stale = load_bert_feature(path, "whatever")
    assert not stale and loaded is not None and loaded.shape == (3, 5)


def test_save_checkpoint_atomic(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "G_100.pth"
    ckpt_utils.save_checkpoint(model, optimizer, 1e-4, 100, str(path))
    assert path.is_file()
    # 一時ファイル (.tmp<pid>) が残っていないこと
    assert list(tmp_path.glob("*.tmp*")) == []
    # 再読込可能（原子的リネーム後も中身が完全であること）
    model2 = torch.nn.Linear(2, 2)
    _, _, lr, iteration = ckpt_utils.load_checkpoint(str(path), model2)
    assert iteration == 100 and lr == 1e-4


def test_safetensors_step_parse():
    from pathlib import Path

    assert _safetensors_step(Path("model_e3_s1200.safetensors")) == 1200
    assert _safetensors_step(Path("model_e3_s1200_ema.safetensors")) == 1200
    assert _safetensors_step(Path("best.safetensors")) == -1


def test_clean_safetensors(tmp_path):
    for step in (100, 200, 300):
        (tmp_path / f"m_e1_s{step}.safetensors").write_bytes(b"x")
    (tmp_path / "m_e3_s300_ema.safetensors").write_bytes(b"x")
    (tmp_path / "best.safetensors").write_bytes(b"x")
    (tmp_path / "G_0.safetensors").write_bytes(b"x")  # step 不明 → 削除しない
    _clean_safetensors(tmp_path, keep=2, best_step=100)
    remaining = {p.name for p in tmp_path.glob("*.safetensors")}
    # 直近2 step（300, 200）+ best_step 指定で 100 も保持
    assert remaining == {
        "m_e1_s100.safetensors",
        "m_e1_s200.safetensors",
        "m_e1_s300.safetensors",
        "m_e3_s300_ema.safetensors",
        "best.safetensors",
        "G_0.safetensors",
    }
    _clean_safetensors(tmp_path, keep=2, best_step=None)
    remaining = {p.name for p in tmp_path.glob("*.safetensors")}
    assert "m_e1_s100.safetensors" not in remaining
    assert "best.safetensors" in remaining and "G_0.safetensors" in remaining


def test_val_list_hash(tmp_path):
    val = tmp_path / "val.list"
    val.write_text("a.wav|spk|JP|あ\n", encoding="utf-8")
    h1 = _val_list_hash(str(val))
    assert _val_list_hash(str(val)) == h1  # 同一内容なら同一ハッシュ
    val.write_text("a.wav|spk|JP|い\n", encoding="utf-8")
    assert _val_list_hash(str(val)) != h1  # 内容変更を検出
    assert _val_list_hash(str(tmp_path / "missing.list")) is None


def test_validate_train_config():
    hps = HyperParameters()
    _validate_train_config(hps)  # 既定値は通る
    hps = HyperParameters()
    hps.train.log_interval = 0
    with pytest.raises(ValueError):
        _validate_train_config(hps)
    hps = HyperParameters()
    hps.train.eval_interval = -1
    with pytest.raises(ValueError):
        _validate_train_config(hps)
    hps = HyperParameters()
    hps.train.accumulation_steps = 0
    with pytest.raises(ValueError):
        _validate_train_config(hps)
    hps = HyperParameters()
    hps.train.ema_decay = 1.5
    with pytest.raises(ValueError):
        _validate_train_config(hps)
    # 不明な monitor キーは例外にせず "mel" へ正規化（評価と停止で同じキーに揃える）
    hps = HyperParameters()
    hps.train.early_stopping_monitor = "bogus"
    _validate_train_config(hps)
    assert hps.train.early_stopping_monitor == "mel"


def test_extract_raw_rejects_path_traversal(tmp_path, monkeypatch):
    import importlib

    # preprocess/__init__.py が `from preprocess.initialize import initialize` するため、
    # `preprocess.initialize` 属性は関数に上書きされる。モジュール本体は sys.modules から取る
    initialize_mod = importlib.import_module("preprocess.initialize")
    monkeypatch.setattr(initialize_mod, "DATASET_ROOT", tmp_path)
    dataset = tmp_path / "m"
    dataset.mkdir()
    with zipfile.ZipFile(dataset / "raw.zip", "w") as zf:
        zf.writestr("a.wav", b"RIFF")
        zf.writestr("../evil.txt", b"pwned")
    with pytest.raises(ValueError):
        initialize_mod.extract_raw("m")
    assert not (tmp_path / "evil.txt").exists()


def test_all_ranks_finite_single_process():
    ok = torch.tensor([1.0, 2.0])
    assert _all_ranks_finite([ok], torch.device("cpu"))
    bad = torch.tensor([float("nan")])
    assert not _all_ranks_finite([ok, bad], torch.device("cpu"))
    inf = torch.tensor([float("inf")])
    assert not _all_ranks_finite([inf], torch.device("cpu"))


def test_zero_loss_value_and_gradient():
    # 有限損失: ゼロ化すると値も勾配もゼロになり、グラフは保たれる
    w = torch.tensor([2.0], requires_grad=True)
    zeroed = _zero_loss(w * 3.0)
    assert float(zeroed.detach()) == 0.0
    zeroed.backward()
    assert float(w.grad.abs().sum()) == 0.0

    # 非有限損失: 値は有限のゼロになり backward は例外なく通る（DDP Reducer を
    # 完走させるため backward 自体は必須。forward 中間値が NaN のとき勾配が NaNに
    # なりうるのは文書化済みで、呼び出し側が zero_grad で破棄する）
    w2 = torch.tensor([2.0], requires_grad=True)
    zeroed2 = _zero_loss(w2 * float("nan"))
    assert torch.isfinite(zeroed2).all() and float(zeroed2.detach()) == 0.0
    zeroed2.backward()


def test_get_text_does_not_mutate_word2ph(tmp_path):
    """get_text が dataset 保持の word2ph を破壊しないこと。

    旧実装は get_text 内で word2ph を in-place 倍化（死コード）していたため、
    DistributedBucketSampler のバケットパディングで生じる重複サンプルを同じ
    DataLoader ワーカーが2回読むと、2回目のアクセスが倍化済み word2ph で
    ハッシュを計算し、生成したばかりの .bert.pt を誤って stale 判定した
    （小規模runの実データ step47 で再現）。
    """
    from train.data_utils import TextAudioSpeakerLoader

    loader = TextAudioSpeakerLoader.__new__(TextAudioSpeakerLoader)
    loader.add_blank = True

    text = "こと"
    phones, tones = ["k", "o"], [0, 0]
    stored_word2ph = [1, 1]  # dataset が保持するリスト（参照で渡される）
    wav_path = tmp_path / "a.flac"
    # add_blank 後の音素長 5（intersperse は前後にも blank を入れ 2n+1）に一致する
    # 有効キャッシュ（新形式・ハッシュ一致）
    save_bert_feature(
        wav_path.with_suffix(".bert.pt"),
        torch.zeros(1024, 5),
        bert_feature_key(text, stored_word2ph),
    )

    ja_bert, phone, tone, language = loader.get_text(
        text, stored_word2ph, phones, tones, "JP", str(wav_path)
    )
    assert ja_bert.shape == (1024, 5)
    assert len(phone) == 5
    # 格納リストが不変であること（1回目のアクセス）
    assert stored_word2ph == [1, 1]
    # 同じリストオブジェクトで2回アクセスしても誤 stale にならないこと
    ja_bert2, *_ = loader.get_text(
        text, stored_word2ph, phones, tones, "JP", str(wav_path)
    )
    assert ja_bert2.shape == (1024, 5)
    assert stored_word2ph == [1, 1]


def test_pyannote_file_format_accepted():
    """_pyannote_file の出力が pyannote-audio 4.x の validate_file を通ること。

    旧形式 {"array", "sampling_rate"} は pyannote 4 で ValueError になる
    （select_best --spk-sim が 別データセットでクラッシュした）。モデルダウンロード不要で
    フォーマット互換のみ検証する
    """
    pytest.importorskip("pyannote.audio")
    from pyannote.audio.core.io import Audio

    from train.select_best import _pyannote_file

    f = _pyannote_file(np.zeros(1600, dtype=np.float32), 16000)
    validated = Audio().validate_file(dict(f))  # 例外を投げないこと
    assert validated["uri"] == "waveform"
    assert validated["waveform"].shape == (1, 1600)


def test_filter_eval_dataset_drops_out_of_bucket_samples():
    """評価からバケット範囲 [32, 1000] フレーム外のサンプルを除外すること。

    長尺サンプルが eval に残ると、音声プレビューの生成だけが max_len=1000 で
    打ち切られフル長の gt と比較できない（別データセットで gt/audio_ に 11.6 秒超が混入した）
    """

    class FakeDataset:
        def __init__(self):
            self.audiopaths_sid_text = [["a"], ["b"], ["c"], ["d"]]
            self.lengths = [10, 32, 1000, 1001]

    ds = FakeDataset()
    assert _filter_eval_dataset(ds, 32, 1000) == 2
    # 境界値（32, 1000）は保持、範囲外（10, 1001）のみ除外
    assert ds.lengths == [32, 1000]
    assert ds.audiopaths_sid_text == [["b"], ["c"]]

    # 全件保持なら無変更・0 を返す
    ds2 = FakeDataset()
    ds2.lengths = [32, 100, 500, 1000]
    assert _filter_eval_dataset(ds2, 32, 1000) == 0
    assert len(ds2.audiopaths_sid_text) == 4
