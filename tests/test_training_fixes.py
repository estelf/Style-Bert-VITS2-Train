"""学習パイプライン修正箇所のユニットテスト（学習・データセット・CUDA 不要な純ロジック）。

カバーする契約:
- load_filepaths_and_text: 空行を除外する（train.list 途中の空行で unpack 例外を出さない）
- bert_cache: ハッシュ一致で再利用 / 不一致は stale 検出 / 旧形式（テンソルのみ）は信頼
- save_checkpoint / save_safetensors: 原子的保存（一時ファイルを残さない・再読込可能）
- _clean_safetensors: keep 個の step だけ残す / best.safetensors と step 不明は削除しない
- _safetensors_step: e/step 名前の解析（_ema 付きも同じ step）
- _val_list_hash: val.list 内容の変更検出
- _validate_train_config: log_interval / eval_interval / num_workers / segment_size 等の不正値を開始時に弾く
- extract_raw: path traversal を持つ zip を拒否する / 音声リテイク時に派生キャッシュを名前に応じて削除する
- _all_ranks_finite / _zero_loss: NaN ガードの単一プロセス動作とゼロ勾配
- get_text: dataset 保持の word2ph を破壊しない（重複サンプル2回目で誤 stale 検出しない）
- load_checkpoint: 話者埋め込み emb_g の不一致は件数に関係なく停止する
- _filter_eval_dataset: 評価からバケット範囲外サンプルを除外する（下限排他は学習サンプラーと一致）
- バケット境界: constants の単一ソース参照とコピー渡し（定数汚染の防止）
- split_train_val: 検証分割が話者の全発話を奪わない / 比率ベース（既定5%）で件数がスケールする /
  max_val_total 超過の縮小は話者按比例で偏りが出ない
- select_best: --stride 間引きと --len-norm（長さ差にペナルティを受けない mel 相関）
- _pyannote_file: pyannote-audio 4.x のメモリ内音声フォーマット互換（select_best --spk-sim）
"""

import collections
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
    （Hina 実データ step47 で再現）。
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
    （select_best --spk-sim が Hina でクラッシュした）。モデルダウンロード不要で
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
    """評価からバケット範囲 (32, 1000] フレーム外のサンプルを除外すること。

    長尺サンプルが eval に残ると、音声プレビューの生成だけが max_len=1000 で
    打ち切られフル長の gt と比較できない（Hina で gt/audio_ に 11.6 秒超が混入した）。
    下限は DistributedBucketSampler の _bisect（境界 < 長さの strict 比較）と揃え、
    ちょうど32フレームも学習では使われないので評価からも外す
    """

    class FakeDataset:
        def __init__(self):
            self.audiopaths_sid_text = [["a"], ["b"], ["c"], ["d"]]
            self.lengths = [10, 32, 1000, 1001]

    ds = FakeDataset()
    assert _filter_eval_dataset(ds, 32, 1000) == 3
    # 上限1000は保持、下限32ちょうども学習バケットでは除外なので評価からも外す
    assert ds.lengths == [1000]
    assert ds.audiopaths_sid_text == [["c"]]

    # 全件保持なら無変更・0 を返す
    ds2 = FakeDataset()
    ds2.lengths = [33, 100, 500, 1000]
    assert _filter_eval_dataset(ds2, 32, 1000) == 0
    assert len(ds2.audiopaths_sid_text) == 4


def test_bucket_boundaries_single_source():
    """バケット境界が constants の単一ソースを参照していること（3箇所の分散バグの回帰防止）。

    check_dataset の警告基準と pipeline の評価除外基準が同じ値であること。
    下限は排他的（サンプラーの strict 比較に揃えた）
    """
    import importlib

    from style_bert_vits2 import constants

    # preprocess/__init__.py が同名的な関数 check_dataset を定義して submodule 属性を
    # シャドウイングするため、モジュール本体は sys.modules から取る
    check_dataset_mod = importlib.import_module("preprocess.check_dataset")
    assert check_dataset_mod.MIN_SPEC_FRAMES == constants.MIN_SPEC_FRAMES == 32
    assert check_dataset_mod.MAX_SPEC_FRAMES == constants.MAX_SPEC_FRAMES == 1000
    assert constants.BUCKET_BOUNDARIES[0] == constants.MIN_SPEC_FRAMES
    assert constants.BUCKET_BOUNDARIES[-1] == constants.MAX_SPEC_FRAMES


def test_bucket_sampler_does_not_mutate_constants():
    """DistributedBucketSampler は受け取った boundaries を pop で破壊するため、
    pipeline が定数のコピーを渡していること（定数本体が汚染されると複数回学習で境界が欠ける）"""
    from style_bert_vits2.constants import BUCKET_BOUNDARIES
    from train.data_utils import DistributedBucketSampler

    snapshot = list(BUCKET_BOUNDARIES)

    class FakeDataset:
        lengths = [40, 100, 350, 999]  # 全件が境界内

        def __len__(self):
            return len(self.lengths)

    # サンプラーは受け取ったリストを参照のまま保持し、空バケット除去の pop で破壊する。
    # つまり定数本体を渡すと BUCKET_BOUNDARIES が汚染される（複数回学習・再開で境界が欠ける）
    passed = list(BUCKET_BOUNDARIES)
    DistributedBucketSampler(
        FakeDataset(),
        batch_size=1,
        boundaries=passed,
        num_replicas=1,
        rank=0,
    )
    assert passed != snapshot  # 破壊されること自体が pipeline のコピー渡しの理由
    assert BUCKET_BOUNDARIES == snapshot  # 定数本体は無傷

    # pipeline が実際にコピーを渡していること（list(...) を外した改変への回帰防止）
    import inspect

    from train import pipeline as pipeline_mod

    assert "list(BUCKET_BOUNDARIES)" in inspect.getsource(pipeline_mod.run)


def test_extract_raw_deletes_stale_derived_caches(tmp_path, monkeypatch):
    """音声リテイク時に派生キャッシュ（spec/mel/bert/npy）が実ファイル名どおり削除されること。

    音声 X.flac の派生ファイルは X.spec.pt / X.mel.pt / X.bert.pt（with_suffix 置換）と
    X.flac.npy（f"{{audiopath}}.npy" 追記）。旧実装は X.flac.spec.pt 等という存在しない
    名前を生成しており、古い spec が黙って再利用され続けていた（音声リテイク時の誤学習）
    """
    import importlib

    initialize_mod = importlib.import_module("preprocess.initialize")
    monkeypatch.setattr(initialize_mod, "DATASET_ROOT", tmp_path)
    dataset = tmp_path / "m"
    dataset.mkdir()
    wavs = dataset / "wavs"

    def make_zip(content: bytes):
        with zipfile.ZipFile(dataset / "raw.zip", "w") as zf:
            zf.writestr("a.flac", content)

    # 1回目の展開
    make_zip(b"RIFF-original")
    initialize_mod.extract_raw("m")
    assert (wavs / "a.flac").read_bytes() == b"RIFF-original"

    # 展開後にできる派生キャッシュ（実運用と同じ名前）を手で置く
    derived = [
        wavs / "a.spec.pt",
        wavs / "a.mel.pt",
        wavs / "a.bert.pt",
        wavs / "a.flac.npy",
    ]
    for p in derived:
        p.write_bytes(b"stale-cache")

    # 音声を入れ替えて再展開（リテイク）→ 派生キャッシュは全て削除されているべき
    make_zip(b"RIFF-retake!")
    initialize_mod.extract_raw("m")
    assert (wavs / "a.flac").read_bytes() == b"RIFF-retake!"
    for p in derived:
        assert (
            not p.exists()
        ), f"{p.name} が古いまま残っている（音声変更が検出されていない）"

    # 無変更の再実行では派生キャッシュを消さない（キャッシュ有効の冪等性）
    (wavs / "a.spec.pt").write_bytes(b"valid-cache")
    initialize_mod.extract_raw("m")
    assert (wavs / "a.spec.pt").read_bytes() == b"valid-cache"


def test_load_checkpoint_emb_g_mismatch_is_fatal(tmp_path):
    """話者埋め込み emb_g.weight の形状不一致は MAX_MISMATCHED_KEYS を通過しても必ず停止すること。

    話者数を変えて再開したとき、emb_g 1キーの不一致は黙って再初期化されると（旧実装）、
    既存話者の埋め込みがランダム値になり誰も気づかないまま別人音声で学習・推論が続く
    """
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self, n_speakers: int):
            super().__init__()
            self.emb_g = nn.Embedding(n_speakers, 4)
            self.linear = nn.Linear(2, 2)

    path = tmp_path / "G_100.pth"
    ckpt_utils.save_checkpoint(
        Model(2), torch.optim.AdamW(Model(2).parameters()), 1e-4, 100, str(path)
    )
    # 話者を増やして再開（emb_g 以外の linear は一致し、不一致は emb_g 1キーだけ）
    with pytest.raises(ValueError, match="emb_g"):
        ckpt_utils.load_checkpoint(str(path), Model(3))

    # 話者構成が一致すれば正常に読める（ガードが正常系を壊さないこと）
    ckpt_utils.load_checkpoint(str(path), Model(2))


def test_validate_train_config_extended():
    """_validate_train_config の追加項目（num_workers / batch_size / segment_size / 早期停止系）"""
    cases = {
        "num_workers": -1,
        "batch_size": 0,
        "epochs": 0,
        "learning_rate": 0.0,
        "early_stopping_patience": 0,
        "early_stopping_min_delta_rel": -0.1,
        "early_stopping_plateau_window": 2,  # 0 か 3 以上でなければならない
        "early_stopping_action": "halt",
        "early_stopping_reduce_lr_factor": 1.5,
        "early_stopping_reduce_lr_times": -1,
    }
    for key, bad_value in cases.items():
        hps = HyperParameters()
        setattr(hps.train, key, bad_value)
        with pytest.raises(ValueError):
            _validate_train_config(hps)

    # segment_size は hop_length の整数倍（slice_segments の整合条件）
    hps = HyperParameters()
    hps.train.segment_size = 16000  # 512 の倍数ではない
    with pytest.raises(ValueError):
        _validate_train_config(hps)

    # num_workers=0 は正当（persistent_workers 条件付き化と整合。メインプロセス読み込み）
    hps = HyperParameters()
    hps.train.num_workers = 0
    _validate_train_config(hps)

    # monitor に lm が追加されている（旧来 mel/dur/kl/total と併せて受理）
    hps = HyperParameters()
    hps.train.early_stopping_monitor = "lm"
    _validate_train_config(hps)
    assert hps.train.early_stopping_monitor == "lm"


def test_split_train_val_keeps_one_train_utt_per_speaker():
    """検証分割（件数固定モード）が話者の全発話を奪わず、必ず1件は学習に残すこと（B3 の回帰防止）"""
    import random

    from preprocess.text import split_train_val

    rng = random.Random(42)
    # 発話数が val_per_lang 以下の話者は、検証を len-1 件に減らして1件は学習へ残す
    spk_map = {
        "rich": [f"r{i}|rich|JP|せりふ{i}\n" for i in range(10)],
        "poor": [f"p{i}|poor|JP|せりふ{i}\n" for i in range(2)],
        "single": ["s0|single|JP|せりふ\n"],
    }
    train, val = split_train_val(
        spk_map, val_per_lang=4, val_ratio=0.05, min_val=20, max_val_total=100, rng=rng
    )
    assert len(val) == 4 + 1 + 0  # rich:4, poor:1, single:0（1件は学習に必須）
    train_spks = {line.split("|")[1] for line in train}
    assert {"rich", "poor", "single"} <= train_spks  # どの話者も学習側に1件以上
    assert len(train) + len(val) == 13

    # val_ratio=0 は全件学習（検証なし）。val_per_lang=0 かつ val_ratio>0 が既定の比率モード
    train0, val0 = split_train_val(spk_map, 0, 0.0, 20, 100, random.Random(42))
    assert val0 == [] and len(train0) == 13

    # max_val_total 超過の余剰は学習へ戻る（件数は減るがどの話者も検証に残る）
    train2, val2 = split_train_val(
        spk_map,
        val_per_lang=4,
        val_ratio=0.05,
        min_val=20,
        max_val_total=3,
        rng=random.Random(42),
    )
    assert len(val2) == 3
    assert len(train2) + len(val2) == 13


def test_split_train_val_ratio():
    """検証分割の比率モード（既定）。発話数に応じて検証件数がスケールすること"""
    import random

    from preprocess.text import speaker_val_count, split_train_val

    # 下限・上限の契約（speaker_val_count 単体）
    assert speaker_val_count(944, 0, 0.05, 20) == 47  # 5% がそのまま効く
    assert speaker_val_count(100, 0, 0.05, 20) == 20  # min_val 下限（100件の1/5まで）
    assert speaker_val_count(10, 0, 0.05, 20) == 2  # 1/5 が上限（10件に20件は回さない）
    assert speaker_val_count(1, 0, 0.05, 20) == 0  # 1件の話者は丸ごと学習側
    assert speaker_val_count(944, 4, 0.05, 20) == 4  # val_per_lang > 0 は比率より優先
    assert speaker_val_count(944, 0, 0.0, 20) == 0  # val_ratio=0 で検証なし

    spk_map = {"solo": [f"u{i}|solo|JP|せりふ{i}\n" for i in range(944)]}
    train, val = split_train_val(spk_map, 0, 0.05, 20, 100, random.Random(42))
    assert len(val) == 47 and len(train) == 897
    # 決定論的（同じシードなら同じ分割。train.seed でシードされる）
    train2, val2 = split_train_val(spk_map, 0, 0.05, 20, 100, random.Random(42))
    assert val == val2 and train == train2
    # 検証と学習が重複していない（検証発話を学習で覚えていては早期停止が測れない）
    assert not ({line for line in val} & {line for line in train})


def test_split_train_val_max_val_total_is_proportional():
    """max_val_total 超過の縮小が話者按比例で、先頭話者への偏りが出ないこと"""
    import random

    from preprocess.text import split_train_val

    spk_map = {
        f"spk{i}": [f"s{i}_{j}|spk{i}|JP|せりふ{j}\n" for j in range(100)]
        for i in range(4)
    }
    # 比率5%だと話者ごとに5件・合計20件。上限10件へ縮小すると 2.5 件ずつ → 3,3,2,2
    train, val = split_train_val(spk_map, 0, 0.05, 20, 10, random.Random(42))
    per_spk = collections.Counter(line.split("|")[1] for line in val)
    assert len(val) == 10
    assert sorted(per_spk.values()) == [2, 2, 3, 3]
    assert len(per_spk) == 4  # 先頭の話者だけ検証に残る、という偏りがない
    assert len(train) + len(val) == 400


def test_select_checkpoints_stride():
    """select_best の --stride 間引き（step 昇順・最終 step は必ず保持）"""
    from pathlib import Path

    from train.select_best import select_checkpoints

    ckpts = [Path(f"m_e1_s{s}.safetensors") for s in range(0, 10000, 1000)]
    assert select_checkpoints(ckpts, 1) == ckpts  # 無効時は全件
    kept = select_checkpoints(ckpts, 3)
    assert kept[0] == ckpts[0]
    assert kept[1] == ckpts[3]
    assert kept[-1] == ckpts[-1]  # 最新は必ず含む
    assert len(kept) == 4  # s0, s3000, s6000, s9000
    # 候補数が stride 以下なら間引きなし
    assert select_checkpoints(ckpts[:2], 5) == ckpts[:2]


def test_mel_correlation_len_norm():
    """--len-norm は時間軸を揃えてから相関するので、長さ差そのもので下がらないこと。

    周波数が上って下る山型スイープ音を、正規化時間（0→1）で同一軌道を 1.5 倍の
    実時間へ引き伸ばして gt を作る（波形を直接補間すると周波数内容まで下がるので
    位相側で生成し直す）。既定（先頭重ね・共通長で打ち切り）は生成の後半が
    下り坂なのに gt の打ち切り後続は上り坂で相関が大きく下がるのに対し、
    len_norm は同一形状へ揃えるので高い相関を維持する
    """
    from train.select_best import mel_correlation

    sr = 16000

    def sweep(duration: float) -> np.ndarray:
        # 正規化時間 u=t/duration に対する山型瞬时周波数（200→1800→200 Hz）
        t = np.arange(int(sr * duration)) / sr
        u = t / duration
        tri = 1.0 - np.abs(2.0 * (u % 1.0) - 1.0)
        phase = 2.0 * np.pi * np.cumsum(200.0 + 1600.0 * tri) / sr
        return np.sin(phase).astype(np.float32)

    gen = (sweep(1.0) * 32767).astype(np.int16)  # tts 合成相当（int16）
    gt = sweep(1.5)  # 正解音声相当（soundfile 読み込みの float [-1,1]）

    raw = mel_correlation(gen, sr, gt, sr, len_norm=False)
    normed = mel_correlation(gen, sr, gt, sr, len_norm=True)
    assert normed > 0.9, f"len_norm 相関が低すぎる: {normed}"
    assert (
        normed > raw + 0.05
    ), f"長さ差があるのに len_norm が既定より十分改善しない: normed={normed} raw={raw}"
