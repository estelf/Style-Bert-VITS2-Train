"""早期停止の判定と best チェックポイント記録のユニットテスト。

学習・データセットを必要としない純粋なロジックテストなので CUDA が無くても走る
（test_main.py の学習テストと違い tests/data/ もゴールデン音声も不要）。

狙いは「早期停止が best ではなく停止トリガーの .pth を残す」問題の回帰防止。
keep_ckpts=1 では停止stepの .pth だけが残り best の .pth は削除されるため、
推論に使うべき best を best_checkpoint.json とログへ残すことを契約にする。
"""

import json

import pytest

from style_bert_vits2.models.hyper_parameters import HyperParameters
from train.pipeline import (
    _best_eval_entry,
    _best_summary,
    _early_stop_check,
    _save_best_checkpoint,
)

# 小規模データセットの実測形状（135文・eval_interval=1000・monitor=mel）。
# step 8000 で best を取り、以降 mel が横ばいになって 11000 でプラトー検出される。
MEL = [
    17.1175,
    17.6661,
    17.5575,
    17.3363,
    18.0245,
    17.0378,
    17.5479,
    16.9257,
    17.7110,
    17.5116,
    17.3654,
]


def make_history(mel_list=None):
    """eval_history.json と同じ形状の履歴を作る"""
    mel_list = MEL if mel_list is None else mel_list
    return [
        {
            "step": (i + 1) * 1000,
            "epoch": (i + 1) * 32,
            "val": {
                "mel": v,
                "dur": 2.8 + 0.09 * i,
                "kl": 2.4 + 0.15 * i,
                "total": v + 5.0,
            },
        }
        for i, v in enumerate(mel_list)
    ]


def make_hps(**train_overrides):
    hps = HyperParameters()
    hps.model_name = "TestModel"
    hps.model_dir = "/tmp/models"
    hps.out_dir = "/tmp/assets"
    for key, value in train_overrides.items():
        setattr(hps.train, key, value)
    return hps


def best_of(history: list[dict], hps, initial_step: int = 0) -> tuple[dict, str]:
    """_best_eval_entry の None を落として返す（このテスト群では必ず評価が存在する前提）"""
    result = _best_eval_entry(history, hps, initial_step)
    assert result is not None
    return result


class TestBestEvalEntry:
    def test_監視指標の最小値を返す(self):
        hps = make_hps(early_stopping_min_steps=0)
        best, key = best_of(make_history(), hps, initial_step=0)
        assert key == "mel"
        assert best["step"] == 8000
        assert best["val"]["mel"] == pytest.approx(16.9257)

    def test_停止判定と同じ窓でbestを選ぶ(self):
        # min_steps=3000 なら step 3000 以下は窓外。窓外に最小値があっても選ばれない
        hps = make_hps(early_stopping_min_steps=3000)
        history = make_history()
        history[0]["val"]["mel"] = 1.0  # step 1000 を全体最小にする（窓外のはず）
        best, _ = best_of(history, hps, initial_step=0)
        assert best["step"] == 8000

    def test_min_steps未達なら全履歴から選ぶ(self):
        # 窓が空になる区間で best を出さないとおかしい（学習序盤の手動中断に備える）
        hps = make_hps(early_stopping_min_steps=3000)
        best, _ = best_of(make_history()[:2], hps, initial_step=0)
        assert best["step"] == 1000

    def test_早期停止無効なら窓を使わない(self):
        hps = make_hps(early_stopping_enabled=False, early_stopping_min_steps=99999)
        best, _ = best_of(make_history(), hps, initial_step=0)
        assert best["step"] == 8000

    def test_monitorがtotalなら別stepになる(self):
        hps = make_hps(early_stopping_min_steps=0, early_stopping_monitor="total")
        best, key = best_of(make_history(), hps, initial_step=0)
        assert key == "total"
        assert best["step"] == 8000  # total = mel + 5 なので最小地点は mel と揃う

    def test不正キーはmelへフォールバック(self):
        hps = make_hps(early_stopping_min_steps=0, early_stopping_monitor="bogus")
        _, key = best_of(make_history(), hps, initial_step=0)
        assert key == "mel"

    def test_評価が無ければNone(self):
        hps = make_hps()
        assert _best_eval_entry([], hps, initial_step=0) is None
        assert _best_eval_entry([{"step": 1, "val": {}}], hps, initial_step=0) is None

    def test_再開時はinitial_stepから数える(self):
        # 履歴に古い step が残っていても、今回の学習開始点 + min_steps より前は窓外
        hps = make_hps(early_stopping_min_steps=3000)
        best, _ = best_of(make_history(), hps, initial_step=5000)
        assert best["step"] == 11000  # 窓は step 9000 以降なのでそこでの最小が 11000

    def test_窓が空なら全履歴へフォールバック(self):
        # 開始点が高すぎて窓が空（まだ評価が無い）なら、全履歴から best を出し続ける
        hps = make_hps(early_stopping_min_steps=3000)
        best, _ = best_of(make_history(), hps, initial_step=8000)
        assert best["step"] == 8000


class TestBestMatchesStopReason:
    """この修正の動機そのもの: 停止理由が指す best と残す best が食い違わないこと"""

    def test_プラトーで停止したランのbestは停止stepより手前(self):
        hps = make_hps()  # 実運用と同じ既定値
        history = make_history()
        stop, reason = _early_stop_check(history, hps, initial_step=0)
        assert stop is True
        assert "プラトー" in reason

        best, key = best_of(history, hps, initial_step=0)
        assert key == "mel"
        assert best["step"] == 8000
        # 停止step（11000）が best として残ってはいけない
        assert best["step"] != history[-1]["step"]

    def test_patience超過でもbestはbest(self):
        hps = make_hps(early_stopping_patience=3, early_stopping_plateau_window=0)
        history = make_history()
        stop, reason = _early_stop_check(history, hps, initial_step=0)
        assert stop is True
        assert "patience" in reason
        best, _ = best_of(history, hps, initial_step=0)
        assert best["step"] == 8000


class TestBestSummary:
    def test_推論に使うファイル名を明示する(self):
        hps = make_hps(early_stopping_min_steps=0)
        best, key = best_of(make_history(), hps, initial_step=0)
        text = _best_summary(hps, best, key)
        assert "step 8000" in text
        assert "TestModel_e256_s8000.safetensors" in text
        assert "再開用" in text  # .pth は再開用だと明示している


class TestSaveBestCheckpoint:
    def test_jsonへ推論資産名を書く(self, tmp_path):
        hps = make_hps(early_stopping_min_steps=0)
        hps.model_dir = str(tmp_path)
        hps.out_dir = str(tmp_path / "assets")
        best, key = best_of(make_history(), hps, initial_step=0)
        _save_best_checkpoint(hps, best, key)

        payload = json.loads((tmp_path / "best_checkpoint.json").read_text("utf-8"))
        assert payload["step"] == 8000
        assert payload["monitor"] == "mel"
        assert payload["value"] == pytest.approx(16.9257)
        assert payload["safetensors"] == "TestModel_e256_s8000.safetensors"
        assert payload["assets_dir"] == str(tmp_path / "assets")

    def test_書き込み失敗は例外にしない(self, tmp_path):
        # 学習本番で disk 障害があっても停止させない（_save_eval_history と同じ慣習）
        hps = make_hps(early_stopping_min_steps=0)
        hps.model_dir = str(tmp_path / "存在しないディレクトリ")
        best, key = best_of(make_history(), hps, initial_step=0)
        _save_best_checkpoint(hps, best, key)  # 例外を投げないこと
