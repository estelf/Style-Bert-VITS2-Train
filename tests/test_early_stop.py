"""早期停止の判定と best チェックポイント記録のユニットテスト。

学習・データセットを必要としない純粋なロジックテストなので CUDA が無くても走る
（test_main.py の学習テストと違い tests/data/ もゴールデン音声も不要）。

狙いは「早期停止が best ではなく停止トリガーの .pth を残す」問題の回帰防止。
keep_ckpts=1 では停止stepの .pth だけが残り best の .pth は削除されるため、
推論に使うべき best を best_checkpoint.json とログへ残すことを契約にする。

カバーする追加契約:
- best 選定の中央値スムージング（early_stopping_best_smooth。ノイズの偶発的低値をくじ引きで選ばない）
- patience の実効閾値 max(min_delta, best × min_delta_rel)（評価ノイズで発火し続けるのを防ぐ）
- 悪化方向の停止（窓内回帰傾きが +閾値 超。|傾き|≦閾値 のプラトーだけでは、明らかに
  悪化する run を早期停止できず epochs 上限まで走る空白地帯があった）
- early_stopping_min_steps=-1（自動）の解決: max(1000, 5×1エポックのstep数)。固定 3000 は
  小規模データセットで39エポック以上に相当し、判定開始前に過学習が進行した実測の修正
- 既定 patience 10→5（実測6 run で判定対象評価が5〜8回しかなく、旧既定は一度も発火不能だった）
- since_step（reduce_lr 後は削減以降の評価だけで判定）
- _reduce_lr の LambdaLR base_lrs スケール（削減がエポック境界を越えて持続）
- best.safetensors のアトミックコピー（tmp→os.replace・同一 best の再コピー防止）
- best.safetensors の中身は EMA 重み（best_use_ema 既定 true。実体が無い step は通常重みへ
  フォールバック）。検証損失の実測で best step を選ぶのは従来どおり通常重み側
"""

import json

import pytest
import torch

from style_bert_vits2.models.hyper_parameters import HyperParameters
from train.pipeline import (
    _best_eval_entry,
    _best_summary,
    _early_stop_check,
    _reduce_lr,
    _resolve_early_stopping_min_steps,
    _save_best_checkpoint,
)

# 小規模データセットの実測形状（135文・eval_interval=1000・monitor=mel）。
# 単発最小は step 8000 で、既定の中央値スムージング（窓5）後は 6000 が best。
# 以降 mel が横ばいになって 11000 でプラトー検出される。
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
        # smooth=1（従来動作）では単発の最小値をそのまま best にする
        hps = make_hps(early_stopping_min_steps=0, early_stopping_best_smooth=1)
        best, key = best_of(make_history(), hps, initial_step=0)
        assert key == "mel"
        assert best["step"] == 8000
        assert best["val"]["mel"] == pytest.approx(16.9257)

    def test_停止判定と同じ窓でbestを選ぶ(self):
        # min_steps=3000 なら step 3000 以下は窓外。窓外に最小値があっても選ばれない
        # （best 選定は既定の中央値スムージング（窓5）で窓内の安定最良 step 6000 になる）
        hps = make_hps(early_stopping_min_steps=3000)
        history = make_history()
        history[0]["val"]["mel"] = 1.0  # step 1000 を全体最小にする（窓外のはず）
        best, _ = best_of(history, hps, initial_step=0)
        assert best["step"] == 6000

    def test_min_steps未達なら全履歴から選ぶ(self):
        # 窓が空になる区間で best を出さないとおかしい（学習序盤の手動中断に備える）
        hps = make_hps(early_stopping_min_steps=3000)
        best, _ = best_of(make_history()[:2], hps, initial_step=0)
        assert best["step"] == 1000

    def test_早期停止無効なら窓を使わない(self):
        hps = make_hps(early_stopping_enabled=False, early_stopping_min_steps=99999)
        best, _ = best_of(make_history(), hps, initial_step=0)
        assert best["step"] == 6000  # 既定の中央値スムージング後では窓内最良は 6000

    def test_monitorがtotalなら別stepになる(self):
        hps = make_hps(
            early_stopping_min_steps=0,
            early_stopping_monitor="total",
            early_stopping_best_smooth=1,
        )
        best, key = best_of(make_history(), hps, initial_step=0)
        assert key == "total"
        assert best["step"] == 8000  # total = mel + 5 なので最小地点は mel と揃う

    def test_monitorにlmが使える(self):
        # WavLM 特徴一致損失も監視キーとして受理される（旧履歴に lm が無い場合は None）
        hps = make_hps(early_stopping_min_steps=0, early_stopping_monitor="lm")
        assert _best_eval_entry(make_history(), hps, initial_step=0) is None
        history = make_history()
        for i, entry in enumerate(history):
            entry["val"]["lm"] = 3.0 + 0.01 * i
        best, key = best_of(history, hps, initial_step=0)
        assert key == "lm"
        assert best["step"] == 1000  # lm は単調増加なので先頭が最小

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
        assert best["step"] == 6000  # 全履歴対象の中央値スムージングでは 6000


class TestBestSmoothing:
    """best 選定の中央値スムージング（ノイズの偶発的低値を「くじ引き」で選ばない）"""

    DIP = [10.0, 10.0, 10.0, 9.5, 10.0, 9.85, 9.8, 9.85, 10.0]

    def test_単発の幸運な低値より安定して低い近傍を選ぶ(self):
        # step 4000（mel 9.5）は単発最小だが近傍は 10.0。step 7000 付近（9.85, 9.8, 9.85）
        # の方が窓内中央値で安定して低いので、スムージング後は 7000 が best になる
        hps = make_hps(early_stopping_min_steps=0)  # 既定 smooth=5
        best, _ = best_of(make_history(self.DIP), hps, initial_step=0)
        assert best["step"] == 7000
        assert best["val"]["mel"] == pytest.approx(9.8)

    def test_smooth_1なら単発最小値(self):
        hps = make_hps(early_stopping_min_steps=0, early_stopping_best_smooth=1)
        best, _ = best_of(make_history(self.DIP), hps, initial_step=0)
        assert best["step"] == 4000

    def test_同スコアでは生値が低い方を選ぶ(self):
        # 窓内中央値が並んだら（9.85/9.8/9.85 の窓はすべて 9.85）生値で決着する
        hps = make_hps(early_stopping_min_steps=0)
        history = make_history([10.0, 9.85, 9.8, 9.85, 10.0])
        best, _ = best_of(history, hps, initial_step=0)
        assert best["val"]["mel"] == pytest.approx(9.8)


class TestBestMatchesStopReason:
    """この修正の動機そのもの: 停止理由が指す best と残す best が食い違わないこと"""

    def test_プラトーで停止したランのbestは停止stepより手前(self):
        # プラトー判定を隔離するため patience を旧既定値で明示する（新既定 patience=5 は
        # この履歴で先に発火する。新既定の発火自体は TestDefaultPatience で契約化する）
        hps = make_hps(early_stopping_min_steps=3000, early_stopping_patience=10)
        history = make_history()
        stop, reason = _early_stop_check(history, hps, initial_step=0)
        assert stop is True
        assert "プラトー" in reason

        best, key = best_of(history, hps, initial_step=0)
        assert key == "mel"
        # 既定の中央値スムージング（窓=5）では単発最小の 8000 ではなく、
        # 近傍ごと安定して低い 6000 が best になる
        assert best["step"] == 6000
        # 停止step（11000）が best として残ってはいけない
        assert best["step"] != history[-1]["step"]

    def test_patience超過でもbestはbest(self):
        hps = make_hps(early_stopping_patience=3, early_stopping_plateau_window=0)
        history = make_history()
        stop, reason = _early_stop_check(history, hps, initial_step=0)
        assert stop is True
        assert "patience" in reason
        best, _ = best_of(history, hps, initial_step=0)
        assert best["step"] == 6000


class TestBestSummary:
    def test_推論に使うファイル名を明示する(self):
        hps = make_hps(early_stopping_min_steps=0)
        best, key = best_of(make_history(), hps, initial_step=0)
        text = _best_summary(hps, best, key)
        assert "step 6000" in text
        assert "再開用" in text  # .pth は再開用だと明示している
        # 既定（best_use_ema=true）は EMA 重みの best.safetensors を推論用として勧める
        assert "best.safetensors" in text
        assert "TestModel_e192_s6000_ema.safetensors" in text

    def test_best_use_ema_offなら通常重み名を勧める(self):
        hps = make_hps(early_stopping_min_steps=0, best_use_ema=False)
        best, key = best_of(make_history(), hps, initial_step=0)
        text = _best_summary(hps, best, key)
        assert "TestModel_e192_s6000.safetensors" in text
        assert "best.safetensors" not in text


class TestSaveBestCheckpoint:
    def test_jsonへ推論資産名を書く(self, tmp_path):
        hps = make_hps(early_stopping_min_steps=0)
        hps.model_dir = str(tmp_path)
        hps.out_dir = str(tmp_path / "assets")
        best, key = best_of(make_history(), hps, initial_step=0)
        _save_best_checkpoint(hps, best, key)

        payload = json.loads((tmp_path / "best_checkpoint.json").read_text("utf-8"))
        assert payload["step"] == 6000
        assert payload["monitor"] == "mel"
        # 中央値スムージング（窓5）後の best entry（step 6000）の生 mel 値
        assert payload["value"] == pytest.approx(17.0378)
        assert payload["safetensors"] == "TestModel_e192_s6000.safetensors"
        assert payload["safetensors_ema"] == "TestModel_e192_s6000_ema.safetensors"
        assert payload["assets_dir"] == str(tmp_path / "assets")
        # 実ファイルがまだ無い（評価直後の記録）ので EMA 実体なし → normal にフォールバック
        assert payload["weights"] == "normal"

    def test_書き込み失敗は例外にしない(self, tmp_path):
        # 学習本番で disk 障害があっても停止させない（_save_eval_history と同じ慣習）
        hps = make_hps(early_stopping_min_steps=0)
        hps.model_dir = str(tmp_path / "存在しないディレクトリ")
        best, key = best_of(make_history(), hps, initial_step=0)
        _save_best_checkpoint(hps, best, key)  # 例外を投げないこと

    def test_best_safetensorsをアトミックにコピーする(self, tmp_path, monkeypatch):
        # 実体 safetensors が存在するとき best.safetensors へコピーし、tmp 残骸を残さない。
        # 直接コピーでなく tmp→os.replace なので、途中で失敗しても既存 best は壊れない。
        # ここでは _ema 実体を持たせないので、best_use_ema 既定 true でも通常重みに
        # フォールバックする（EMA 導入前の古い資産・ema_decay=0 でも壊れないこと）
        import train.pipeline as pipeline_mod

        hps = make_hps(early_stopping_min_steps=0)
        hps.model_dir = str(tmp_path)
        assets = tmp_path / "assets"
        assets.mkdir()
        hps.out_dir = str(assets)
        src = assets / "TestModel_e192_s6000.safetensors"
        src.write_bytes(b"model-payload")
        best, key = best_of(make_history(), hps, initial_step=0)

        monkeypatch.setattr(pipeline_mod, "_best_copied_step", None)
        _save_best_checkpoint(hps, best, key)

        assert (assets / "best.safetensors").read_bytes() == b"model-payload"
        assert not list(assets.glob("*.tmp*"))  # 一時ファイル残骸なし
        # 同じ best の再評価ではコピーし直さない（_best_copied_step でスキップ）
        (assets / "best.safetensors").write_bytes(b"sentinel")
        _save_best_checkpoint(hps, best, key)
        assert (assets / "best.safetensors").read_bytes() == b"sentinel"

    def test_既定はEMA重みをbestにコピーする(self, tmp_path, monkeypatch):
        # 同一 step 比較で EMA 重みの方が合成品質に優位（実測run 実測: 8/8 候補で UTMOS 優位、
        # mel 相関 7/8 優位）。best step の選定は通常重みの検証損失で行ったまま、
        # 推論の1点だけ EMA 実体にする
        import train.pipeline as pipeline_mod

        hps = make_hps(early_stopping_min_steps=0)
        hps.model_dir = str(tmp_path)
        assets = tmp_path / "assets"
        assets.mkdir()
        hps.out_dir = str(assets)
        (assets / "TestModel_e192_s6000.safetensors").write_bytes(b"normal-payload")
        (assets / "TestModel_e192_s6000_ema.safetensors").write_bytes(b"ema-payload")
        best, key = best_of(make_history(), hps, initial_step=0)

        monkeypatch.setattr(pipeline_mod, "_best_copied_step", None)
        _save_best_checkpoint(hps, best, key)

        assert (assets / "best.safetensors").read_bytes() == b"ema-payload"
        payload = json.loads((tmp_path / "best_checkpoint.json").read_text("utf-8"))
        assert payload["weights"] == "ema"
        # 選定に使った通常重み名も残す（聞き比べ・select_best 候補として実体は両方ある）
        assert payload["safetensors"] == "TestModel_e192_s6000.safetensors"

    def test_best_use_ema_offなら通常重みをコピーする(self, tmp_path, monkeypatch):
        import train.pipeline as pipeline_mod

        hps = make_hps(early_stopping_min_steps=0, best_use_ema=False)
        hps.model_dir = str(tmp_path)
        assets = tmp_path / "assets"
        assets.mkdir()
        hps.out_dir = str(assets)
        (assets / "TestModel_e192_s6000.safetensors").write_bytes(b"normal-payload")
        (assets / "TestModel_e192_s6000_ema.safetensors").write_bytes(b"ema-payload")
        best, key = best_of(make_history(), hps, initial_step=0)

        monkeypatch.setattr(pipeline_mod, "_best_copied_step", None)
        _save_best_checkpoint(hps, best, key)

        assert (assets / "best.safetensors").read_bytes() == b"normal-payload"
        payload = json.loads((tmp_path / "best_checkpoint.json").read_text("utf-8"))
        assert payload["weights"] == "normal"


class TestRelativeMinDelta:
    """patience 判定の実効閾値 = max(min_delta, best × min_delta_rel)。

    検証損失は評価ノイズで揺れる（検証47件の実測で評価1点の揺らぎ σ=0.70%）。絶対値
    min_delta=0.001 だけだとノイズの偶発更新で patience が永遠に発火せず epochs 上限まで
    走り続ける（小規模runの実測で確認した「プラトー検出しか停止しない」状態の回帰防止）。
    割合の既定も 0.5% → 1% に上げた。0.5% はノイズ帯（0.7σ）なので約4件に1件は運だけで
    best が更新され、やはり patience が発火しなかった（実測run の eval_history で実測）。
    """

    # best=10.0 に対し ±0.3% 前後で揺れるだけの実質横ばい（単発最小は 9.98）
    NOISY = [10.0, 10.02, 9.99, 10.01, 9.98, 10.03, 9.99, 10.02]

    def test_ノイズ揺れは相対閾値でpatience発火(self):
        # 相対 1%（既定）なら 10.0 からの改善 0.02〜0.05（0.2〜0.5%）は best 更新とみなさない
        hps = make_hps(
            early_stopping_min_steps=0,
            early_stopping_patience=4,
            early_stopping_plateau_window=0,
        )
        stop, reason = _early_stop_check(make_history(self.NOISY), hps, initial_step=0)
        assert stop is True
        assert "patience" in reason

    def test_実測ノイズ帯の低下はbest更新にしない(self):
        # 1評価ぶんの低下幅が 0.7%（検証47件の実測 σ=0.70% と同じ大きさ）で、その後は
        # 横ばいの履歴。新既定 1% では改善と数えないので patience=4 が発火するが、
        # 旧既定 0.5% では best 更新になって counter がリセットされ発火しない
        noisy_dip = [10.0, 9.93, 9.95, 9.94, 9.96]
        hps = make_hps(
            early_stopping_min_steps=0,
            early_stopping_patience=4,
            early_stopping_plateau_window=0,
        )
        stop, reason = _early_stop_check(make_history(noisy_dip), hps, initial_step=0)
        assert stop is True
        assert "patience" in reason

        old_default = make_hps(
            early_stopping_min_steps=0,
            early_stopping_patience=4,
            early_stopping_plateau_window=0,
            early_stopping_min_delta_rel=0.005,
        )
        stop, _ = _early_stop_check(
            make_history(noisy_dip), old_default, initial_step=0
        )
        assert stop is False  # 参考: 旧既定では止まらない（だから 1% に上げた）

    def test_min_delta_rel_0なら従来動作でノイズ更新を許す(self):
        # 絶対値 0.001 のみ（旧動作）なら 9.99 / 9.98 が best 更新になり発火しない
        # （最終 counter は 3。patience=4 なら旧動作では止まらない閾値にしておく）
        hps = make_hps(
            early_stopping_min_steps=0,
            early_stopping_patience=4,
            early_stopping_plateau_window=0,
            early_stopping_min_delta_rel=0.0,
        )
        stop, _ = _early_stop_check(make_history(self.NOISY), hps, initial_step=0)
        assert stop is False

    def test_明確な改善は相対閾値でもbest更新(self):
        # 2% ずつ改善し続ける履歴なら patience は発火しない（実効閾値 1% を超える改善）。
        # ちょうど 1% は境界で、判定が `v < best - 閾値` の strict 比較なので閾値未満扱い
        improving = [10.0, 9.8, 9.6, 9.4, 9.2, 9.0, 8.8, 8.6]
        hps = make_hps(
            early_stopping_min_steps=0,
            early_stopping_patience=3,
            early_stopping_plateau_window=0,
        )
        stop, _ = _early_stop_check(make_history(improving), hps, initial_step=0)
        assert stop is False

    def test_初回評価は必ずbest登録(self):
        # best=inf に相対閾値（inf×rel=inf）を引くと nan 比較で counter が増えていた
        # 回帰の防御。patience=2 で 2 評価分、単調改善なら停止しないこと
        hps = make_hps(
            early_stopping_min_steps=0,
            early_stopping_patience=2,
            early_stopping_plateau_window=0,
        )
        stop, _ = _early_stop_check(make_history([100.0, 50.0]), hps, initial_step=0)
        assert stop is False


class TestSinceStep:
    """reduce_lr 後は削減 step 以降の評価だけで patience/プラトーを判定する"""

    def test_削減前の横ばいは判定に数えない(self):
        # MEL 履歴は新既定（patience=5・min_steps 未解決=全窓）なら step 6000 で patience
        # 停止するが、削減を step 8000 で受けた想定（since_step=8000）なら窓は 9000 以降の
        # 3 評価。ここは改善継続中なので停止しない
        hps = make_hps()  # 実運用と同じ既定値（この履歴では since_step なしで停止する）
        history = make_history()
        stop, _ = _early_stop_check(history, hps, initial_step=0)
        assert stop is True  # 前提: since_step なしなら停止する履歴であること

        stop, _ = _early_stop_check(history, hps, initial_step=0, since_step=8000)
        assert stop is False

    def test_since_stepNoneは従来どおり全窓(self):
        hps = make_hps()
        history = make_history()
        stop_none = _early_stop_check(history, hps, initial_step=0)
        stop_wide = _early_stop_check(history, hps, initial_step=0, since_step=None)
        assert stop_none == stop_wide


class TestWorseningStop:
    """悪化が明らかなら停止する（窓内回帰傾きが +閾値 を超過）。

    旧実装は |傾き|≦閾値（プラトー）のみ停止で、ノイズ帯を超えて悪化する run は
    プラトー判定では止まらなかった。best は保存済みなので悪化中の続行に益はなく、
    評価回数の少ない小規模データセットでは patience も発火しにくく epochs 上限まで
    走りうる（小規模run/別run で kl・total が単調上昇した実測の空白地帯の修正）。
    """

    # 横ばいから悪化に転じた履歴（窓5の傾き +0.3/評価、閾値は |平均|×0.002 ≒ +0.022）
    WORSENING = [10.0, 10.0, 10.0, 10.0, 10.0, 10.3, 10.6, 10.9, 11.2, 11.5]

    def test_明らかな悪化で停止する(self):
        # patience を無効化して傾き判定だけを隔離する（この履歴では patience も成立するが、
        # 悪化停止が単独で機能することを示すため）
        hps = make_hps(early_stopping_min_steps=0, early_stopping_patience=99)
        stop, reason = _early_stop_check(
            make_history(self.WORSENING), hps, initial_step=0
        )
        assert stop is True
        assert "悪化" in reason

    def test_ノイズ帯内の悪化はプラトーとして停止する(self):
        # MEL 履歴の窓5傾きは +0.022（閾値 ±0.035 内）→ 悪化ではなくプラトー扱い（従来動作）
        hps = make_hps(early_stopping_min_steps=3000, early_stopping_patience=10)
        stop, reason = _early_stop_check(make_history(), hps, initial_step=0)
        assert stop is True
        assert "プラトー" in reason

    def test_改善継続中は悪化もプラトーも発火しない(self):
        # 窓5の傾き -0.5/評価（閾値 -0.02 を大きく下回る改善中）なら停止しない
        improving = [12.0, 11.5, 11.0, 10.5, 10.0, 9.5, 9.0]
        hps = make_hps(early_stopping_min_steps=0, early_stopping_patience=99)
        stop, _ = _early_stop_check(make_history(improving), hps, initial_step=0)
        assert stop is False


class TestDefaultPatience:
    """新既定 patience=5 は実測長（判定対象評価5〜8回）の run で到達可能であること。

    旧既定 10 は実測6 run すべてで counter 最大 6〜7 に届かず発火不能だった
    （全 run がプラトー停止）。patience は悪化停止と並ぶ安全網なので、短い run でも
    発火できる値に下げる
    """

    def test_既定patienceはMEL履歴で発火する(self):
        hps = make_hps(early_stopping_min_steps=3000)  # patience は新既定 5 のまま
        stop, reason = _early_stop_check(make_history(), hps, initial_step=0)
        assert stop is True
        assert "patience" in reason


class TestAdaptiveMinSteps:
    """early_stopping_min_steps=-1（自動）は max(1000, 5×1エポックのstep数) に解決される"""

    def test_autoは下限1000(self):
        # 小規模run 相当: 152文 / batch_size 2 = 76 step/エポック → 5×76=380 < 1000 → 下限
        hps = make_hps(early_stopping_min_steps=-1)
        assert _resolve_early_stopping_min_steps(hps, 76) == 1000
        # 解決値は hps へ書き戻される（以降の _early_stop_check / _best_eval_entry が参照）
        assert hps.train.early_stopping_min_steps == 1000

    def test_autoは5エポック相当(self):
        # 大規模run 相当: 686文 / batch_size 2 = 343 step/エポック → 5×343=1715
        # （旧固定 3000 なら約9エポック。5エポックで判定開始を早める）
        hps = make_hps(early_stopping_min_steps=-1)
        assert _resolve_early_stopping_min_steps(hps, 343) == 1715
        assert hps.train.early_stopping_min_steps == 1715

    def test_明示指定はそのまま返す(self):
        for value in (0, 3000):
            hps = make_hps(early_stopping_min_steps=value)
            assert _resolve_early_stopping_min_steps(hps, 343) == value
            assert hps.train.early_stopping_min_steps == value


class TestReduceLrMechanism:
    """_reduce_lr が LambdaLR の base_lrs もスケールし、削減がエポック境界を越えて
    持続すること（param_groups["lr"] だけだと scheduler.step() で元に戻る）"""

    def _make(self, lr_lambda):
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return opt, sched

    def test_定数lambdaで削減がstepを越えて持続(self):
        opt, sched = self._make(lambda epoch: 1.0)
        _reduce_lr([sched], 0.5)
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-5)
        opt.step()  # scheduler.step() 前に optimizer.step()（警告抑止のため本来の呼び出し順にする）
        sched.step()  # エポック境界の再計算でも 5e-5 のまま（base_lrs がスケール済み）
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-5)

    def test_warmuplambdaでも削減が乗算で持続(self):
        # lr_lambda = (epoch+1)/10 の成長カードルでも base_lrs 削減は維持される
        opt, sched = self._make(lambda epoch: (epoch + 1) / 10)
        opt.step()
        sched.step()  # epoch 1 → lr = 1e-4 × 0.2
        _reduce_lr([sched], 0.5)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-5)
        opt.step()
        sched.step()  # epoch 2 → 削減後 base_lrs × 0.3
        assert opt.param_groups[0]["lr"] == pytest.approx(1.5e-5)

    def test_Noneスケジューラは無視(self):
        # net_dur_disc / net_wd 無し設定では scheduler が None のまま渡される
        opt, sched = self._make(lambda epoch: 1.0)
        _reduce_lr([None, sched, None], 0.5)
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-5)

    def test_複数回削減は乗算(self):
        opt, sched = self._make(lambda epoch: 1.0)
        _reduce_lr([sched], 0.5)
        opt.step()
        sched.step()
        _reduce_lr([sched], 0.5)
        opt.step()
        sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(2.5e-5)

    def test_再開復元はbase_lrsとlrの両方が揃う(self):
        # torch の LRScheduler は last_epoch>=0 で構築されると param_groups["lr"] を
        # base_lrs × λ から再計算して上書きする（公式 docstring の警告どおり、torch 2.14
        # で実測）。pipeline は optimizer 読込後にスケジューラを構築するため、optimizer
        # 状態の削減済み lr は構築で失われ削減前カードルに落ちる。よって run() の復元は
        # 構築後に _reduce_lr(factor^prior) を呼び、base_lrs・lr ともに揃える
        opt, sched = self._make(lambda epoch: 1.0)
        _reduce_lr([sched], 0.5)
        opt.step()
        sched.step()  # run1: 削減後カードル lr=5e-5 で保存されたと仮定
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-5)

        # 再開を模擬: initial_lr は config 値のまま残り、スケジューラ構築で lr が再計算される
        resumed = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda epoch: 1.0, last_epoch=sched.last_epoch
        )
        assert resumed.base_lrs[0] == pytest.approx(1e-4)  # 未削減の config 値に戻る
        assert opt.param_groups[0]["lr"] == pytest.approx(
            1e-4
        )  # 構築が上書き（実測挙動）

        _reduce_lr([resumed], 0.5)  # run() の復元（prior=1 回分）と同じ呼び出し
        assert opt.param_groups[0]["lr"] == pytest.approx(
            5e-5
        )  # 現在の lr も直ちに削減後へ
        opt.step()
        resumed.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(
            5e-5
        )  # エポック境界を越えて持続
