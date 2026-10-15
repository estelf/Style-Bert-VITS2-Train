"""
Style-Bert-VITS2 モデルのハイパーパラメータを表す Pydantic モデル。

このモデルのフィールド既定値が **全設定の単一のソース（SSOT）** である。
- 前処理 Step1 は `default_config_dict()` でこの既定値から Data/<モデル>/config.json を生成する
  （旧 configs/config_jp_extra.json テンプレートは、値がすべてここに移ったため削除した）
- ロード時に config.json へ存在しないキーがあっても、この既定値がフェイルセーフとして適用される
- CLI からの設定上書き（`preprocess_all.py --set`）も、このスキーマを正としてパス・値を検証する
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

# 学習時にのみ動的に設定されるパラメータ (通常 config.json には存在しない)
_RUNTIME_ONLY_KEYS = ("model_dir", "out_dir", "dataset_path", "speedup")


class HyperParametersTrain(BaseModel):
    log_interval: int = 200
    eval_interval: int = 1000
    seed: int = 42
    epochs: int = 1000
    learning_rate: float = 0.0001
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    batch_size: int = 2
    lr_decay: float = (
        0.99996  # エポック単位で係数を適用する（既定値だと100エポックで約0.4%しか減衰せず実質定数LR）
    )
    segment_size: int = 16384
    warmup_epochs: int = 0
    c_mel: int = 45
    c_kl: float = 1.0
    skip_optimizer: bool = False
    freeze_JP_bert: bool = False
    freeze_style: bool = False
    freeze_decoder: bool = False
    keep_ckpts: int = 1  # 0 で全チェックポイント保持
    keep_safetensors: int = (
        6  # model_assets/ の .safetensors を step 単位で保持する個数。0 で全残し。
        # eval_interval=1000 ごとに通常+EMA の2枚（計約500MB）が増えるため、0 のまま
        # 長期学習だと disk を圧迫する。best.safetensors と best step の実体は間引き
        # 対象外なので、推論・select_best 用の1点は消えない（値の根拠は docs/DECISIONS.md §9.4）
    )
    spec_cache: bool = True  # スペクトルキャッシュ（.spec.pt）を使うか
    num_workers: int = (
        2  # データローダのワーカー数（.spec.pt キャッシュが無い初回エポックの STFT 並列数。メモリが厳しいなら 1）
    )
    accumulation_steps: int = (
        1  # 勾配累積ステップ数。1 で従来動作。VRAM 不足で batch_size を上げられない時の実効バッチ拡大用
        # （注意: global_step はオプティマイザ更新数でなくマイクロバッチ（forward-backward）数を数える。
        #   accumulation_steps>1 では eval_interval / early_stopping_min_steps 等のステップ基準の値も
        #   マイクロバッチ単位になる。実オプティマイザ更新数は global_step / accumulation_steps）
    )
    adversarial_warmup_steps: int = (
        0  # 敵対損失（fm/gen/dur_gen/lm 系）をこの実行開始から線形に 0→1 スケールするステップ数。0 で無効
    )
    ema_decay: float = (
        0.999  # 生成器パラメータの EMA 減衰率（既定有効。0 で無効）。評価・最終保存に
        # `<モデル名>_e<epoch>_s<step>_ema.safetensors` が併記される。検証損失の実測は
        # 通常重み側だが、同一 step 比較で EMA 重みの方が合成品質に優位なため、
        # best.safetensors は既定で EMA 側をコピーする（best_use_ema を参照）
    )
    best_use_ema: bool = (
        True  # best.safetensors を best step の EMA 重み（本体の _ema.safetensors）から
        # コピーする。best step 自体は通常重みの検証損失で選んだまま、書き出す実体だけ
        # 差し替える。EMA 実体が無い step（ema_decay=0 等）は自動的に通常重みへフォールバック。
        # false で従来動作（best.safetensors = 検証を実測した通常重み）
    )
    nan_guard_max_skips: int = (
        10  # 非有限損失（NaN/inf）がこの回数連続で出たら学習停止。0 で無効（スキップのみ）
    )
    # ---- 早期停止（既定で有効。val.list の非敵対損失で判定する。既存 config.json は
    # キー無しでこの既定値が適用されるため壊れない。無効化は early_stopping_enabled: false）----
    early_stopping_enabled: bool = True
    early_stopping_monitor: str = (
        "mel"  # 監視する検証損失: mel | dur | kl | lm | total（lm は WavLM 特徴一致。total は mel+dur+kl の和で lm は含まない）
    )
    early_stopping_min_steps: int = (
        -1  # 今回の学習開始点（再開時は再開step）からの経過がこれを超える評価のみ判定に使う。
        # -1（既定・自動）は学習開始時にデータセット規模から max(1000, 5×1エポックのstep数) へ
        # 解決される（train.pipeline._resolve_early_stopping_min_steps）。0 で全履歴対象。
        # 自動解決に変えた経緯は docs/DECISIONS.md §9.4
    )
    early_stopping_patience: int = (
        5  # best 更新（min_delta 以上の改善）が無い評価が続いたら停止。
        # 短めの run でも発火できる値（発火不能だった旧既定からの経緯は docs/DECISIONS.md §9.4）
    )
    early_stopping_min_delta: float = 0.001  # これ未満の改善は best 更新とみなさない
    early_stopping_min_delta_rel: float = (
        0.01  # 絶対値ではなく best に対する割合で見る閾値（1%）。実際の判定は max(min_delta, best×この値)。
        # 検証損失は評価ノイズで揺れるため、閾値はノイズ帯より広く取る（既定 1% ≒ 1.4σ。
        # 狭いとノイズの偶発更新で patience が発火しない。σ 実測と旧既定の経緯は docs/DECISIONS.md §9.3）。
        # 0 で従来動作
    )
    early_stopping_best_smooth: int = (
        5  # best チェックポイント選定を窓内中央値で行う幅（1 で従来動作の単発最小値）。
        # 評価ノイズで偶発的に低く出た step を best に選ぶ「くじ引き」を防ぎ、
        # 近傍ごと安定して低い step を選ぶ。0 以下は 1 扱い
    )
    early_stopping_action: str = (
        "stop"  # 停止条件到達時の動作: "stop" で学習停止、"reduce_lr" で学習率を半減して続行し、
        # reduce_lr_times 回使い切ったら停止（ReduceLROnPlateau 相当。プラトー後に再改善する学習で有効）
    )
    early_stopping_reduce_lr_factor: float = (
        0.5  # early_stopping_action="reduce_lr" のときの学習率乗数（0〜1）
    )
    early_stopping_reduce_lr_times: int = (
        3  # 学習率削減を許容する最大回数。使い切ったら停止条件到達で学習停止
    )
    early_stopping_plateau_window: int = (
        5  # プラトー検出の窓（評価回数）。0 で無効。patience より先に平坦化で止めたい時に使う
    )
    early_stopping_plateau_slope_eps: float = (
        0.002  # 窓内回帰傾きの絶対値が |平均値|×この値 以下ならプラトーとみなす。
        # 傾きが +閾値 を超える（ノイズ帯を超えて悪化）場合も停止する。悪化が明らかな
        # 学習を続けても best は保存済みなので得られない（docs/DECISIONS.md §9.4）
    )


class HyperParametersData(BaseModel):
    training_files: str = "Data/Dummy/train.list"
    validation_files: str = "Data/Dummy/val.list"
    max_wav_value: float = 32768.0
    sampling_rate: int = 44100
    filter_length: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_mel_channels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: float | None = None
    add_blank: bool = True
    n_speakers: int = 1
    cleaned_text: bool = True
    spk2id: dict[str, int] = {
        "Dummy": 0,
    }
    num_styles: int = 1
    style2id: dict[str, int] = {
        "Neutral": 0,
    }


class HyperParametersModelSLM(BaseModel):
    model: str = "./pretrained/slm/wavlm-base-plus"
    sr: int = 16000
    hidden: int = 768
    nlayers: int = 13
    initial_channel: int = 64


class HyperParametersModel(BaseModel):
    use_spk_conditioned_encoder: bool = True
    use_noise_scaled_mas: bool = True
    use_mel_posterior_encoder: bool = False
    use_duration_discriminator: bool = False
    use_wavlm_discriminator: bool = True
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = [3, 7, 11]
    resblock_dilation_sizes: list[list[int]] = [
        [1, 3, 5],
        [1, 3, 5],
        [1, 3, 5],
    ]
    upsample_rates: list[int] = [8, 8, 2, 2, 2]
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = [16, 16, 8, 2, 2]
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    gin_channels: int = 512
    slm: HyperParametersModelSLM = HyperParametersModelSLM()


class HyperParameters(BaseModel):
    model_name: str = "Dummy"
    version: str = "2.0-JP-Extra"
    train: HyperParametersTrain = HyperParametersTrain()
    data: HyperParametersData = HyperParametersData()
    model: HyperParametersModel = HyperParametersModel()

    # 以下は学習時にのみ動的に設定されるパラメータ (通常 config.json には存在しない)
    model_dir: str | None = None
    out_dir: str | None = None
    dataset_path: str | None = None
    speedup: bool = False

    # model_ 以下を Pydantic の保護対象から除外する
    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    def default_config_dict(cls) -> dict:
        """前処理 Step1 が Data/<モデル>/config.json を作る際の base を返す。

        旧 configs/config_jp_extra.json テンプレートが担っていた役割を、既定値の単一ソースである
        この Pydantic モデルから生成する（テンプレ二重管理の解消）。学習時にだけ入る
        model_dir/out_dir/dataset_path/speedup は config.json に含めない。
        """
        return cls().model_dump(mode="json", exclude=set(_RUNTIME_ONLY_KEYS))

    @staticmethod
    def load_from_json(json_path: str | Path) -> "HyperParameters":
        """
        与えられた JSON ファイルからハイパーパラメータを読み込む。

        Args:
            json_path (Union[str, Path]): JSON ファイルのパス

        Returns:
            HyperParameters: ハイパーパラメータ
        """

        with open(json_path, encoding="utf-8") as f:
            return HyperParameters.model_validate_json(f.read())
