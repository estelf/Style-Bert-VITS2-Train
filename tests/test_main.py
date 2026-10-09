"""コア学習フロー（前処理 → 学習 → 合成）とコアライブラリ推論の回帰テスト。

テストデータは tests/data/<モデル名>/（esd.list + raw.zip の完成済みデータセット契約の例）に置く。
・データセット・ゴールデンデータは git に含まれないので、開発時に各自で用意する（git clone 直後は存在しないのが正常）
・差し替え: データセットを tests/data/ に置いて SBV2_TEST_MODEL を変えるだけでよい（DATASET_ROOT は自動で tests/data に向く）
・学習は毎回必ず行う（model_assets/ に成果物が残っていてもスキップしない。step1 の initialize は reset_models=True で
  models/ を事前学習モデル（G_0.safetensors）からリセットするため、常に同じ初期状態の1エポックになり、
  ゴールデン音声との比較が決定論的に保たれ、train/ の変更は必ず検出される）
・ゴールデンデータについて: 学習は意図的に1エポック（シード固定42）しか行わないため重みはノイズ混じりだが、
  その出力こそがゴールデン（tests/references/）である。目的は品質検証ではなく、transformers 等のバージョン変更で
  BERT 出力＝推論結果が破壊的に変わったことを相関差分検知するためのもので、「1エポックのノイズ音声」でも契約として一致を見る必要がある。
  ※train/pipeline.py で torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG を有効化しているため、
    GPU 学習でもビット単位で再現可能。音声長・波形ともゴールデンと一致して初めて通る強い契約チェックになっている。
・train/pipeline.py は nccl / torch.cuda.set_device() / .cuda() を無条件に使うため CPU では学習できない（CPU 学習非対応）。
  よって学習を伴うテストは CUDA がある環境でのみ実行され、無い環境では skip される。
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import wavfile

# style_bert_vits2 / preprocess を import する前にデータセットルート差し替えを示す（サブプロセスにも継承される）
TEST_DATA_ROOT = Path(__file__).parent / "data"
os.environ["SBV2_DATASET_ROOT"] = str(TEST_DATA_ROOT)

import preprocess as preprocess_pkg  # noqa: E402
from style_bert_vits2.constants import ASSETS_ROOT, DATASET_ROOT  # noqa: E402
from style_bert_vits2.logging import logger  # noqa: E402
from style_bert_vits2.tts_model import TTSModel  # noqa: E402

# 入れ替え可能なテストデータセット名（tests/data/<モデル名> を使う）
MODEL_NAME = os.environ.get(
    "SBV2_TEST_MODEL", "model_1"
)  # デフォルトは model_1（tests/data/model_1/ に esd.list + raw.zip がある想定）
DATASET_PATH = DATASET_ROOT / MODEL_NAME
# 学習で保存され、推論に使うディレクトリ（config.json / style_vectors.npy / .safetensors）
MODELS_PATH = ASSETS_ROOT / MODEL_NAME
REFERENCE_DIR = Path(__file__).parent / "references"

# 固定テキスト・固定スタイル（参照音声生成時と必ず一致させること）
TEST_TEXT = "こんにちは、初めまして。あなたの名前はなんていうの？"
TEST_STYLE = "Neutral"


def _prepare_dataset():
    """テストデータセットを学習可能な状態に前処理する（環境構築ダウンロード → Step 1〜6）。

    毎回必ず全ステップを実行する（BERT特徴・スタイル特徴は既存ファイルがあれば再利用され速い）。
    特に Step 1 の initialize は reset_models=True で models/ を事前学習モデルでリセットするので、
    学習は常に同じ初期状態の1エポックになり、ゴールデン音声との比較が決定論的に保たれる。
    """
    if (
        not (DATASET_PATH / "esd.list").is_file()
        or not (DATASET_PATH / "raw.zip").is_file()
    ):
        pytest.skip(
            f"テストデータセット {DATASET_PATH}（esd.list + raw.zip）が未配置です。"
            "データセットは git に含まれないため、各自で tests/data/<モデル名>/ に配置してください"
            f"（現在のデフォルト: SBV2_TEST_MODEL={MODEL_NAME}）。"
        )
    # エポック数は最小にして、数ステップの学習で済ませる（回帰テストなので品質は問わない）
    preprocess_pkg.preprocess_all(
        model_name=MODEL_NAME,
        batch_size=2,
        epochs=1,
        save_every_steps=1000,
        num_processes=2,
        freeze_JP_bert=False,
        freeze_style=False,
        freeze_decoder=False,
        log_interval=1000,
        val_per_lang=0,
        yomi_error="raise",
        reset_models=True,  # 前回の学習チェックポイントを残さず、毎回同じ初期状態から1エポック学習する
    )


@pytest.fixture(scope="module")
def trained_model_path() -> Path:
    """前処理（Step 1〜6）→ 必ず1エポック学習を実行し、今回生成された .safetensors を返す。

    成果物をクリアしてから毎回学習するので、train/ の変更は必ず検出される（古いモデルで推論するだけにならない）。
    学習パイプラインは CUDA 必須のため、GPU が無い環境ではここで skip する。テスト同士の順序・実行履歴に依存しない。
    """
    if not torch.cuda.is_available():
        pytest.skip(
            "train/pipeline.py は nccl / .cuda() を無条件に使うため学習が CPU では動作しない（CUDA 環境でのみ実行）"
        )
    _prepare_dataset()
    # 前回実行の成果物を消し、必ず今回の学習結果を使う
    for stale in MODELS_PATH.glob("*.safetensors"):
        stale.unlink()
    subprocess.run(
        [sys.executable, "-m", "train.pipeline", "--model_name", MODEL_NAME],
        check=True,
    )
    g_files = sorted(MODELS_PATH.glob("*.safetensors"))
    assert len(g_files) > 0, "学習済みモデルが見つかりません"
    return g_files[0]


def synthesize_and_compare(model_file: Path):
    """固定テキスト＋固定スタイルで合成し、ゴールデン音声（1エポック学習ノイズモデルの決定論的出力）と差分比較する"""

    # 学習済みモデル（.safetensors）+ style_vectors.npy + config.json の構成でロード
    model = TTSModel(
        model_path=model_file,
        config_path=MODELS_PATH / "config.json",
        style_vec_path=MODELS_PATH / "style_vectors.npy",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.load()

    # 決定論性を担保するため合成直前にシードを固定
    torch.manual_seed(42)
    sample_rate, audio = model.infer(
        text=TEST_TEXT,
        speaker_id=0,
        style=TEST_STYLE,
        style_weight=1.0,
        line_split=False,
    )
    model.unload()

    reference_path = REFERENCE_DIR / MODEL_NAME / f"{TEST_STYLE}.wav"
    if not reference_path.exists():
        # 初回は参照音声を作成してスキップ（以降の実行で差分を検知する）
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        with open(reference_path, "wb") as f:
            wavfile.write(f, sample_rate, audio)
        pytest.skip(f"参照音声を作成しました: {reference_path}")

    ref_sr, ref_audio = wavfile.read(reference_path)
    assert ref_sr == sample_rate, "サンプリングレートが参照音声と一致しません"
    # 学習（use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG）と合成はビット単位で再現可能なので、
    # duration の ceil() 量子化も含めて音声長は完全一致してこそ契約である（ズレたら何か壊れた証拠）
    assert len(audio) == len(
        ref_audio
    ), f"音声長が参照音声と一致しません: {len(audio)} != {len(ref_audio)}"

    # 相関による差分閾値テスト（完全一致ではなく破壊的変更の検知が目的）
    n = min(len(audio), len(ref_audio))
    audio, ref_audio = audio[:n], ref_audio[:n]
    a = audio.astype(np.float64) / np.abs(audio).max()
    b = ref_audio.astype(np.float64) / np.abs(ref_audio).max()
    corr = float(np.corrcoef(a, b)[0, 1])
    logger.info(f"Regression correlation with reference: {corr:.6f}")
    assert corr > 0.98, (
        f"合成音声が参照音声と大きく乖離しています (corr={corr:.4f})。"
        "ライブラリバージョン変更に伴う破壊的変更の可能性があります。"
    )


def test_pipeline_and_synthesize(trained_model_path):
    """前処理 → 学習（毎回実行・seed 42 で決定論的）→ 合成 → ゴールデン音声との差分比較の本番構成回帰テスト"""
    synthesize_and_compare(trained_model_path)


def test_infer_full_half_precision(trained_model_path):
    """コアライブラリのフルFP16推論スモークテスト（重みごと半精度にキャストして合成できること）
    ※bfloat16 は仮数が8ビットしかなく duration の ceil() 量子化と組み合わせて精度劣化が大きいため廃止済み"""
    with pytest.raises(ValueError):
        TTSModel(
            model_path=trained_model_path,
            config_path=MODELS_PATH / "config.json",
            style_vec_path=MODELS_PATH / "style_vectors.npy",
            dtype="bfloat16",
        )
    model = TTSModel(
        model_path=trained_model_path,
        config_path=MODELS_PATH / "config.json",
        style_vec_path=MODELS_PATH / "style_vectors.npy",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float16",
    )
    model.load()
    # 重みがフル半精度でロードされていることを確認
    assert model.net_g is not None
    assert next(model.net_g.parameters()).dtype == torch.float16

    torch.manual_seed(42)
    sample_rate, audio = model.infer(
        text=TEST_TEXT,
        speaker_id=0,
        style=TEST_STYLE,
        style_weight=1.0,
        line_split=False,
    )
    model.unload()

    assert sample_rate == 44100
    assert len(audio) > 0 and np.all(np.isfinite(audio))
    assert np.abs(audio.astype(np.float64)).max() > 0, "音が出力されていない"
