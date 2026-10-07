import os
from pathlib import Path

from style_bert_vits2.utils.strenum import StrEnum


# Style-Bert-VITS2 のバージョン（pyproject.toml の [project].version と必ず揃える）
VERSION = "2.7.0"

# Style-Bert-VITS2 のベースディレクトリ
BASE_DIR = Path(__file__).parent.parent

# 学習データセットのルート（{DATASET_ROOT}/{model_name} に esd.list + raw.zip を置く）
## 環境変数 SBV2_DATASET_ROOT で差し替え可能（回帰テストは tests/data を使う。サブプロセスも同じ環境変数を継承する）
DATASET_ROOT = Path(os.environ.get("SBV2_DATASET_ROOT", str(BASE_DIR / "Data")))

# 学習済みモデル資産のルート（学習時は {ASSETS_ROOT}/{model_name} に保存し、推論時はここから読み込む）
ASSETS_ROOT = BASE_DIR / "model_assets"

# 学習時の分散環境変数のデフォルト（環境変数が未設定の場合に pipeline が使用。上書きしたい場合は環境変数を直接指定）
TRAIN_ENV_DEFAULTS = {
    "MASTER_ADDR": "::1",  # IPv6ループバック（WSL2 mirrored mode でも確実にbind/connectできるため localhost でなくリテラル指定）
    "MASTER_PORT": "10086",
    "WORLD_SIZE": "1",
    "RANK": "0",
    "LOCAL_RANK": "0",
}


# 利用可能な言語
## 本リポジトリは日本語のみに対応する（英語はカタカナ入力が前提。ローマ字のままの英単語は pyopenjtalk に1文字ずつ読まれるため、ユーザー辞書で吸収するかカタカナで書く運用とする）
class Languages(StrEnum):
    JP = "JP"


# 日本語 BERT モデルのデフォルトパス
DEFAULT_BERT_MODEL_PATH = BASE_DIR / "pretrained" / "bert" / "deberta-v2-large-japanese-char-wwm"

# デフォルトのユーザー辞書ディレクトリ
## style_bert_vits2.nlp.japanese.user_dict モジュールのデフォルト値として利用される
## ライブラリとしての利用などで外部のユーザー辞書を指定したい場合は、user_dict 以下の各関数の実行時、引数に辞書データファイルのパスを指定する
DEFAULT_USER_DICT_DIR = BASE_DIR / "dict_data"

# デフォルトの推論パラメータ
DEFAULT_STYLE = "Neutral"
DEFAULT_STYLE_WEIGHT = 1.0
DEFAULT_SDP_RATIO = 0.2
DEFAULT_NOISE = 0.6
DEFAULT_NOISEW = 0.8
DEFAULT_LENGTH = 1.0
DEFAULT_LINE_SPLIT = True
DEFAULT_SPLIT_INTERVAL = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT = 0.7
DEFAULT_ASSIST_TEXT_WEIGHT = 1.0

# Gradio のテーマ
## Built-in theme: "default", "base", "monochrome", "soft", "glass"
## See https://huggingface.co/spaces/gradio/theme-gallery for more themes
GRADIO_THEME = "NoCrypt/miku"
