"""学習済みモデルの試聴GUI（音声合成タブのみ）のトップレベル呼び出し口。

学習・前処理はコマンドライン（`preprocess_all.py` / `train_model.py`）で行うため、ここにはGUIを持たせない。
"""

import argparse

import gradio as gr
import torch

from gradio_tabs.inference import create_inference_app
from style_bert_vits2.constants import ASSETS_ROOT, GRADIO_THEME, VERSION
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModelHolder

# このプロセスからはワーカーを起動して辞書を使いたいので、ここで初期化
pyopenjtalk_worker.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--dtype",
    type=str,
    default="float16",
    choices=["float32", "float16"],
    help="推論時の重みの精度（既定 float16 = フル半精度推論。float32 で全精度）。bfloat16 は精度劣化が大きいため推論では非対応",
)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=None)
parser.add_argument("--no_autolaunch", action="store_true")
parser.add_argument("--share", action="store_true")

args = parser.parse_args()
device = args.device
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

model_holder = TTSModelHolder(ASSETS_ROOT, device, dtype=args.dtype)

with gr.Blocks() as app:
    gr.Markdown(f"# Style-Bert-VITS2 試聴GUI (version {VERSION})")
    create_inference_app(model_holder=model_holder)

app.launch(
    theme=GRADIO_THEME,
    server_name=args.host,
    server_port=args.port,
    inbrowser=not args.no_autolaunch,
    share=args.share,
)
