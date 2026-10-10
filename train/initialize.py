import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from style_bert_vits2.constants import BASE_DIR
from style_bert_vits2.logging import logger


def download_bert_models():
    # CWD 依存を避けるため BASE_DIR 基準で解決する
    with open(BASE_DIR / "pretrained/bert/bert_models.json", encoding="utf-8") as fp:
        models = json.load(fp)
    for k, v in models.items():
        local_path = (BASE_DIR / "pretrained/bert").joinpath(k)
        for file in v["files"]:
            if not Path(local_path).joinpath(file).exists():
                logger.info(f"Downloading {k} {file}")
                hf_hub_download(v["repo_id"], file, local_dir=local_path)


def download_slm_model():
    local_path = BASE_DIR / "pretrained/slm/wavlm-base-plus"
    file = "pytorch_model.bin"
    if not Path(local_path).joinpath(file).exists():
        logger.info(f"Downloading wavlm-base-plus {file}")
        hf_hub_download("microsoft/wavlm-base-plus", file, local_dir=local_path)


def download_jp_extra_pretrained_models():
    files = ["G_0.safetensors", "D_0.safetensors", "WD_0.safetensors"]
    local_path = BASE_DIR / "pretrained/jp_extra"
    for file in files:
        if not Path(local_path).joinpath(file).exists():
            logger.info(f"Downloading JP-Extra pretrained {file}")
            hf_hub_download(
                "litagin/Style-Bert-VITS2-2.0-base-JP-Extra", file, local_dir=local_path
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_infer", action="store_true")
    args = parser.parse_args()

    download_bert_models()

    if not args.only_infer:
        download_slm_model()
        download_jp_extra_pretrained_models()


if __name__ == "__main__":
    main()
