from __future__ import annotations

from typing import TYPE_CHECKING

from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata

if TYPE_CHECKING:
    import torch


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    日本語のテキストから BERT の特徴量を抽出する (PyTorch 推論)

    Args:
        text (str): 日本語のテキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        device (str): 推論に利用するデバイス
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        torch.Tensor: BERT の特徴量
    """

    import torch

    # 各単語が何文字かを作る `word2ph` を使う必要があるので、読めない文字は必ず無視する
    # でないと `word2ph` の結果とテキストの文字数結果が整合性が取れない
    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = bert_models.load_model(device_map=device)
    bert_models.transfer_model(device)

    style_res_mean = None
    with torch.no_grad():
        tokenizer = bert_models.load_tokenizer()
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  # type: ignore
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        if assist_text:
            style_inputs = tokenizer(assist_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)  # type: ignore
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    assert len(word2ph) == len(text) + 2, text
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - assist_text_weight)
                + style_res_mean.repeat(word2phone[i], 1) * assist_text_weight
            )
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T


def extract_bert_feature_batch(
    texts: list[str],
    word2phs: list[list[int]],
    device: str,
    batch_size: int = 8,
) -> list[torch.Tensor]:
    """複数テキストの BERT 特徴を一括抽出する（preprocess/bert_gen.py 用・assist_text 非対応）。

    1件ずつの forward に対し、右パディング + attention_mask でバッチ forward して GPU
    稼働率を上げる。右パディングでは実トークンの位置（DeBERTa の相対距離を含む）が
    単一抽出と同じまま余分な pad だけ後ろに付くため、実トークンの出力は単一抽出と一致する
    （浮動小数の丸め差は許容。既存キャッシュは再生成されない限り影響しない）

    Args:
        texts (list[str]): テキストのリスト
        word2phs (list[list[int]]): 各テキストの word2ph（add_blank 済みなら済みで渡す）
        device (str): 推論に利用するデバイス
        batch_size (int): forward のバッチサイズ

    Returns:
        list[torch.Tensor]: 各テキストの BERT 特徴（extract_bert_feature と同じ形）
    """
    import torch

    # 単一版と同じく、word2ph と文字数の整合を取るため読めない文字を無視して正規化する
    processed = ["".join(text_to_sep_kata(t, raise_yomi_error=False)[0]) for t in texts]

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = bert_models.load_model(device_map=device)
    bert_models.transfer_model(device)
    tokenizer = bert_models.load_tokenizer()

    results: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(processed), batch_size):
            chunk = processed[start : start + batch_size]
            w2p = word2phs[start : start + batch_size]
            inputs = tokenizer(chunk, return_tensors="pt", padding=True)
            for i in inputs:
                inputs[i] = inputs[i].to(device)  # type: ignore
            res = model(**inputs, output_hidden_states=True)
            hidden = torch.cat(res["hidden_states"][-3:-2], -1).cpu()
            for b, (t, w) in enumerate(zip(chunk, w2p)):
                assert len(w) == len(t) + 2, t
                phone_level_feature = [
                    hidden[b, i].repeat(w[i], 1) for i in range(len(w))
                ]
                results.append(torch.cat(phone_level_feature, dim=0).T)
    return results
