from typing import Any, Optional

import torch
from numpy.typing import NDArray

from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models_jp_extra import SynthesizerTrn
from style_bert_vits2.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    extract_bert_feature,
)
from style_bert_vits2.nlp.symbols import SYMBOLS


def get_net_g(
    model_path: str,
    version: str,
    device: str,
    hps: HyperParameters,
    dtype: Optional[torch.dtype] = None,
) -> SynthesizerTrn:
    """JP-Extra モデルをデバイスにロードする。

    Args:
        dtype (Optional[torch.dtype]): 推論時の重みの精度（float16 を指定するとフル半精度推論になる）。None の場合は float32 のまま。
    """
    logger.info("Using JP-Extra model")
    net_g = SynthesizerTrn(
        n_vocab=len(SYMBOLS),
        spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        # hps.model 以下のすべての値を引数に渡す
        use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
        use_duration_discriminator=hps.model.use_duration_discriminator,
        use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_layers_q=hps.model.n_layers_q,
        use_spectral_norm=hps.model.use_spectral_norm,
        gin_channels=hps.model.gin_channels,
        slm=hps.model.slm,
    ).to(device)
    net_g.state_dict()
    _ = net_g.eval()
    if model_path.endswith(".pth") or model_path.endswith(".pt"):
        _ = utils.checkpoints.load_checkpoint(
            model_path, net_g, None, skip_optimizer=True, device=device
        )
    elif model_path.endswith(".safetensors"):
        _ = utils.safetensors.load_safetensors(model_path, net_g, True, device=device)
    else:
        raise ValueError(f"Unknown model format: {model_path}")
    if dtype is not None and dtype != torch.float32:
        # フルFP16推論: 重みごとキャストする（学習時と同様に全レイヤが半精度で動く）
        logger.info(f"Casting model weights to {dtype} for full half-precision inference")
        net_g = net_g.to(dtype)
    return net_g


def get_text(
    text: str,
    hps: HyperParameters,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    norm_text, phone, tone, word2ph = clean_text_with_given_phone_tone(
        text,
        given_phone=given_phone,
        given_tone=given_tone,
        # 推論時のみ呼び出されるので、raise_yomi_error は False に設定
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    ja_bert = extract_bert_feature(
        norm_text,
        word2ph,
        device,
        assist_text,
        assist_text_weight,
    )
    del word2ph
    assert ja_bert.shape[-1] == len(phone), phone

    assert (
        ja_bert.shape[-1] == len(phone)
    ), f"Bert seq len {ja_bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return ja_bert, phone, tone, language


def infer(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,  # In the original Bert-VITS2, its speaker_name: str, but here it's id
    hps: HyperParameters,
    net_g: SynthesizerTrn,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> NDArray[Any]:
    ja_bert, phones, tones, lang_ids = get_text(
        text,
        hps,
        device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        ja_bert = ja_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        ja_bert = ja_bert[:, :-2]

    # モデルがフルFP16推論用にキャストされていても、float入力の精度を揃えて入力する
    model_dtype = next(net_g.parameters()).dtype

    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device=device, dtype=model_dtype).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        style_vec_tensor = (
            torch.from_numpy(style_vec).to(device=device, dtype=model_dtype).unsqueeze(0)
        )
        del phones
        sid_tensor = torch.LongTensor([sid]).to(device)

        output = net_g.infer(
            x_tst,
            x_tst_lengths,
            sid_tensor,
            tones,
            lang_ids,
            ja_bert,
            style_vec=style_vec_tensor,
            length_scale=length_scale,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
        )

        audio = output[0][0, 0].data.cpu().float().numpy()

        del (
            x_tst,
            tones,
            lang_ids,
            ja_bert,
            x_tst_lengths,
            sid_tensor,
            style_vec,
        )  # , emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio
