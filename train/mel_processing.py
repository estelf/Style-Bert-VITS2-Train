import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

from style_bert_vits2.models.commons import reflect_pad_1d


# 破棄予告（FutureWarning/DeprecationWarning）を握り潰すとバージョン変更の破壊的変更を検知できなくなるため、
# プロセス全体の warnings.filterwarnings(action="ignore") は行わない（必要なら呼び出し側で個別に抑止する）
MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


# 計算結果のキャッシュ。キーには値を一意に決める全パラメータを含めること
# （mel 変換は n_fft・num_mels・sampling_rate・fmin・fmax でも変わるため、fmax だけでは別設定同士で衝突する）
mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = reflect_pad_1d(
        y.unsqueeze(1),
        int((n_fft - hop_size) / 2),
        int((n_fft - hop_size) / 2),
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",  # center=False なので実際には使われない（念のため明示）
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    params_device = (
        str(n_fft)
        + "_"
        + str(num_mels)
        + "_"
        + str(sampling_rate)
        + "_"
        + str(fmin)
        + "_"
        + str(fmax)
        + "_"
        + str(spec.dtype)
        + "_"
        + str(spec.device)
    )
    if params_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[params_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    spec = torch.matmul(mel_basis[params_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    # ハン窓は win_size だけで決まるが、mel 変換行列は n_fft・num_mels・sampling_rate・fmin も効くため全パラメータをキーにする
    params_dtype_device = (
        str(n_fft)
        + "_"
        + str(num_mels)
        + "_"
        + str(sampling_rate)
        + "_"
        + str(fmin)
        + "_"
        + str(fmax)
        + "_"
        + dtype_device
    )
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if params_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[params_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = reflect_pad_1d(
        y.unsqueeze(1),
        int((n_fft - hop_size) / 2),
        int((n_fft - hop_size) / 2),
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",  # center=False なので実際には使われない（念のため明示）
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[params_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
