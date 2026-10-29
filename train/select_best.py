"""学習中に model_assets/<モデル名>/ へ溜まる .safetensors を品質メトリクスで評価し、best checkpoint を選ぶ。

使い方:
    uv run -m train.select_best -m <モデル名> [--utmos] [--spk-sim] [--device cuda|cpu] [--style Neutral] [--stride N] [--len-norm]

メトリクス（val.list の文をシード固定42で合成して計算。回帰テストと同じ慣習）:
    - mel_corr: 正解音声（val.list の wav）との log-mel スペクトログラム相関。追加依存ゼロ。
      既定は発話長がずれると下がる仕様なので、長さ差を無視して形状だけ見たいときは --len-norm
    - utmos:    UTMOS22 による MOS 予測（torch.hub 経由。--utmos 指定時のみ、初回はネットワーク必要）
    - spk_sim:  合成音声と正解音声の話者埋め込み（pyannote wespeaker）の cos 類似度。
                --spk-sim 指定時のみ。未取得なら pretrained/ へダウンロードが発生する（トークン不要）

候補が多い長期学習では --stride N で間引ける（最新 step は必ず評価対象。評価時間は候補数に比例）

早期停止（train/pipeline.py の early_stopping_*）と組み合わせ、停止後に全候補を評価して
推論に使う1点を選ぶ運用を想定する。早期停止が記録した models/best_checkpoint.json
（検証損失上の best）もレポートへ併記し、合成品質 best との一致・不一致を示す。
結果は model_assets/<モデル名>/best_selection.json に保存。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from style_bert_vits2.constants import ASSETS_ROOT, DATASET_ROOT
from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.tts_model import TTSModel

SEED = 42  # 回帰テスト（tests/test_main.py）と同じ固定シード


def parse_args():
    parser = argparse.ArgumentParser(
        description=".safetensors を品質メトリクスで評価して best checkpoint を選ぶ"
    )
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help="モデル名（model_assets/<モデル名> を評価）",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu（既定は自動判定）")
    parser.add_argument(
        "--style",
        default=None,
        help="合成に使うスタイル名（既定: Neutral、無ければ style2id の先頭）",
    )
    parser.add_argument(
        "--sdp-ratio",
        type=float,
        default=0.0,
        help="合成の SDP 比（既定 0.0。決定論的でチェックポイント間の比較が安定する）",
    )
    parser.add_argument(
        "--utmos",
        action="store_true",
        help="UTMOS（MOS予測）も計算する（torch.hub ダウンロードが発生するため初回は要ネットワーク）",
    )
    parser.add_argument(
        "--spk-sim",
        action="store_true",
        help="話者類似度（pyannote wespeaker 埋め込みの cos 類似度）も計算する（未取得なら pretrained/ に約98MBダウンロード。トークン不要）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="N 件おきに .safetensors を間引いて評価する（最終 step は必ず含める。"
        "eval_interval=1000 の長期学習で候補が数百個溜まったときの評価時間を短縮する。既定 1=全件）",
    )
    parser.add_argument(
        "--len-norm",
        action="store_true",
        help="mel_corr を比較前に両 log-mel を時間軸線形補間で同一フレーム数へ揃えて計算する。"
        "既定の mel_corr は発話長がずれると（無音の長さ差だけでも）不利に下がるため、"
        "スペクトル軌道の形状だけを比較したいときに有効",
    )
    return parser.parse_args()


def step_of(path: Path) -> int:
    """ファイル名 <モデル名>_e<epoch>_s<step>.safetensors から step を取り出す（取れなければ 0）"""
    m = re.search(r"_s(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def logmel(x: torch.Tensor, sr: int) -> torch.Tensor:
    """[1, N] の波形から log-mel スペクトログラム [1, 80, T] を作る"""
    spec = torch.stft(
        x,
        n_fft=1024,
        hop_length=256,
        window=torch.hann_window(1024),
        return_complex=True,
    ).abs()
    fb = AF.melscale_fbanks(513, 0, sr / 2, 80, sr)
    mel = torch.matmul(fb.T[None], spec)
    return torch.log(mel.clamp_min(1e-5))


def mel_correlation(
    gen_int16: np.ndarray,
    gen_sr: int,
    gt: np.ndarray,
    gt_sr: int,
    len_norm: bool = False,
) -> float:
    """合成音声と正解音声の log-mel 相関。

    len_norm=False（既定）は時間軸を先頭から重ねて共通部分を比較する粗い指標で、
    発話長がずれると（無音の長さ差だけでも）下がる。len_norm=True は両者を同一
    フレーム数へ線形補間してから比較するため、長さ差のペナルティを受けず
    スペクトル軌道の形状だけを見る（比較用途として頑健な代わりに、
    生成が極端に短い/長いといった劣化は検出できなくなる点に注意）
    """
    g = torch.tensor(gen_int16.astype(np.float32) / 32768.0)[None]
    t = torch.tensor(gt.astype(np.float32))[None]
    if gt_sr != gen_sr:
        t = AF.resample(t, gt_sr, gen_sr)
    mg, mt = logmel(g, gen_sr), logmel(t, gen_sr)
    if len_norm:
        # 時間軸（最終次元）を短い側へ揃えて線形補間。[1, mel, T] 形のまま interpolate
        n = min(mg.shape[-1], mt.shape[-1])
        if n >= 2:
            mg = F.interpolate(mg, size=n, mode="linear", align_corners=False)
            mt = F.interpolate(mt, size=n, mode="linear", align_corners=False)
    n = min(mg.shape[-1], mt.shape[-1])
    a, b = mg[..., :n].flatten().numpy(), mt[..., :n].flatten().numpy()
    if a.size == 0 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def select_checkpoints(ckpts: list[Path], stride: int) -> list[Path]:
    """stride 件おきに候補を間引く（step 昇順を前提に、末尾＝最新は必ず残す）。

    stride<=1 は全件。間引いても best_selection.json の比較は「同一条件の候補同士」で
    行われるため順位判定の趣旨は保たれる（評価時間を候補数に比例して短縮できる）
    """
    if stride <= 1 or len(ckpts) <= stride:
        return list(ckpts)
    kept = ckpts[::stride]
    if kept[-1] != ckpts[-1]:
        kept.append(ckpts[-1])
    return kept


def load_utmos(device: str):
    """UTMOS22 を torch.hub からロード（失敗時は None を返し mel_corr のみで続行）"""
    try:
        model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )
        return model.to(device).eval()
    except (
        Exception
    ) as e:  # ネットオフライン・hub 障害など。評価自体は続けたいので広く捕まえる
        logger.warning(f"UTMOS の読み込みに失敗したため mel_corr のみで評価します: {e}")
        return None


def utmos_score(model, audio_int16: np.ndarray, sr: int, device: str) -> float:
    x = torch.tensor(audio_int16.astype(np.float32) / 32768.0)[None]
    x = AF.resample(x, sr, 16000)
    with torch.no_grad():
        return float(model(x.to(device), torch.tensor([16000], device=device)))


def load_speaker_inference():
    """話者埋め込みモデル（pretrained/ 配下の pyannote wespeaker）をロード（失敗時は None。評価は続行）"""
    try:
        # preprocess.style_gen と同じモデル・同じ経路を再利用する（遅延ロード・未取得なら取得）
        from style_bert_vits2.speaker import load_speaker_inference as _load

        return _load()
    except Exception as e:  # オフライン・未取得等。評価自体は続けたいので広く捕まえる
        logger.warning(
            f"話者埋め込みモデルの読み込みに失敗したため spk_sim なしで評価します: {e}"
        )
        return None


def _pyannote_file(wave_float32: np.ndarray, sr: int) -> dict:
    """float32 [-1,1] の波形を pyannote-audio 4.x のメモリ内形式へ変換する。

    旧形式 {"array", "sampling_rate"}（pyannote 2/3.x 系）は 4 で削除されており、
    渡すと ValueError（Neither 'waveform' nor 'audio'）になる。pyannote は
    (channels, samples) のテンソルを受け取り、モデルサンプルレートへのリサンプリングは
    Inference 側で行う
    """
    wave = torch.from_numpy(np.ascontiguousarray(wave_float32, dtype=np.float32))
    return {"waveform": wave.unsqueeze(0), "sample_rate": sr}


def speaker_similarity(
    inference, audio_int16: np.ndarray, sr: int, gt_vec: np.ndarray
) -> float:
    """合成音声の話者埋め込みと正解音声埋め込みの cos 類似度（話者保持の度合い）"""
    gen = audio_int16.astype(np.float32) / 32768.0
    vec = np.asarray(inference(_pyannote_file(gen, sr))).flatten()
    denom = np.linalg.norm(vec) * np.linalg.norm(gt_vec)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(vec, gt_vec) / denom)


def rank_sum(results: list[dict], metrics: list[str]) -> list[float]:
    """メトリクスごとの順位（1位=1）の合計。小さいほど best。同順は mel_corr 高優先"""
    ranks = [0.0] * len(results)
    for m in metrics:
        order = sorted(range(len(results)), key=lambda i: -results[i][m])
        for rank, i in enumerate(order, start=1):
            ranks[i] += rank
    return ranks


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    assets_dir = ASSETS_ROOT / args.model_name
    config_path = assets_dir / "config.json"
    style_vec_path = assets_dir / "style_vectors.npy"
    if not config_path.is_file() or not style_vec_path.is_file():
        sys.exit(
            f"{assets_dir} に config.json / style_vectors.npy がありません（推論資産の3点セットが必要です）"
        )
    hps = HyperParameters.load_from_json(config_path)

    # val.list から評価文と正解音声を取り出す（形式: wavパス|話者|言語|セリフ|音素|トーン|word2ph）
    val_path = Path(hps.data.validation_files)
    if not val_path.is_file():
        sys.exit(f"val.list が見つかりません: {val_path}")
    evals = []
    for line in val_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4 or not parts[0]:
            continue
        evals.append({"wav": parts[0], "spk": parts[1], "text": parts[3]})
    if not evals:
        sys.exit(
            f"val.list が空です: {val_path}"
            "（前処理を --val_ratio 0 で実行していないか、--val_ratio を上げて再前処理してください）"
        )

    style = args.style or (
        "Neutral" if "Neutral" in hps.data.style2id else next(iter(hps.data.style2id))
    )
    # best.safetensors は監視 best のコピー（実体は同じ step のファイルに評価させる）
    ckpts = sorted(
        (p for p in assets_dir.glob("*.safetensors") if p.name != "best.safetensors"),
        key=step_of,
    )
    if not ckpts:
        sys.exit(f"{assets_dir} に .safetensors がありません")
    if args.stride < 1:
        sys.exit(f"--stride は 1 以上を指定してください（現在: {args.stride}）")
    n_before = len(ckpts)
    ckpts = select_checkpoints(ckpts, args.stride)
    if len(ckpts) < n_before:
        logger.info(
            f"--stride {args.stride} により候補を {n_before} → {len(ckpts)} 件へ間引きました"
            "（最新 step は必ず評価対象に含まれます）"
        )

    utmos_model = load_utmos(device) if args.utmos else None
    spk_inference = load_speaker_inference() if args.spk_sim else None
    if spk_inference is not None:
        # 正解音声の話者埋め込みはチェックポイントに依存しないので先に1回だけ計算する
        for ev in evals:
            gt, gt_sr = sf.read(ev["wav"], dtype="float32")
            if gt.ndim > 1:
                gt = gt.mean(axis=1)
            ev["gt_vec"] = np.asarray(
                spk_inference(_pyannote_file(gt, gt_sr))
            ).flatten()

    results = []
    for ckpt in ckpts:
        model = TTSModel(
            model_path=ckpt,
            config_path=config_path,
            style_vec_path=style_vec_path,
            device=device,
        )
        model.load()
        corrs, scores, sims = [], [], []
        for ev in evals:
            torch.manual_seed(SEED)
            sr, audio = model.infer(
                text=ev["text"],
                speaker_id=hps.data.spk2id[ev["spk"]],
                style=style,
                style_weight=1.0,
                line_split=False,
                sdp_ratio=args.sdp_ratio,
            )
            gt, gt_sr = sf.read(ev["wav"], dtype="float32")
            if gt.ndim > 1:
                gt = gt.mean(axis=1)
            corrs.append(mel_correlation(audio, sr, gt, gt_sr, len_norm=args.len_norm))
            if utmos_model is not None:
                scores.append(utmos_score(utmos_model, audio, sr, device))
            if spk_inference is not None:
                sims.append(speaker_similarity(spk_inference, audio, sr, ev["gt_vec"]))
        model.unload()

        row = {
            "checkpoint": ckpt.name,
            "step": step_of(ckpt),
            "mel_corr": float(np.mean(corrs)),
        }
        if scores:
            row["utmos"] = float(np.mean(scores))
        if sims:
            row["spk_sim"] = float(np.mean(sims))
        results.append(row)
        extra = f" utmos={row['utmos']:.3f}" if "utmos" in row else ""
        extra += f" spk_sim={row['spk_sim']:.3f}" if "spk_sim" in row else ""
        logger.info(f"{ckpt.name}: mel_corr={row['mel_corr']:.4f}{extra}")

    metrics = ["mel_corr"]
    if "utmos" in results[0]:
        metrics.append("utmos")
    if "spk_sim" in results[0]:
        metrics.append("spk_sim")
    ranks = rank_sum(results, metrics)
    for row, r in zip(results, ranks):
        row["rank_sum"] = r
    best = min(zip(results, ranks), key=lambda x: (x[1], -x[0]["mel_corr"]))[0]

    # 早期停止が記録した検証損失上の best（models/best_checkpoint.json）を併記し、
    # 合成品質 best との一致・不一致をレポートに残す（食い違えば mel_corr 側を推奨）
    monitor_best = None
    monitor_path = DATASET_ROOT / args.model_name / "models" / "best_checkpoint.json"
    if monitor_path.is_file():
        try:
            monitor_best = json.loads(monitor_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"best_checkpoint.json の読み込みに失敗しました: {e}")

    report = {
        "model_name": args.model_name,
        "style": style,
        "sdp_ratio": args.sdp_ratio,
        "seed": SEED,
        "stride": args.stride,
        "len_norm": args.len_norm,
        "n_candidates": len(ckpts),
        "metrics": metrics,
        "best_checkpoint": best["checkpoint"],
        "early_stopping_best": monitor_best,
        "results": results,
    }
    out_path = assets_dir / "best_selection.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"評価結果を保存しました: {out_path}")
    logger.info(f"推奨 best checkpoint: {best['checkpoint']}")
    if monitor_best is not None:
        agree = monitor_best.get("safetensors") == best["checkpoint"]
        logger.info(
            f"早期停止の監視 best（{monitor_best.get('monitor')} step {monitor_best.get('step')}）との比較: "
            + (
                "一致"
                if agree
                else "不一致。合成品質評価では別の .safetensors が推奨されています"
            )
        )


if __name__ == "__main__":
    main()
