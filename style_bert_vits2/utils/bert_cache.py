"""BERT特徴キャッシュ（*.bert.pt）の整合性管理。

特徴本体と、その生成根拠（テキスト＋word2ph のハッシュ）を一緒に保存する。
音声ファイルの中身が変わった場合は preprocess.initialize.extract_raw が派生キャッシュ
（.spec.pt / .bert.pt / .npy）を削除するが、esd.list の同長リテイクや dict_data/ 辞書更新で
**読みだけ変わった**ケースは検出できなかった。こちら側でハッシュ照合して、古い BERT 特徴を
黙って学習・推論に使わないようにする（train/data_utils.py と preprocess/bert_gen.py の両方から使う）。
"""

import hashlib
import os
from pathlib import Path

import torch


def bert_feature_key(text: str, word2ph: list[int]) -> str:
    """BERT特徴の生成根拠のハッシュ。intersperse 前の（train.list に書かれた生の）word2ph を渡す"""
    payload = text + "|" + " ".join(map(str, word2ph))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_bert_feature(path: str | Path, feature: torch.Tensor, key: str) -> None:
    """特徴とハッシュを原子的に保存する（DataLoader ワーカーと並走しても壊れないよう tmp → rename）"""
    tmp_path = f"{path}.tmp{os.getpid()}"
    try:
        torch.save({"feature": feature, "key": key}, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_bert_feature(
    path: str | Path, expected_key: str
) -> tuple[torch.Tensor | None, bool]:
    """キャッシュを読み、(特徴, 古いキャッシュか) を返す。

    - ファイル無し / 破損 → (None, False): 再生成対象
    - 新形式で key 一致 → (特徴, False)
    - 新形式で key 不一致 → (None, True): テキストまたは辞書が変わった（Step 5 の再生成が必要）
    - 旧形式（テンソルのみで key 情報なし） → (特徴, False): 検証手段が無いため形状チェックのみで信頼する。
      （ハッシュ導入以前のキャッシュを全再生成すると、シード固定の回帰テストのゴールデン音声が
      再現できなくなるため意図的に許容。テキスト変更時に初めて不一致が検出される）
    """
    p = Path(path)
    if not p.is_file():
        return None, False
    try:
        obj = torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        return None, False
    if isinstance(obj, dict):
        feature = obj.get("feature")
        if not isinstance(feature, torch.Tensor):
            return None, False
        if obj.get("key") == expected_key:
            return feature, False
        return None, True
    if isinstance(obj, torch.Tensor):
        return obj, False
    return None, False
