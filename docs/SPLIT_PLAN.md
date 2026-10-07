# Style-Bert-VITS2 分割計画

本リポジトリを「学習コア」を中心に分割・簡素化するための方針ドキュメント。
これまでの方針検討で**確定した決定事項**と、残す作業・未決事項をまとめる。

---

## 1. 全体方針（確定済み）

| # | 項目 | 決定内容 |
|---|------|----------|
| 1 | コアに残す機能 | 学習機能（CLI＋ログ出力）、スタイルベクトル作成（学習の後工程として維持）、学習済みモデルの試聴 GUI、TensorBoard による学習曲線確認 |
| 2 | 分離する機能 | データセット作成（スライス・ASR・リサンプリング等）、マージ、評価、アプリとしての推論（簡易UI・APIサーバー）。分離先はコアを `style-bert-vits2` パッケージとして依存する形で利用 |
| 3 | 廃止する機能 | ONNX 変換、`.bat` 系スクリプト、Colab 向け機能、コアからの API サーバー機能 |
| 4 | 言語 | **日本語のみに絞る。ただしモデルは再学習せず既存チェックポイントそのまま使う** |
| 5 | 英語混在テキスト | **カタカナ入力必須＋ユーザー辞書（`dict_data/`）で吸収**。自動転写レイヤーは追加しない。学習データ側の工夫が必要になったら分離先データセットリポジトリで対応する |
| 6 | 環境 | パッケージ管理は uv に統一（`pyproject.toml` + `uv.lock`）。requirements*.txt は全廃。バージョンは最新化していく |

---

## 2. コアリポジトリに残す範囲

### 学習パイプライン
- **学習スクリプトは `train_ms_jp_extra.py` に統一**（`train_ms.py` は削除）。手元の既存チェックポイントが全て JP-Extra 系であることを確認済みであり、v2.1 アーキテクチャ `models/models.py` も削除して JP-Extra 1本化とする
- 学習前処理チェーン: `preprocess/check_dataset.py`（データセット検証）→ `preprocess/text.py` → `preprocess/bert_gen.py` → `preprocess/style_gen.py`（スタイルベクトル作成は学習後工程。既存フローと一致）。トップレベルの `preprocess_all.py` が呼び出し口
- `data_utils.py`, `mel_processing.py`, `losses.py`, `default_style.py`, `configs/`
- 学習ログ・学習曲線: TensorBoard をそのまま利用

### 推論ライブラリコード（アプリではなく部品としてコアに残る）
- `style_bert_vits2/tts_model.py`, `voice.py`, `models/`
- `style_bert_vits2/nlp/`（日本語のみ。下記 §4 参照）
- `pretrained/`（bert/jp_extra/slm を統合的に一元管理）、`dict_data/`

### 試聴 GUI
- スリム化した Gradio UI（音声合成タブ相当。設定項目は既存のまま）
- 台本編集・API 連携を伴うアプリ本体（Editor / API サーバー）は分離先で持つ

---

## 3. 削除・移動対象一覧

### 3.1 コアから削除（廃止機能）

| 対象 | 備考 |
|---|---|
| `convert_onnx.py`, `convert_bert_onnx.py`, `gradio_tabs/convert_onnx.py` | ONNX 変換の廃止 |
| `style_bert_vits2/nlp/onnx_bert_models.py`、`bert/*-onnx` モデル群 | 同上 |
| `TTSModelHolder(..., onnx_providers=..., ignore_onnx=...)` の引数整理 | コード修正 |
| `tests/test_main.py` の ONNX テスト（4本）と ONNX 系テストの整理 | テスト書き直しは §6 |
| `*.bat`, `scripts/*.bat` | uv でのインストール手順に置換 |
| `colab.ipynb`, `requirements-colab.txt`, `requirements-infer.txt`, `library.ipynb` | Colab・配布形態の整理 |
| `server_fastapi.py`, `default_config.yml` の `server:` セクション | API サーバーは分離先で持つ |
| `style_bert_vits2/models/models.py`（v2.1 アーキテクチャ）と `pretrained/` 関連の分岐 | 手元モデルは全て JP-Extra のため。`infer.py` の v2.1 分岐も削除 |
| `initialize.py` の `download_default_models()`（デフォルト TTS モデルのダウンロード） | デフォルトモデルは使わない。BERT（JP）・事前学習（JP-Extra）・SLM のみ維持 |

### 3.2 分離先リポジトリへ移動（計3リポジトリ）

| 対象 | 移行先 |
|---|---|
| `slice.py`（音声スライス）, `transcribe.py`（ASR）, `resample.py`（リサンプリング・ラウドネス正規化・無音トリム）, `gradio_tabs/dataset.py` | **データセット作成リポジトリ**: 生音声 → 加工済み音声＋リストファイルの生成までを責任範囲とする。コアは完成済みデータセットを受け取るだけ |
| `gradio_tabs/merge.py`（マージ機能）, `speech_mos.py`（自然性評価・最適ステップ選定） | **ユーティリティリポジトリ**: コア由来の `TTSModelHolder` + safetensors に依存する独立ツール（マージ・評価） |
| アプリとしての推論（Editor 本体・API サーバー等、`server_editor.py` 含む） | コアを `style-bert-vits2` パッケージとして依存する形で利用 |

※ 移動対象 4 スクリプトはコアへの依存が `config` / `logging` 程度しかなく、クリーンに移せることを確認済み。

### 3.3 削除しない・現状維持（重要）

- **`symbols.py` は変更しない**: トークン表・トーンオフセット・`LANGUAGE_ID_MAP` は既存チェックポイントに焼き込まれているため、再学習なしの方針では多言語表のまま残す
- `librosa` は `data_utils.py` / `mel_processing.py` が使用するためコアに残る
- `default_style.py`（フォルダ分け→スタイル生成）は `style_gen` 経由の学習後工程なのでコア残留

---

## 4. 日本語のみの構成（確定済み）

再学習なしで進めるため、**モデル構造・トークン表はそのまま**で、ルーティングとモデルダウンロードだけを削る。

- `style_bert_vits2/nlp/english/`, `style_bert_vits2/nlp/chinese/` を削除し、`nlp/__init__.py` の EN/ZH ルーティングを除去
  - 全入力を日本語経路（カタカナ化 → 日本語 BERT）に流す。EN/ZH セグメントが無いため `en_bert` 等は常にゼロ埋めとなり、既存チェックポイントと整合する（学習時も純日本語データではゼロ埋めだったため互換 OK）
- `bert/bert_models.json` から EN（`deberta-v3-large`）・ZH（`chinese-roberta-wwm-ext-large`）エントリを削除（遅延ロードなので他コード修正なしで動作。ダウンロード量・VRAM 削減）
- **仕様としての明文化**: 英語はカタカナ入力が前提。ローマ字のままの英単語は pyopenjtalk に1文字ずつ読まれるため、ユーザー辞書で吸収するかカタカナで書く運用とする
- 削除可能になる依存: `cmudict`, `g2p_en`, `nltk`, `jieba`, `pypinyin`, `cn2an` および ONNX/ASR 系（`onnx*`, `faster-whisper`, `pyloudnorm` など分離移動分）
- CLI/UI から言語選択・`--freeze_EN_bert` / `--freeze_ZH_bert` 等のオプションを削減
- スクリプト統一に伴い `use_jp_extra` フラグ由来の分岐（`preprocess_text.py` / `data_utils.py` / 学習タブ / `infer.py`）を全て削除。設定も `configs/config_jp_extra.json` 一本化、`initialize.py` の `pretrained/`（v2.1 用）ダウンロードも不要に

---

## 5. データセット・モデルの契約（コアが受け取る入力）

### データセット構造（分離先リポジトリの出力＝コアの入力）

```
xx/データセット1
  ├─ esd.list
  │   音声ファイル名 | 話者名 | JP | セリフ文字列
  └─ raw.zip
      └─ .flac 音声データ（スライス・正規化・トリム済み）
```

- 現行の `preprocess/text.py` が読む `.list` 形式と一致しており追加変換は不要
- ノイズカット・ラウドネス正規化・無音トリムは**分離先側の責任**（現状 `resample.py` が持つ機能もそちらへ移る）。コアの学習フローから Step2（resample）は消え、`preprocess_all.py` は `preprocess_text → bert_gen → style_gen` の構成になる
- セリフは日本語表記（英語混在時はカタカナ）で記述される前提

### 学習済みモデル構造

```
xx/モデル名
  ├─ 学習済みモデル（.safetensors）
  ├─ style_vectors.npy
  └─ config.json
```

- 1話者＝1モデル。データセット名とモデル名は統一する

---

## 6. 環境・パッケージ（確定済み）

- **uv + `pyproject.toml` / `uv.lock` に一元化**、requirements*.txt は全廃
- バージョンは基本的に最新化していく方針（学習済みモデル互換性の範囲ではあるが、更新優先で合意済み）
- 注意点: `transformers` 等のバージョン変更は BERT 出力＝推論結果が変わりうる。固定テキスト＋固定スタイルで合成し参照音声と差分比較する**回帰テストを1本用意**し、破壊的変更を検知できるようにする（`torch<2.4`, `numpy<2` 等のピン解除時も同様）
- `initialize.py`: デフォルト TTS モデルのダウンロードは廃止。JP BERT・JP-Extra 事前学習モデル・SLM（いずれも `pretrained/` 配下にダウンロード）のみ残す

---

## 7. 実施ステップ（案）

1. **環境統一**: uv + pyproject 化、requirements 全廃
2. **JP のみ化・スクリプト統一**: EN/ZH モジュール・依存・bert_models.json エントリ削除。学習は `train_ms_jp_extra.py` に統一（`train_ms.py` / `models/models.py` 削除、`use_jp_extra` 分岐撤去）。`app.py`/テストから言語オプションを除去
3. **便利機能の分離**: データセット作成系 → データセット作成リポジトリ、マージ・評価 → ユーティリティリポジトリ、コア側のタブ・フローから該当ステップを削除
4. **廃止機能の削除**: ONNX 一式、`.bat`, Colab 系, `server_fastapi.py`、デフォルトモデルダウンロード
5. **最新化と検証**: バージョン更新、回帰テスト（合成音声の差分閾値テスト）で既存モデル互換を確認
6. **分離先リポジトリ側に契約ドキュメント整備**: データセット形式（§5）・コアの pip パッケージ依存（`style-bert-vits2`）のバージョン方針

## 8. 未決事項

なし。本ドキュメント記載の項目はすべて確定済み。
