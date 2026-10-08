# 実操作ガイド（データセット → 学習 → 成果物 → 試聴）

このリポジトリはフォーク元の Style-Bert-VITS2 を日本語・JP-Extra 一本化した「学習コア」です。uv 経由で実行するので、コマンドはすべて `uv run` で実行します。トップレベルの呼び出し口は3つだけ覚えていれば大丈夫です。

| 呼び出し口 | 役割 |
|---|---|
| `preprocess_all.py` | 名前通り**前処理のみ**（Step 1〜6）。学習設定（epochs・batch_size等）を `Data/<モデル名>/config.json` に書き込む |
| `train_model.py` | 学習本体の実行（新規・再開どちらもこれ） |
| `app.py` | 試聴GUI（学習済みモデルを選んで試聴するだけ。学習自体はCLI） |

## 0. インストール

```bash
uv sync
# GPU を使う場合のみ（OS/CUDA に応じた torch を入れる。例: CUDA 12.8）
uv pip install "torch" "torchaudio" --index-url https://download.pytorch.org/whl/cu128
```

なお、日本語BERT・JP-Extra事前学習モデル・SLM のダウンロード（`pretrained/` への取得）は前処理時に自動で行われます。手動でまとめて取得しておきたい場合のみ `uv run -m train.initialize` を実行してください（環境構築用のコマンドであり、モデルごとの運用手順には出てきません）。

## 1. どんなデータセットを受け入れるか（契約）

スライス・ASR文字起こし・正規化まで済んだ「完成済みデータセット」だけを `Data/<モデル名>/` に置きます（音声加工そのものは分離先リポジトリ側の責任）。

```
Data/<モデル名>/
  ├─ esd.list    # 1行1文: <音声ファイル名>|<話者名>|JP|<セリフ文字列>
  └─ raw.zip     # 解凍済み音声（.flac / .wav）。Neutral 1本で使うので基本はフォルダ分け不要
```

- **検証**: 前処理の Step 3（`preprocess/check_dataset.py`）が、全音声のサンプリングレート（44100Hz）とモノラルを自動的にチェックし、不正なファイルがあれば名前付きで即失敗します
- **1話者＝1モデル**です。データセットのディレクトリ名がそのままモデル名になります
- 英語はカタカナ入力が前提です。細かい読みは `dict_data/` のユーザー辞書で調整します

esd.list の例:

```
23d9900d1e68ea43.flac|test|JP|みなさんこんにちは.今回は申し訳ないですが...
```

## 2. どのような流れで学習するのか

前処理（`preprocess_all.py`）→ 学習（`train_model.py`）の2段階です。

```bash
# ① 前処理のみ（Step 1〜6）。epochs・batch_size 等は Data/<モデル名>/config.json に書き込まれる
uv run preprocess_all.py -m <モデル名> [-b 2] [-e 100] [-s 1000]

# ② 学習本体（書き込まれた設定で実行。新規も再開も同じコマンド）
uv run train_model.py -m <モデル名>
```

前処理の各ステップと、決定論的に動くようシード固定済みである点（config の seed: 42）以外は気にしなくていい構造です。

| Step | モジュール | やること | 出力 |
|---|---|---|---|
| 1 | initialize | モデル設定生成・事前学習モデルコピー | `Data/<m>/config.json`、`models/*_0.safetensors` |
| 2 | extract | raw.zip を解凍 | `wavs/*.flac` |
| 3 | check_dataset | サンプリングレート・モノラル検証 | （検証のみ） |
| 4 | preprocess/text | テキスト整形 | `train.list` / `val.list` |
| 5 | preprocess/bert_gen | BERT特徴抽出 | `wavs/<utt>.bert.pt` |
| 6 | preprocess/style_gen | スタイル特徴抽出 | `wavs/<utt>.flac.npy` |

（学習本体は `train/pipeline.py` が Step 1〜6 の成果物を読み、`models/` にチェックポイント・ログ、`model_assets/<m>/` に推論資産を出力します）

主なオプション（`preprocess_all.py -h` でも確認できます）:

| 引数 | デフォルト | 意味 |
|---|---|---|
| `-b, --batch_size` | 2 | バッチサイズ（VRAMが足りないなら下げる） |
| `-e, --epochs` | 100 | エポック数 |
| `-s, --save_every_steps` | 1000 | このステップごとに保存・学習終了 |
| `--val_per_lang` | 4 | 話者ごとの検証データ数（0 で無効化。検証発話は TensorBoard の eval/ に生成音声・正解音声がログされる） |
| `--yomi_error` | raise | 読み上げエラーの扱い（raise / skip / use） |

学習の再開は、同じモデル名で `uv run train_model.py -m <モデル名>` を実行するだけです。スタイル生成は `model_assets/<モデル名>/` に推論資産（config.json + style_vectors.npy）が既にあれば自動でスキップされるため、コマンドラインでの指定は不要です。

スタイルは **Neutral 1本のみが既定**です（Neutral だけでも十分高い精度が出るため）。サブディレクトリごとにスタイルを分けたい場合だけ `uv run train_model.py -m <モデル名> --styles_by_dirs` を明示してください（フォルダ分けしていてもこのフラグが無ければ Neutral として扱われます）。

## 3. 成果物はどこに生成されるか

```
Data/<モデル名>/models/            ← 学習の途中状態・ログ
  ├─ G_*.pth / D_*.pth / WD_*.pth / DUR_*.pth    チェックポイント（オプティマイザ状態込み、再開用）
  └─ events.out.tfevents.*           TensorBoard の学習曲線（`uv run -m tensorboard --logdir Data/<モデル名>/models` で閲覧）

model_assets/<モデル名>/           ← 推論・共有用の成果物（この3点セット）
  ├─ config.json                       そのモデルの設定（設計図）
  ├─ <モデル名>_e<epoch>_s<step>.safetensors  学習済み重み
  └─ style_vectors.npy                 スタイルベクトル（既定は Neutral 1本のみ。--styles_by_dirs 時はフォルダ分けの各スタイル分も）
```

**モデルを共有するときは `model_assets/<モデル名>/` の3点を丸ごと渡してください。** 名前が統一されていない場合は `style_vectors.npy` と config.json も同じセットのものを使う必要があります。

## 4. どのように試聴するのか

### 試聴GUI で（おすすめ）

```bash
uv run app.py [--device cuda] [--port <ポート>] [--share]
```

- 音声合成タブのみ。model_assets/ 配下のモデルを選んで試聴できます。スタイルと強度の選択も可能
- 学習・前処理はコマンドラインで行うため、GUI に学習設定はありません
- 学習曲線（TensorBoard）は別途 `uv run tensorboard --logdir Data/<モデル名>/models` で起動して確認します

### Python ライブラリとして使う

```python
from pathlib import Path
from style_bert_vits2.tts_model import TTSModel

p = Path("model_assets/<モデル名>")
model = TTSModel(
    model_path=next(p.glob("*.safetensors")),
    config_path=p / "config.json",
    style_vec_path=p / "style_vectors.npy",
    device="cuda",  # CPU でも可
    dtype="float16",  # 推論時の重みの精度。既定（フル半精度推論でVRAM節約）。全精度なら "float32"。学習時の既定は float32（`preprocess_all.py --dtype`、fp16学習は不可のため選択不可）
)
sr, audio = model.infer(text="こんにちは", speaker_id=0, style="Neutral")
```

## 補足

- 環境変数 `SBV2_WORKER_PORT`: pyopenjtalk ワーカーのポートを明示指定したい場合のみ（デフォルト7861、使用中なら自動で空きポートを選択）
- 動作確認済み環境: WSL2 / Ubuntu Desktop。GPU が無い環境でも推論（試聴）は可能です
- 回帰テスト: `uv run pytest -v`（前処理→学習1エポック→合成→ゴールデン音声との比較が数分で完了します）。データセット・ゴールデンデータは git に含まれないので、各自で用意します。手順は `tests/data/<モデル名>/`（esd.list + raw.zip）にデータを配置して `uv run pytest -v` を実行するだけでよく、ゴールデン音声（`tests/references/`）は初回実行時に自動作成され、2回目以降の実行で差分検知に使われます。別のデータセットに変えたい場合も `tests/data/` に置いて `SBV2_TEST_MODEL=<モデル名>` を指定するだけです（テスト自体は品質検証ではなく、シード固定した1エポック学習ノイズ混じりの出力が壊れていないことの契約チェックです）
