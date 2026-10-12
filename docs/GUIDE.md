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
uv run preprocess_all.py -m <モデル名> [-b 4] [-e 100] [-s 1000]

# ② 学習本体（書き込まれた設定で実行。新規も再開も同じコマンド）
uv run train_model.py -m <モデル名>
```

前処理の各ステップと、シード固定済みである点（config の seed: 42）以外は気にしなくていい構造です。学習の決定論モード（cuDNN の決定論カーネル強制）は速度ペナルティがあるため本番では既定 OFF で、回帰テストのみ環境変数 `SBV2_DETERMINISTIC=1` で有効化します。

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
| `-b, --batch_size` | 未指定=既存 config.json を維持 | バッチサイズ（VRAMが足りないなら下げる） |
| `-e, --epochs` | 未指定=既存 config.json を維持 | エポック数の**打ち切り上限**（早期停止が既定有効のため、収束すればこの前に自動停止する） |
| `-s, --save_every_steps` | 未指定=既存 config.json を維持 | このステップごとにチェックポイント・推論資産を保存（学習自体は epochs 上限または早期停止まで継続） |
| `--val_per_lang` | 4 | 話者ごとの検証データ数（0 で無効化。検証発話は TensorBoard の eval/ に生成音声・正解音声がログされる） |
| `--yomi_error` | raise | 読み上げエラーの扱い（raise / skip / use） |
| `--freeze_JP_bert` / `--freeze_style` / `--freeze_decoder` | 未指定=既存 config.json を維持 | 指定したときだけ true になる。false に戻すには config.json を直接編集 |

前処理の Step 1 は**既存 config.json の設定値を維持**します（テンプレートに既存値を重ねて更新）。learning_rate や early_stopping_* など CLI 引数に対応しないキーの手動編集は再実行でも失われません。まっさらにしたい場合は `Data/<モデル名>/config.json` を削除してください。

学習の再開は、同じモデル名で `uv run train_model.py -m <モデル名>` を実行するだけです。スタイル生成は `model_assets/<モデル名>/` に推論資産（config.json + style_vectors.npy）が既にあれば自動でスキップされるため、コマンドラインでの指定は不要です。

再開時のチェックポイント読み込みが失敗した場合（破損・config とモデルの不整合など）、**既定では学習を開始せずエラー終了します**。そのままスクラッチ学習へフォールバックすると、keep_ckpts による旧チェックポイント削除で学習進捗が失われるためです。意図的にスクラッチへ切り替える場合のみ `uv run train_model.py -m <モデル名> --allow_scratch` を明示してください。

BERT特徴キャッシュ（`*.bert.pt`）には生成根拠（テキスト＋word2ph）のハッシュが保存されます。esd.list のリテイクや `dict_data/` 辞書更新で**読みが変わった**状態で学習を開始すると、古い特徴は使わずエラーになるため、`preprocess_all.py` を再実行して Step 5 をやり直してください（変更が無ければキャッシュは再利用され高速です）。

スタイルは **Neutral 1本のみが既定**です（Neutral だけでも十分高い精度が出るため）。サブディレクトリごとにスタイルを分けたい場合だけ `uv run train_model.py -m <モデル名> --styles_by_dirs` を明示してください（フォルダ分けしていてもこのフラグが無ければ Neutral として扱われます）。

### 早期停止と best checkpoint の自動選定

早期停止は **既定で有効** です。`eval_interval` ごとに **val.list の検証損失**（非敵対損失のみ）を計算し、以下の条件のどちらかで学習を停止します。設定は `Data/<モデル名>/config.json` の `train`（キーが無い既存 config も Pydantic 既定値で動きます）。

```json
"early_stopping_enabled": true,          // 既定 true。無効化すれば epochs まで回し切る
"early_stopping_monitor": "mel",         // mel | dur | kl | total
"early_stopping_min_steps": 3000,        // 今回の学習開始点（再開時は再開step）からこの経過step以降の評価だけを判定に使う。0 で全履歴対象
"early_stopping_patience": 10,           // best 更新が無い評価が続いたら停止
"early_stopping_min_delta": 0.001,       // これ未満の改善は best 更新とみなさない
"early_stopping_plateau_window": 5,      // プラトー検出の窓（0 で無効）
"early_stopping_plateau_slope_eps": 0.002 // 窓内回帰傾きが |平均|×この値 以下で停止
```

- 監視指標は TensorBoard に `val/*` として記録されます。全履歴は `models/eval_history.json` に保存されますが、**判定には `early_stopping_min_steps` 経過後の評価のみを使い**、それ以前の履歴（再開前を含む）は best・patience・プラトー窓に一切数えません
- 履歴には val.list の内容ハッシュが記録され、**検証セットを変更した場合は旧検証セット上の損失を判定から除外します**（別セットの損失と混ぜて best を誤判定しないため）
- そのため新規・再開のどちらでも、最低 `min_steps` は学習が積まれてから判定が始まります（既定 eval_interval=1000 なら、停止最早は新規で step 8000=プラトー経路 / 14000=patience経路、step 3100 からの再開なら 11000 / 17000）
- 停止してもチェックポイント・safetensors は評価ステップ分まで保存済みのまま終了します
- val.list が空（前処理で `--val_per_lang 0`）の場合や `--speedup` 指定時は早期停止が機能しないため、**開始時に警告を出してから epochs 上限まで走ります**
- val が数文しかない小規模データセットでは評価がノイジーなので、早すぎる停止が出るようなら patience を長め（8〜15）に、plateau_window を 0（無効）に調整してください

監視 best（検証損失上の最良）は毎回 `models/best_checkpoint.json` に記録され、対応する `.safetensors` は `model_assets/<モデル名>/best.safetensors` としても1点コピーされます。`keep_safetensors` で古い eval 資産を間引いても best の実体は残ります。

学習中に `model_assets/<モデル名>/` へ溜まった `<モデル名>_e<epoch>_s<step>.safetensors` 群から推論に使いたい1点を選ぶには、品質メトリクスでの評価スクリプトを使います。

```bash
uv run -m train.select_best -m <モデル名> [--utmos] [--spk-sim]   # val.list をシード42で合成して評価
```

`mel_corr`（正解音声との log-mel 相関）を基に、`--utmos` で UTMOS（MOS予測）、`--spk-sim` で話者類似度（pyannote wespeaker 埋め込みの cos 類似度、HF トークン必要）を併用でき、順位合計で best を選び `model_assets/<モデル名>/best_selection.json` に保存します。レポートには早期停止が記録した `best_checkpoint.json`（検証損失 best）も併記され、合成品質 best との一致・不一致を確認できます。

### 学習の高度な設定（config.json `train`）

既定値のまま従来動作で、必要に応じて調整するキーです（`ema_decay` のみ既定で有効）。

```json
"accumulation_steps": 1,          // 勾配累積ステップ数。VRAM 不足で batch_size を上げられない時の実効バッチ拡大用（DDP の no_sync で中間ステップの通信を省略）
"adversarial_warmup_steps": 0,    // 敵対損失（fm/gen/dur_gen/lm 系）を学習開始から線形に 0→1 スケールするステップ数。判別器未熟期の暴れ防止。0 で無効
"ema_decay": 0.999,               // 生成器パラメータの EMA 減衰率（0 で無効）。評価・最終保存に `<モデル名>_e<epoch>_s<step>_ema.safetensors` が併記される。best.safetensors と早期停止の判定は検証損失を実測している通常重み側で行うため、_ema は聞き比べ・select_best 用の候補となる
"keep_safetensors": 0,            // model_assets/ の .safetensors を step 単位で保持する個数。0 で全残し（best.safetensors と best の step は常に保持）
"nan_guard_max_skips": 10         // 非有限損失（NaN/inf）がこの回数連続で出たら学習停止。発生前の重みは保持され、検出ステップは更新をスキップ。0 で無効
```

## 3. 成果物はどこに生成されるか

```
Data/<モデル名>/models/            ← 学習の途中状態・ログ
  ├─ G_*.pth / D_*.pth / WD_*.pth / DUR_*.pth    チェックポイント（オプティマイザ状態込み、再開用）
  ├─ eval_history.json               検証損失の全履歴（早期停止の判定単位。val.list のハッシュ付き）
  ├─ best_checkpoint.json            監視指標上の best（どの .safetensors を推論に使うべきか）
  └─ events.out.tfevents.*           TensorBoard の学習曲線（見方は [TENSORBOARD.md](TENSORBOARD.md)）

model_assets/<モデル名>/           ← 推論・共有用の成果物（この3点セット）
  ├─ config.json                       そのモデルの設定（設計図）
  ├─ <モデル名>_e<epoch>_s<step>.safetensors  学習済み重み（ema_decay 有効時は *_ema.safetensors も併記）
  ├─ best.safetensors                  監視 best のコピー（keep_safetensors で間引かれても残る）
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
- 学習曲線（TensorBoard）は別途 `uv run tensorboard --logdir Data/<モデル名>/models` で起動して確認します。ログの見方は [TENSORBOARD.md](TENSORBOARD.md) を参照

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
    dtype="float16",  # 推論時の重みの精度。既定（フル半精度推論でVRAM節約）。全精度なら "float32"。学習は常に float32（bf16/fp16 学習経路は廃止済み）
)
sr, audio = model.infer(text="こんにちは", speaker_id=0, style="Neutral")
```

## 補足

- 環境変数 `SBV2_WORKER_PORT`: pyopenjtalk ワーカーのポートを明示指定したい場合のみ（デフォルト7861、使用中なら自動で空きポートを選択）
- 環境変数 `SBV2_DATASET_ROOT`: データセットルート（デフォルト `Data/`。回帰テストは `tests/data` を使う）
- 環境変数 `SBV2_ASSETS_ROOT`: 学習済みモデル資産のルート（デフォルト `model_assets/`。回帰テストは `tests/model_assets` を使い、実モデルを絶対に触らない）
- 環境変数 `SBV2_DETERMINISTIC=1`: 学習の決定論モードを有効化（ゴールデン音声比較用。本番学習は速度優先で既定 OFF）
- 動作確認済み環境: WSL2 / Ubuntu Desktop。GPU が無い環境でも推論（試聴）は可能です
- 回帰テスト: `uv run pytest -v`（前処理→学習1エポック→合成→ゴールデン音声との比較が数分で完了します）。データセット・ゴールデンデータは git に含まれないので、各自で用意します。手順は `tests/data/<モデル名>/`（esd.list + raw.zip）にデータを配置して `uv run pytest -v` を実行するだけでよく、ゴールデン音声（`tests/references/`）は初回実行時に自動作成され、2回目以降の実行で差分検知に使われます。別のデータセットに変えたい場合も `tests/data/` に置いて `SBV2_TEST_MODEL=<モデル名>` を指定するだけです（テスト自体は品質検証ではなく、シード固定した1エポック学習ノイズ混じりの出力が壊れていないことの契約チェックです）
