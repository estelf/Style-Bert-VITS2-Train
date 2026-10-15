# 実操作ガイド（データセット → 学習 → 成果物 → 試聴）

このリポジトリはフォーク元の Style-Bert-VITS2 を日本語・JP-Extra 一本化した「学習コア」です。uv 経由で実行するので、コマンドはすべて `uv run` で実行します。トップレベルの呼び出し口は3つだけ覚えていれば大丈夫です。ユースケース別のコマンドをすぐ引きたいときは [チートシート](../チートシート.md)、設計や既定値の判断の経緯（なぜこうなっているか）は [DECISIONS.md](DECISIONS.md) を参照してください。

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
# ① 前処理のみ（Step 1〜6）。設定は --set で Data/<モデル名>/config.json に書き込まれる
uv run preprocess_all.py -m <モデル名> --set train.batch_size=4 --set train.epochs=100

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
| `--set <dotted.path>=<value>` | 未指定=既存 config.json を維持 | config.json の**任意の設定を1キー上書き**（複数指定可）。値は JSON 解釈（`4`→int, `true`→bool, `0.0001`→float, `[0.8,0.99]`→list, `null`→None、それ以外は文字列）。パス・値は `HyperParameters` スキーマで検証され、打ち間違いはエラーになる |
| `--val_ratio` | 0.05 | **検証発話の比率（既定）**。話者ごとに `round(発話数×ratio)` 件を検証へ回す。0 で検証なし（全件学習。早期停止は機能しなくなる） |
| `--min_val` | 20 | 比率ベースの検証件数の**下限**。ただしその話者の発話数の 1/5 を超えない（検証損失の評価ノイズは 1/sqrt(件数) なので、少なすぎると早期停止の判定が機能しない） |
| `--val_per_lang` | 0 | 検証発話を話者ごとにこの件数へ**固定**する（>0 のとき `--val_ratio` より優先。言語ごとではない。旧来動作の互換フラグ） |
| `--max_val_total` | 100 | 検証発話の総上限。超えた分は**話者按比例で縮小**して学習へ戻る（縮小時は Step 4 で警告される） |
| `--yomi_error` | raise | 読み上げエラーの扱い（raise / skip / use） |
| `--num_processes` | CPUコア数の半分 | スタイル特徴生成の並列プロセス数 |
| `--reset_models` | OFF | 既存 models/ をバックアップし事前学習モデルでリセット（既定は config.json のみ更新） |

検証分割は**話者単位・決定論的**（`config.json` の `train.seed` 既定42 でシードされるため、同じデータなら何度前処理しても同じ分割になる）です。発話1件の話者は検証なし（丸ごと学習側）で、どの話者も学習側に必ず1件以上残ります。検証発話は TensorBoard の `eval/` に生成音声・正解音声がログされ、`train.select_best` の評価対象にもなります。

Step 4 のログに `train/val 分割: 学習 N 件 / 検証 M 件（全体の x%、…）` として確定値が出ます（検証が20件を切った場合は早期停止の判定が信頼できない旨の警告も出ます）。

学習設定は個別フラグではなく `--set` に集約しました（設定の単一ソースは `style_bert_vits2/models/hyper_parameters.py` の `HyperParameters`）。よく使うキー:

- `train.batch_size`（VRAMが足りないなら下げる）
- `train.epochs`（エポック数の**打ち切り上限**。早期停止が既定有効のため、収束すればこの前に自動停止する）
- `train.eval_interval`（このステップごとにチェックポイント・推論資産を保存。学習自体は epochs 上限または早期停止まで継続）
- `train.log_interval`
- `train.freeze_JP_bert` / `train.freeze_style` / `train.freeze_decoder`（`--set train.freeze_style=true`。`--set ...=false` で明示的に戻せる）

前処理の Step 1 は**既存 config.json の設定値を維持**します（既定値に既存値を重ねて更新）。`--set` で指定しなかったキー（learning_rate や early_stopping_* など）の手動編集は再実行でも失われません。まっさらにしたい場合は `Data/<モデル名>/config.json` を削除してください。

学習の再開は、同じモデル名で `uv run train_model.py -m <モデル名>` を実行するだけです。スタイル生成は `model_assets/<モデル名>/` に推論資産（config.json + style_vectors.npy）が既にあれば自動でスキップされるため、コマンドラインでの指定は不要です。`-m` に存在しないモデル名を渡した場合は、ゴミディレクトリを作る前に「学習可能なデータセット」の候補一覧付きでエラー終了します。

再開時のチェックポイント読み込みが失敗した場合（破損・config とモデルの不整合など）、**既定では学習を開始せずエラー終了します**。そのままスクラッチ学習へフォールバックすると、keep_ckpts による旧チェックポイント削除で学習進捗が失われるためです。意図的にスクラッチへ切り替える場合のみ `uv run train_model.py -m <モデル名> --allow_scratch` を明示してください。

BERT特徴キャッシュ（`*.bert.pt`）には生成根拠（テキスト＋word2ph）のハッシュが保存されます。esd.list のリテイクや `dict_data/` 辞書更新で**読みが変わった**状態で学習を開始すると、古い特徴は使わずエラーになるため、`preprocess_all.py` を再実行して Step 5 をやり直してください（変更が無ければキャッシュは再利用され高速です）。

スタイルは **Neutral 1本のみが既定**です（Neutral だけでも十分高い精度が出るため）。サブディレクトリごとにスタイルを分けたい場合だけ `uv run train_model.py -m <モデル名> --styles_by_dirs` を明示してください（フォルダ分けしていてもこのフラグが無ければ Neutral として扱われます）。

### 早期停止と best checkpoint の自動選定

早期停止は **既定で有効** です。`eval_interval` ごとに **val.list の検証損失**（非敵対損失のみ）を計算し、以下の条件（patience / プラトー / 悪化）のいずれかで学習を停止します。設定は `Data/<モデル名>/config.json` の `train`（キーが無い既存 config も Pydantic 既定値で動きます）。

```json
"early_stopping_enabled": true,          // 既定 true。無効化すれば epochs まで回し切る
"early_stopping_monitor": "mel",         // mel | dur | kl | lm | total（lm は WavLM 特徴一致損失。total は mel+dur+kl の和で lm を含まない）
"early_stopping_min_steps": -1,          // 今回の学習開始点（再開時は再開step）からこの経過step以降の評価だけを判定に使う。既定 -1=自動（学習開始時にデータセット規模から max(1000, 5×1エポックのstep数) へ解決）。0 で全履歴対象
"early_stopping_patience": 5,            // best 更新が無い評価が続いたら停止
"early_stopping_min_delta": 0.001,       // これ未満の改善は best 更新とみなさない
"early_stopping_min_delta_rel": 0.01,    // best に対する割合で見る閾値（1%）。実際の判定は max(min_delta, best×この値)。0 で従来動作
"early_stopping_best_smooth": 5,         // best 選定を窓内中央値で行う幅（1 で従来動作の単発最小値）
"early_stopping_action": "stop",         // 停止条件到達時の動作: "stop" で学習停止、"reduce_lr" で学習率を削減して続行（下記）
"early_stopping_reduce_lr_factor": 0.5,  // action="reduce_lr" のときの学習率乗数（0〜1）
"early_stopping_reduce_lr_times": 3,     // 学習率削減の最大回数。使い切ったら停止
"early_stopping_plateau_window": 5,      // プラトー・悪化検出の窓（0 で無効）
"early_stopping_plateau_slope_eps": 0.002 // 窓内回帰傾きが |平均|×この値 以下でプラトー停止。+閾値 を超えたら悪化停止
```

- 監視指標は TensorBoard に `val/*` として記録されます。全履歴は `models/eval_history.json` に保存されますが、**判定には `early_stopping_min_steps` 経過後の評価のみを使い**、それ以前の履歴（再開前を含む）は best・patience・プラトー窓に一切数えません
- **検証損失は復号音声の全文で測ります**（学習のような `segment_size`＝0.37秒のスライス1点ではない）。スライス1点だと測定音声が少なく評価ノイズが収束幅より大きくなって best 判定がコイントスになるためです（実測値は [DECISIONS.md §9.2](DECISIONS.md) 参照）。`kl`・`dur` は元から全文基準、`mel`・`lm` を全文復号で測ります
- 履歴には val.list の内容ハッシュが記録され、**検証セットを変更した場合は旧検証セット上の損失を判定から除外します**（別セットの損失と混ぜて best を誤判定しないため）。同様に測定方法のバージョン（`metric_version`。v1=スライス1点、v2=全文）も記録し、**旧測定の履歴は絶対値が別物なので判定から除外します**
- そのため新規・再開のどちらでも、最低 `min_steps` は学習が積まれてから判定が始まります。既定 `min_steps: -1`（自動）は学習開始時にデータセット規模から `max(1000, 5×1エポックのstep数)` へ解決されます（例: 学習152文・batch_size 2 → 76 step/エポック → 下限の 1000、686文 → 343 step/エポック → 1715）。停止最早は `min_steps + プラトー窓(5)×eval_interval`（プラトー/悪化経路）または `min_steps + (patience+1)×eval_interval`（patience経路）で、既定 eval_interval=1000・解決後 min_steps=1000 なら step 6000 / 7000 から
- **悪化検出**: プラトーと同じ窓の回帰傾きが +閾値（|平均|×`plateau_slope_eps`）を超えたら停止します。ノイズ帯を超えて損失が上昇している run は、best が保存済みでもう得るものがないため早期に打ち切ります（続行するのは傾きが −閾値 未満＝着実な改善中のときのみ）。mel 監視では横ばいに見えて kl・total だけが単調上昇する小規模データセットの過学習で特に効きます
- 停止してもチェックポイント・safetensors は評価ステップ分まで保存済みのまま終了します
- val.list が空（前処理で `--val_ratio 0`）の場合や `--speedup` 指定時は早期停止が機能しないため、**開始時に警告を出してから epochs 上限まで走ります**
- 検証セットを変える（`--val_ratio` などを変えて再前処理する）と旧履歴は判定から自動除外されますが、**既存の重みは新しい検証発話を学習で見てしまっている**ため検証として成り立ちません。検証を意味のあるものにするには `preprocess_all.py --reset_models` で新規学習からやり直してください
- 検証が20件未満の小規模データセットでは評価がノイジーです（ノイズは 1/sqrt(件数) で減る）。`--min_val` を上げるか、早すぎる停止が出るようなら patience を長め（8〜15）に、plateau_window を 0（無効）に調整してください
- **ノイズ対策（既定動作）**: 検証損失は評価ノイズで揺れるため（件数に依存。検証47件の全文評価で評価1点の揺らぎ σ≒0.70%、検証4件なら数%）、patience の改善判定は絶対値 `min_delta` ではなく `max(min_delta, best × min_delta_rel)` の実効閾値で行い、閾値をノイズ帯より広く取ります（既定 1% ≒ 1.4σ）。これがないとノイズの偶発更新で patience が永遠に発火せず epochs 上限まで走り続けます。best チェックポイントも単発の最小値ではなく `early_stopping_best_smooth` 窓内の中央値で選び、偶発的に低く出た評価を best にする「くじ引き」を防ぎます（既定 窓5）。実測による既定値の決め手は [DECISIONS.md §9.3](DECISIONS.md) 参照
- **`early_stopping_action: "reduce_lr"`**: 停止条件に到達しても学習率を `reduce_lr_factor` 倍して続行する ReduceLROnPlateau 相当の動作です。削減回数は `eval_history.json` に記録され、**再開時も自動復元**されます。削減後は patience/プラトーの判定窓もリセットされ、`reduce_lr_times` を使い切った時点で停止します。DDP では全ランクへ broadcast して同一の学習率を維持します
- **`lm` 指標**: WavLM feature matching 損失（実音声と生成音声の WavLM 特徴距離。非敵対）。mel のフレーム対応とは別軸で知覚的近さを測ります。`total` との後方互換のため `total` には含めないので、監視するなら `monitor: "lm"` を明示します（`use_wavlm_discriminator: false` の設定では計算されません）
- 履歴エントリには現在の学習率（`lr`）と削減回数（`lr_reductions`）も記録され、reduce_lr の効果を目視確認できます

監視 best（検証損失上の最良）は毎回 `models/best_checkpoint.json` に記録され、対応する実体は `model_assets/<モデル名>/best.safetensors` として1点コピーされます。`keep_safetensors` で古い eval 資産を間引いても best の実体は残ります。

**best step の選定は通常重みで検証損失を実測して行いますが、`best.safetensors` として書き出す1点は既定で EMA 重みです**（`best_use_ema: true`）。同一 step の通常/EMA 比較で EMA 側が合成品質に優位だったためです（実測は [DECISIONS.md §9.5](DECISIONS.md) 参照）。検証を実測した通常重みは `<モデル名>_e<epoch>_s<step>.safetensors` にそのまま残るので、聞き比べ・`select_best.py` の候補は従来どおり両方あります。`best_checkpoint.json` の `weights` キーが `best.safetensors` の中身（`ema` / `normal`）、`safetensors` / `safetensors_ema` が両実体のファイル名です。EMA 実体が無い step（`ema_decay: 0` など）は自動的に通常重みへフォールバックします。

学習中に `model_assets/<モデル名>/` へ溜まった `<モデル名>_e<epoch>_s<step>.safetensors` 群から推論に使いたい1点を選ぶには、品質メトリクスでの評価スクリプトを使います。

```bash
uv run -m train.select_best -m <モデル名> [--utmos] [--spk-sim] [--stride N] [--len-norm]   # val.list をシード42で合成して評価
```

`mel_corr`（正解音声との log-mel 相関）を基に、`--utmos` で UTMOS（MOS予測）、`--spk-sim` で話者類似度（pyannote wespeaker 埋め込みの cos 類似度、HF トークン必要）を併用でき、順位合計で best を選び `model_assets/<モデル名>/best_selection.json` に保存します。レポートには早期停止が記録した `best_checkpoint.json`（検証損失 best）も併記され、合成品質 best との一致・不一致を確認できます。

- `--stride N`: 候補が多い長期学習で N 件おきに間引いて評価時間を短縮（最終 step は必ず評価対象に含まれる）
- `--len-norm`: mel_corr を比較前に両 log-mel を同一フレーム数へ線形補間してから計算する。既定の mel_corr は発話長がずれると（無音の長さ差だけでも）不利に下がるため、スペクトル軌道の形状だけを比較したいときに有効（代わりに生成が極端に短い/長い劣化は検出できなくなる）

### 学習の高度な設定（config.json `train`）

既定値のまま従来動作で、必要に応じて調整するキーです（`ema_decay` と `best_use_ema` は既定で有効）。

```json
"accumulation_steps": 1,          // 勾配累積ステップ数。VRAM 不足で batch_size を上げられない時の実効バッチ拡大用（DDP の no_sync で中間ステップの通信を省略）
"adversarial_warmup_steps": 0,    // 敵対損失（fm/gen/dur_gen/lm 系）を学習開始から線形に 0→1 スケールするステップ数。判別器未熟期の暴れ防止。0 で無効
"ema_decay": 0.999,               // 生成器パラメータの EMA 減衰率（0 で無効）。評価・最終保存に `<モデル名>_e<epoch>_s<step>_ema.safetensors` が併記される。早期停止の判定は検証損失を実測している通常重み側で行う
"best_use_ema": true,             // best.safetensors を best step の EMA 重みからコピーする（best step の選定は通常重みの検証損失のまま）。false で従来動作＝検証を実測した通常重みをコピー
"keep_safetensors": 6,            // model_assets/ の .safetensors を step 単位で保持する個数。0 で全残し（best.safetensors と best の step は常に保持）。eval_interval=1000 ごとに通常+EMA の2枚（約500MB）増えるため、長期学習の disk 圧迫を防ぐ
"nan_guard_max_skips": 10         // 非有限損失（NaN/inf）がこの回数連続で出たら学習停止。発生前の重みは保持され、検出ステップは更新をスキップ。0 で無効
```

- **`global_step` はオプティマイザ更新数ではなくマイクロバッチ（forward-backward）回数**です。ログ・チェックポイント名・`eval_interval` / `early_stopping_min_steps` などのステップ基準はすべてこのマイクロバッチ単位で、`accumulation_steps > 1` の実行では実オプティマイザ更新数は `global_step / accumulation_steps` になります（`accumulation_steps: 1` の既定では両者は一致）

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
  ├─ best.safetensors                  監視 best のコピー（既定は EMA 重み。keep_safetensors で間引かれても残る）
  └─ style_vectors.npy                 スタイルベクトル（既定は Neutral 1本のみ。--styles_by_dirs 時はフォルダ分けの各スタイル分も）
```

**モデルを共有するときは `model_assets/<モデル名>/` の3点を丸ごと渡してください。** 名前が統一されていない場合は `style_vectors.npy` と config.json も同じセットのものを使う必要があります。

実行ログ（`Data/<モデル名>/train_<日時>.log` と `preprocess_<日時>.log`）は実行のたびに増え続けないよう、loguru の retention で**直近10本だけ自動保持**されます（古いものから削除）。

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
