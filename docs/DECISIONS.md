# 判断の経緯（設計・既定値の来歴）

この文書は **「なぜ現状の実装がこうなっているか」の経緯** を記録する。操作手順は [GUIDE.md](GUIDE.md)、既定値の単一ソースは `style_bert_vits2/models/hyper_parameters.py`、ユースケース別のコマンドは [チートシート](../チートシート.md) を参照。運用ドキュメントやコードコメントには「今の挙動」だけを書き、実測データ・旧既定値・失敗談はこの文書に集約する。既定値や設計を変えるときは、ここで来歴を確認して**同じ轍を繰り返さないこと**を推奨する。

日付はコミット日時（上流フォーク起点: 2025-08-24、分割以降の独自開発: 2026-10〜）。

---

## 1. フォークと分割（2026-10-07, d5e5f45）

- **現状**: 上流 [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)（v2.7.0）のフォークを「日本語・JP-Extra 一本化の学習コア」として再編。データセット作成（スライス・ASR・リサンプリング）・マージ・評価・アプリ推論（エディター/API サーバー）は分離先リポジトリへ移し、ONNX・.bat・Colab は廃止。
- **経緯**: 方針・削除対象一覧・分離先3リポジトリの契約は [SPLIT_PLAN.md](SPLIT_PLAN.md) に記録済み（歴史記録として保存）。計画と最終実装の差分（学習呼び出し口・config 管理・前処理ステップ数）も同文書の冒頭にまとめた。
- **レイアウト**: `src/` ではなくフラットレイアウト。`style_bert_vits2/`（コアライブラリ）/ `train/`（学習パイプライン）/ `preprocess/`（前処理チェーン）にパッケージ化し、上流の `text_module/`・`configs/`・`train_ms*.py` は消滅。トップレベルの呼び出し口は `preprocess_all.py` / `train_model.py` / `app.py` の3つに固定した（「順序や状態ファイルに依存しない冪等設計」を契約とするため）。

## 2. JP-Extra 一本化・日本語のみ（2026-10-07）

- **現状**: 学習・推論は `models/models_jp_extra.py`（2.0-JP-Extra）のみ。v2.1 アーキテクチャ `models/models.py` と `infer.py` の v2.1 分岐は削除。EN/ZH の nlp モジュール・依存（`g2p_en`/`nltk`/`jieba`/`pypinyin`/`cn2an` 等）・`bert_models.json` の EN/ZH エントリも削除し、全入力を日本語経路に流す。
- **経緯**: 手元の既存チェックポイントが全て JP-Extra 系であることを確認済みで、「モデルを再学習せず既存チェックポイントそのまま使う」方針が確定していたため（SPLIT_PLAN §1-4）。v2.1 対応を残すコストが回収できないと判断。
- **`symbols.py` の多言語トークン表を残している理由**: トークン表・トーンオフセット・`LANGUAGE_ID_MAP` は既存チェックポイントに焼き込まれている（`language_emb = nn.Embedding(3, ...)` が依存）。再学習なしの方針では削除不可。純日本語データでは EN/ZH 特徴は常にゼロ埋めで学習時と同一になるため互換は保たれる（SPLIT_PLAN §4）。
- **英語混在テキスト**: カタカナ入力必須＋ユーザー辞書（`dict_data/`）で吸収。自動転写レイヤーは追加しない（SPLIT_PLAN §1-5）。ローマ字のままの英単語は pyopenjtalk に1文字ずつ読まれるため注意。

## 3. G2P: pyopenjtalk → pyopenjtalk-plus（2026-10-08, be1969d）

- **現状**: 依存は `pyopenjtalk-plus[onnxruntime,tsqyomi]`。`import pyopenjtalk` のまま動くドロップイン代替。
- **経緯**: 旧 pyopenjtalk-dict は numpy 2.x 環境でソースビルド（cmake / cython>=3.1 / setuptools<81 のピン）が必要で環境構築の失敗要因になっていた。pyopenjtalk-plus は numpy 2.x 向け事前ビルド済み wheel が全プラットフォームに配布されており、ビルド制約が消えた。
- **extras の意味**: `tsqyomi` は同形異音語の文脈読み選択、`onnxruntime` は「何」読み推定。両方既定の依存に含める（extras にすると `uv run` の自動 sync で黙って除去される問題があったため、そもそも extras に置かない。§4 参照）。

## 4. 環境・パッケージ管理（uv）

- **現状**: `pyproject.toml` + `uv.lock` に一元化（requirements*.txt 全廃）。すべて `uv run ...` 経由で実行する。バージョンは `[project].version` が単一のソースで、`constants.VERSION` は `importlib.metadata.version()` から読む（uv_build は動的バージョン非対応のため静的指定のまま）。
- **dependency-groups に置いた理由（陷阱の記録）**: 実行に必要な `gui`（gradio）と `train`（pyopenjtalk 由来の pyannote.audio）を当初 **extras** に置いていたが、`uv run` の自動 sync が「既定エクストラ＝無し」で環境を完全一致させ、**gradio と pyopenjtalk-plus を黙って除去して壊れた**。`import gradio` が namespace package として成功してしまうため破損が実行時まで表面化しない。extras は「入れない選択ができる」ことが壊れる方向に効いたため、`[dependency-groups]` + `default-groups = ["dev", "gui", "train"]`（既定で必ず入る）に移して再発を防いでいる。
- **長時間ジョブの癖**: 学習実行中の `uv run` 自動 sync は環境を変える可能性があるため `--no-sync` を使う。破損が疑わしいときは `uv pip install --reinstall --no-deps <pkg>`（torch を巻き込まない）。
- **torch**: PyPI の CUDA ビルドがそのまま入る構成。OS/CUDA に合わせて `uv pip install "torch" "torchaudio" --index-url ...` で差し替える（cu128 等）。

## 5. 設定の単一ソース（config 管理）

- **現状**: 全設定の SSOT は `style_bert_vits2/models/hyper_parameters.py` の Pydantic 既定値。前処理 Step1 が `default_config_dict()` から `Data/<モデル>/config.json` を生成し、ロード時は欠損キーに既定値がフェイルセーフとして効く。CLI からの設定上書きは `preprocess_all.py --set dotted.path=value` に集約（パス・値はスキーマで検証、打ち間違いはエラー）。
- **経緯**: 旧 `configs/config_jp_extra.json` テンプレートは値が Pydantic へ全て移ったため削除（テンプレ二重管理の解消）。個別 CLI フラグ（`--epochs` 等）はキーの増減ごとにスクリプト改修が必要で、テンプレとの二重管理の温床にもなったため `--set` 1本に寄せている。`--set batchsize=4` のような打ち間違いが黙って無視されないことも検証の目的に含める。
- **既存 config.json の値を維持する設計**: 前処理の再実行で手編集（learning_rate や early_stopping_* など）が失われないよう、Step1 は「既定値へ既存値を重ねる」。epochs を変えたいだけで学習がやり直しにならないよう、既存チェックポイントがある場合は config.json のみ更新し models/ は触らない（リセットは `--reset_models` 明示時のみ）。
- **`--set` 禁止キー**: model_name・データセット派生パス・version は毎回自動決定されるため、手動指定と衝突したら自動値が最優先で上書きする（紛らわしいので明確に弾く）。config の `version` は推論時の互換判定に使うモデルフォーマット識別子で、パッケージバージョンから生成して手動同期を無くした。

## 6. データセット契約と検証

- **契約**: `Data/<モデル名>/` に `esd.list`（`<ファイル名>|<話者>|JP|<セリフ>`）＋ `raw.zip`。スライス・ASR・正規化は分離先リポジトリの責任で、コアは完成済みデータセットだけを受け取る。
- **尺検証は警告のみ（2026-10-09, 40c6139）**: 旧実装はバケット境界外（約0.37秒未満・11.6秒超）をエラーにしていたが、実際には DistributedBucketSampler が境界外を黙って除外するだけで学習は成立するため、前処理段階では警告に格下げ（エラーにすると1本でも混ざったデータセットが前処理不能になっていた）。バケット境界は `constants.py` の `MIN_SPEC_FRAMES`/`MAX_SPEC_FRAMES` が単一ソースで、check_dataset・サンプラー・評価除外が同じ値を参照する。
- **バケット長を実フレーム数基準に（同上）**: 旧実装はファイルサイズからの推定（16bit PCM WAV 前提）で、FLAC では実フレーム数とズレてバケット分けが崩れていた。`_spec_length()` がヘッダのみ読んで `center=False` STFT の実出力フレーム数を再現する（出力フレーム数は窓長でなく n_fft で決まる。torch.stft で実測確認）。
- **zip 展開の path traversal 対策**: 絶対パスや `..` を含むエントリは `wavs/` の外へ書けるため拒否（悪意ある zip からのファイル上書き防止）。
- **音声差し替え時のキャッシュ整合**: extract_raw は中身が同一なら上書きしない（`.spec.pt` 等の派生キャッシュを生かす）。変わったファイルは古い派生キャッシュを削除してから展開する。実ファイル名の規約は `X.spec.pt`/`X.mel.pt`/`X.bert.pt`（with_suffix 置換）と `X.flac.npy`（文字列追記）の2系統があり、混同すると存在しない `X.flac.spec.pt` を生成して古い spec が黙って再利用され続ける（音声リテイク時の誤学習防止のため命名どおりに削除）。

## 7. キャッシュと再現性

- **BERT 特徴キャッシュ（`*.bert.pt`）**: 生成根拠（テキスト＋word2ph）のハッシュを添えて保存。esd.list のリテイクや辞書更新で読みが変わった状態で学習を開始すると、古い特徴で音素と BERT 特徴が食い違うため**学習側でも照合して明示的に失敗する**（学習中に再生成はできないので Step 5 再実行を要求）。ハッシュは list に書かれた生の word2ph（intersperse 前）から作る規約で、bert_gen と data_utils で共有する。
- **data_utils の word2ph 倍化削除**: 旧実装は get_text 内で word2ph を倍化していたが ja_bert は bert_gen 時点で音素長へ展開済みであり死コード。しかも dataset のリストを in-place で破壊するため、バケットの batch 倍数パディングで生じる重複サンプルを同じワーカーが2回読むと2回目のハッシュ計算が倍化済み word2ph で行われ誤 stale 検出になった（実データで再現）。不変性を保証するため削除。
- **スペクトルキャッシュ（`.spec.pt`）**: キーに STFT 設定と音声内容の sha1 を含める。Step 2 を通さずに wavs/ を差し替えた（リテイク）ときに古い spec が黙って使われ続けるのを防ぐため。音声はキャッシュ有無に関係なく毎回ロード済みなので sha1 は STFT 再計算に比べ無視できるコスト。保存は tmp→rename のアトミック（書き込み中のクラッシュで壊れたキャッシュが全エポック再計算扱いになるのを防ぐ）。
- **検証分割の決定論性**: 話者単位・`train.seed`（既定42）でシードするため、同じデータなら何度前処理しても同じ分割になる（再開時に検証セットが勝手に変わって早期停止履歴が壊れるのを防ぐ）。
- **決定論モード**: `SBV2_DETERMINISTIC=1` のときのみ `use_deterministic_algorithms(True, warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG`。回帰テストのゴールデン音声比較の前提だが、決定論カーネルは遅いため本番学習は既定 OFF。`warn_only=True` は torch 更新で新規 op が非決定扱いになったとき学習・テスト自体が RuntimeError で止まらないため（回帰テストは相関ベースの差分検知でビット単位一致を要求しない）。
- **評価の決定論性**: 検証損失の計算は RNG 状態を退避してシードを固定し、MAS ノイズを 0 にする（評価間で値を比較できるようにするため）。学習ループの乱数系列は戻す。

## 8. 検証分割の比率化

- **現状**: 既定は比率ベース（`--val_ratio 0.05`、話者ごとに `round(発話数×ratio)`、下限 `--min_val 20` ただし発話数の1/5まで、総上限 `--max_val_total 100` 超過分は話者按比例で縮小）。`--val_per_lang` は件数固定の旧来互換フラグ（>0 で比率より優先）。
- **経緯**: 上流の「話者あたり固定10件」方式は、大規模データセットでは検証が少なすぎて早期停止の判定が機能せず、小規模では検証に食われすぎた。検証損失の評価ノイズは 1/sqrt(件数) で減るため、比率＋下限の設計に変更した（2026-10-14, abce088）。発話1件の話者は検証なし（丸ごと学習側）、どの話者も学習側に必ず1件以上残る。検証が20件を切った場合は Step 4 で警告を出す（早期停止の best 判定がくじ引きになるレベル）。

## 9. 早期停止と best checkpoint 選定の進化

段階的に強化してきた経緯（詳細な実測値は各段階の動機記録）。

### 9.1 導入（2026-10-12, 7ce5c2b / PR#5）

- 検証損失（非敵対損失のみ）による早期停止を**既定有効**で追加。`eval_interval` ごとに評価し、patience / プラトー検出（窓内回帰傾き）で停止。`eval_history.json`（全履歴）と `best_checkpoint.json`（監視 best）を記録、`model_assets/` へ `best.safetensors` を1点コピー。
- 同時期に EMA（生成器）、勾配累積、敵対損失ウォームアップ、NaN ガード、`keep_safetensors` を追加。
- 当時の既定: `min_steps=3000`、`patience=10`、`min_delta=0.001`（絶対値のみ）、best は単発最小値、検証損失は **segment_size スライス1点**で測定。

### 9.2 検証損失の全文評価（2026-10-14, abce088 / PR#6）

- **問題（実測）**: スライス1点測定では測定される音声が「検証件数×0.37秒」分しかなく、検証4件・9000step のランで評価間 1σ≒0.19 あり、best(22.6078) と停止時(22.6240) の差 0.016 を区別できなかった。**best 判定がコイントス**になっていた。
- **対策**: mel・lm を復号音声の全文で測る（kl・dur は元から全文基準）。測定音は検証4件でも 1.5秒→8.1秒（5.4倍）、発話間のばらつき sd は 2.13→1.44 に低下。全文復号は推論時の挙動そのものでもあり、スライス損失より合成品質との対応が良い。
- **履歴の互換性**: 測定方法のバージョン（`metric_version`: v1=スライス、v2=全文）を履歴に記録し、旧測定の履歴は絶対値が別物なので判定から除外（v1 の 22.6 相当と v2 の 21.9 相当を同じ best 比較に使うと壊れる）。同様に val.list の内容ハッシュを記録し、検証セット変更後は旧検証セット上の損失を判定から除外する。

### 9.3 ノイズ対策としての実効閾値と中央値 best（同上）

- **patience の実効閾値**: 検証47件の全文評価で評価1点の揺らぎ σ≒0.70%（検証4件なら数%）と実測。絶対値 `min_delta=0.001` だけだと閾値がノイズ帯より狭く、patience が永遠に発火せず epochs 上限まで走った（0.5% は 0.7σ で約4件に1件の割合で運だけの best 更新が起きていた実測あり）。`max(min_delta, best × min_delta_rel)`（既定 1% ≒ 1.4σ）の実効閾値で、ノイズ帯より広い改善だけを best 更新とみなすようにした。
- **best の中央値スムージング**: 単発の最小値だと「偶発的に低く出た評価」を best に選ぶくじ引きになる。窓内中央値（既定5）で近傍ごと安定して低い step を選ぶ。窓3でも運の要素が残ったため実測ノイズに合わせて5にした。

### 9.4 自動 min_steps・patience 5・悪化検出（2026-10-14 以降、作業ツリー）

- **`min_steps: -1`（自動）**: 旧固定 3000 は大規模データ（686文）で9〜13エポック相当と妥当だったが、小規模（152文・batch_size 2）では39エポック以上に相当し、判定開始を待っている間に kl 単調上昇の過学習が進行済みだった実測がある（152文の小規模run）。学習開始時にデータセット規模から `max(1000, 5×1エポックのstep数)` へ解決する方式に変更（152文→1000、686文→1715）。5エポックは回帰傾きがプラトー閾値を大きく超える急速改善期を越えつつ過学習が深まりすぎる前の妥協点。下限1000は極小データセットで判定窓（プラトーに評価5回分必要）が短くなりすぎるのを防ぐ。
- **`patience: 10 → 5`**: 実測6 run すべてで min_steps 経過後の判定対象評価が5〜8回しかなく、counter は最大6〜7。旧既定10は**一度も発火不能**で全 run がプラトー停止だった。悪化停止と並ぶ安全網なので、短めの run でも発火できる値に下げた。
- **悪化検出**: 旧実装は |傾き|≦閾値（プラトー）のみ停止で、ノイズ帯を超えて**悪化**する run はプラトー判定で止まらなかった。評価回数の少ない小規模データセットでは patience も発火しにくく、mel 監視では横ばいに見えて kl・total だけが単調上昇する過学習（小規模run2件の実測）が epochs 上限まで走る空白地帯があった。best は保存済みなので悪化中の続行に益はなく、窓内回帰傾きが +閾値 を超えたら停止するようにした（続行するのは傾きが −閾値 未満＝着実な改善中のときのみ）。
- **`keep_safetensors: 12 → 6`**: 旧既定12の実測で6モデル約28GB（1モデル約5GB）に膨らんだため半減。best が停止の3〜6k step 前でも best 実体は保持されることはテスト済み。

### 9.5 best.safetensors の EMA 化（2026-10-14, 6ca4137 前後）

- best step の**選定**は通常重みで検証損失を実測して行うが、書き出す1点は既定で EMA 重み（`best_use_ema: true`）。同一 step の通常/EMA 比較で EMA 側が合成品質に優位だったため（実測run: 評価8候補すべてで UTMOS が上・平均 +0.27、`mel_corr` も 7/8 で上）。検証を実測した通常重みは `<モデル名>_e<epoch>_s<step>.safetensors` にそのまま残り、聞き比べ・select_best の候補は従来どおり両方ある。EMA 実体が無い step（`ema_decay: 0` 等）は自動で通常重みへフォールバック。

### 9.6 監視指標の設計判断

- **敵対損失を監視に使わない**: 判別器の状態に依存して単調でなくなるため。
- **`total` に lm を含まない**: lm は `use_wavlm_discriminator` の on/off で計算されないため、total に入れると実行によって total の意味が変わり、eval_history をまたいだ早期停止判定が壊れる。lm を監視したい場合は `monitor: "lm"` を明示する。mel は c_mel（既定45）倍で桁が大きいので total は実質 mel 支配。dur 変化だけを追いたいなら `monitor: "dur"`。
- **`lm`（WavLM feature matching）**: 敵対的ではない「実音声と生成音声の WavLM 特徴距離」。mel のフレーム対応とは別軸で知覚的近さを測る。
- **検証セットを変えたら重みもやり直し**: 旧履歴は判定から自動除外されるが、既存の重みは新しい検証発話を学習で見てしまっており検証として成り立たない。`--reset_models` で新規学習からやり直す運用（GUIDE 参照）。

## 10. 学習パイプラインの堅牢化・高速化

- **再開失敗は既定で停止（`--allow_scratch` で明示フォールバック）**: 再開時のチェックポイント読み込み失敗でスクラッチへ握り潰すと、keep_ckpts=1（既定）では次の保存後の clean_checkpoints が旧チェックポイントを削除し学習進捗が失われる。checkpoints.py の MAX_MISMATCHED_KEYS ガードの意図（大規模不一致を黙って通さない）を機能させるため、既定は停止に倒した。
- **事前学習モデル読み込み失敗も停止**: 存在するのに読めない（破損・不整合）ときにランダム初期化で黙って続行すると、壊れた土台で学習が進み誰も気づかないままチェックポイントが上書きされていく。
- **NaN ガード**: 非有限損失は backward しても勾配がゴミなので、**グラフを保ったまま値だけゼロにして** backward する（DDP Reducer の整合性維持）。実行・不実行がランク間で分かれると集合通信が不一致でハングするため、有限性判定は必ず all_reduce 経由で全ランクへ共有する。連続回数が閾値（既定10）を超えたら停止（発生前の重みは保持）。
- **DDP timeout 300秒→30分**: 評価ステップでは rank 0 だけが検証音声の生成＋チェックポイント4点保存を行い、他ランクは次の集合通信で待つ。検証文が多い・ディスクが遅い環境では 300 秒では足りず学習中盤で NCCL タイムアウト落ちしたため拡大。
- **NCCL デバイスの早期確定**: スタイル生成後の barrier（最初の集合通信）より前に `torch.cuda.set_device()` を呼ぶ。後回しだと全ランクがデフォルトの cuda:0 で communicator を作ろうとして失敗・ハングする。
- **MASTER_PORT / MASTER_ADDR**: 既定ポート固定だと同時実行する2本目が衝突するため、未設定のときだけ空きポートを選ぶ（close→bind の TOCTOU の窓は単一プロセス起動のみ再試行で解消）。既定 MASTER_ADDR は IPv6 ループバック `::1`（WSL2 mirrored mode でも確実に bind/connect できるため localhost でなくリテラル指定）。IPv6 自体が使えない環境では 127.0.0.1 へフォールバック（`_free_port` は IPv4 にフォールバックするのにアドレスが ::1 のままだと不整合になる）。Windows は gloo バックエンド。
- **ログ**: マルチGPUでは全ランクが同じログファイルを掴んで混線するため rank 0 だけがファイルへ書く。実行のたびに増え続けないよう loguru の retention で直近10本だけ保持（ファイル名の日時は loguru の `{time}` マクロで書くこと。Python 側で整形した文字列だとパターンを認識されず retention が効かない）。
- **`gradient_as_bucket_view=True` + `zero_grad(set_to_none=False)`**: 勾配を allreduce バッファの実体に共有して VRAM を節約する組み合わせ。ビューが壊れるため zero_grad は必ずゼロ埋めで呼ぶ（勾配テンソルを None にして作り直すとバケットビューの対応が壊れ、torch 側も毎イテレーションでビューが破棄され節約効果が消えると明記）。
- **WavLM forward 統合（2026-10-10, eca78e1）**: 旧実装は forward()/generator()/discriminator() がそれぞれ独立に WavLM を forward していて倍計算だった。判別器ステップで実音声（no_grad）・生成音声（勾配なし）を1回ずつに統合し、生成器側は勾配を流すため生成音声のみ再 forward（no_grad 版とは別計算になるため不可避）。
- **SLM 敵対損失の未加算修正（同上）**: JP-Extra 既定は `use_duration_discriminator: false` なのに、生成器への lm_gen 加算が duration 判別器の有無でネストされた内側にあり**常に未加算**になっていた。duration 判別器の有無と無関係に必ず加算する。
- **事前学習モデル誤削除の修正（同上）**: 学習開始時のクリーンアップ処理が事前学習モデル（G_0.safetensors 等）を間引く対象に含めていた不具合。
- **EMA は rank 0 のみ**: パラメータは DDP でランク間同期されるため、EMA の更新・保持・保存は rank 0 だけでよい。
- **reduce_lr 状態の再開復元**: LambdaLR は毎エポック base_lrs から LR を組み立てるため、再開時に過去の削減を base_lrs と現在 lr の両方へ再適用しないと黙って元の LR スケジュールへ戻る。削減回数は eval_history.json の lr_reductions に記録し、判定状態を持つ rank 0 から他ランクへ broadcast で揃える（学習率はランク間で必ず同一でなければならない）。
- **`global_step` はオプティマイザ更新数でなくマイクロバッチ回数**: 勾配累積導入時の設計判断。ログ・チェックポイント名・eval_interval / early_stopping_min_steps 等のステップ基準はすべてマイクロバッチ単位で、`accumulation_steps > 1` では実オプティマイザ更新数は `global_step / accumulation_steps`。累積ウィンドウの先頭以外でクリップすると合計勾配が過小になるため、実際のクリップは更新ステップのみ。
- **勾配ノルムの計算節約**: clip_grad_value_(..., None) はクリップせず全パラメータを走査するだけなので、ログ出力ステップのみ計算した（旧実装は毎回走査するだけだった）。
- **TF32 / flash SDP**: 学習の既定で有効（速度優先）。学習が壊れる場合は TF32 を落として確認すること。`torch.backends.cuda.sdp_kernel("flash")` は deprecated のため enable_* 系で同等の設定を明示。
- **tqdm**: 早期停止が有効な epochs は到達保証のない上限にすぎないので、バーは総数なし（経過step・レートのみ）にして %/上限で収束を誤解させない。
- **画像ログ撤去**: スペクトログラム・アテンションの TensorBoard 画像ログは上流の matplotlib 削除済み API に依存していたため撤去（`utils.plot_*` も削除）。TensorBoard に出るのはスカラーと音声のみ。

## 11. 推論・成果物・スタイル

- **推論 dtype 既定 float16**: フル半精度推論で VRAM 節約。bfloat16 は仮数が8ビットしかなく、duration の ceil() 量子化と組み合わせて精度劣化が大きいため推論では非対応。**学習は float32 のみ**（bf16/fp16 学習経路は廃止済み）。
- **BERT・合成モデルの遅延ロード**: 試聴GUIの起動を軽く保つため、`TTSModel` 構築時点では net_g をロードせず、初回合成時にロードする。BERT も同様に初回合成の `extract_bert_feature()` から `bert_models.load_model()` で遅延ロードされる（学習済みデータセットの BERT 特徴量は前処理で `preprocess.bert_gen` が抽出済みなので、試聴時に別途ロードするだけでよい）。**ファイルの取得**は別問題で、`TTSModelHolder` 構築時（＝app.py が launch 前に通る地点）に済ませる（§14）。
- **スタイル既定 Neutral 1本**: Neutral だけでも十分高い精度が出るため。サブディレクトリごとのスタイル生成は `--styles_by_dirs` 明示時のみ（フォルダ分けしていてもフラグが無ければ Neutral として扱われる）。スタイルベクトルの代表値は MAD（中央値絶対偏差）基準の反復トリム平均（外れ値の音声に引きずられないため。1.4826 は正規分布で MAD が標準偏差と一致する補正係数）。サブディレクトリが0〜1個ではスタイル分けの意味がないため Neutral 1本で作り直す。
- **共有契約**: `model_assets/<モデル名>/` の3点セット（config.json + safetensors + style_vectors.npy）をセットで渡す。名前が統一されていない場合は style_vectors.npy と config.json も同じセットのものが必要。
- **`best.safetensors` のコピーはアトミック**（tmp→os.replace）で、同一 best の再コピー（~120MB）は step 記録で抑止する。keep_safetensors の間引きでも best と best step の実体は消えない。
- **改行分割合成の無音間隔**: config のサンプリングレートで生成する（44100 ハードコードだと別レートのモデルで無音長がズレる）。
- **音声の16bit変換**: 無音入力だと max が 0 で 0除算→NaN になるため、正規化は最大値があるときだけ行う。

## 12. テスト設計

- **回帰テストの目的は品質検証でなく契約チェック**: 前処理→シード固定42の1エポック学習→合成→ゴールデン音声との相関差分検知。1エポックの重みはノイズ混じりだが、その出力こそがゴールデン。transformers 等のバージョン変更で BERT 出力＝推論結果が破壊的に変わったことを検知するのが目的（SPLIT_PLAN §6 の合意「破壊的変更を検知できるようにする」の実装）。
- **毎回必ず学習する**: initialize は reset_models=True で models/ を G_0.safetensors からリセットし、常に同じ初期状態の1エポックにする（成果物が残っていてもスキップしない。train/ の変更は必ず検出される）。
- **実モデル隔離**: conftest.py で SBV2_DATASET_ROOT=tests/data、SBV2_ASSETS_ROOT=tests/model_assets を設定する。既定の Data/・model_assets/ に向くと**ユーザーの実モデルを削除・上書きする**。constants は import 時に環境変数をバインドするため、conftest は全テストモジュールより先に設定する必要がある（`pytest tests/` の import 順で test_early_stop 等が先に constants を読むと既定 Data/ が確定して黙って skip される）。
- **CPU 学習非対応**: pipeline は nccl / torch.cuda.set_device() / .cuda() を無条件に使うため、学習を伴うテストは CUDA がある環境でのみ実行され、無い環境では skip される。
- **ゴールデン再生成の条件**: G2P の挙動が変わる変更（pyopenjtalk-plus のバージョン、tsqyomi/onnxruntime extras の有効化、dict_data/ 辞書更新）を入れたときは削除して再生成する（読みが変われば学習結果も変わるため）。
- **早期停止の単体テスト**（test_early_stop.py）: 停止理由が指す best と残す best が食い違わないこと（この修正の動機そのもの）、悪化・プラトー・patience の発火条件、min_steps 自動解決、reduce_lr の base_lrs スケール、best.safetensors アトミックコピーなどを契約化する。

## 13. 型チェック（pyright）ベースライン

- error にしているのは「当リポジトリで直せる型問題」だけ。残りは実バグでなく型情報の限界が原因と全件確認できたため warning にしている（error のままだとノイズに埋もれて新規の実エラーが見えなくなる）。締める際は pyproject.toml のコメントにある該当箇所から順に error へ戻す。件数・箇所は pyproject.toml の `[tool.pyright]` コメントが現行の実測と一致していること（pyright 更新で増減するため要確認）。
- `train/` 配下だけを executionEnvironments で緩和しているのは、判別器・writer・optimizer が機能オフ時に None のまま受け渡され、損失変数がフラグガードの中で条件付き定義されるという学習ループの設計上の型ノイズが集中するため。

## 14. 事前学習モデルの取得（2026-10-29）

- **現状**: 取得は `style_bert_vits2/pretrained.py` の `ensure_pretrained()` 1本。対象はコード内 manifest `PRETRAINED_REPOS` の4リポジトリだけ（bert=deberta-v2-large-japanese-char-wwm / slm=wavlm-base-plus / jp_extra=Style-Bert-VITS2-2.0-base-JP-Extra / speaker=pyannote wespeaker）。`pretrained/` 配下は**軽い config 類も含めて全てダウンロード成果物**で git に含まない（取得元が単一ソース）。上流由来の `initialize.py` の `download_default_models()`（デフォルト TTS モデル）に続き、`train/initialize.py` と `pretrained/bert/bert_models.json` も廃止した。
- **ライブラリ側に寄せた理由**: 取得は学習（SLM・JP-Extra）にも推論（BERT・話者埋め込み）にも必要な下準備なのに、旧実装はパイプライン側の `train/initialize.py` に住んでいた。そのため ①前処理を飛ばして `train_model.py` を単体実行すると SLM 欠落で落ちる ②`app.py` は BERT が無いと「ディレクトリは存在するが重みが無い」状態で `from_pretrained` の分かりにくい OSError に遭遇する、という2つの穴が残っていた。ライブラリ側に置けば消費側の入口で守れる。
- **旧実装の穴（再発防止）**: 完備チェックが `DEFAULT_BERT_MODEL_PATH.exists()`（**ディレクトリ**の存在）だった。git 同梱の `config.json` だけで pass し、重み欠落が実行時まで表面化しなかった。現在はファイル単位（0 バイトも欠落とみなす）で、取得後も欠けが残れば `PretrainedModelMissing`。
- **manifest をコード内に置いた理由**: JP-Extra 一本化で4件固定。JSON（旧 bert_models.json）は多言語対応の名残で、遅延ロード前提のデータ駆動に見見かけだけの自由度があった。dataclass にすれば型検査が効き、`hyper_parameters.py` の `slm.model` や pyannote の `AssetFileName.Model` との一致をテストで契約できる（ズレると「取得済みなのに読めない」が静かに起きる）。
- **重みは safetensors 1本**: `ku-nlp/deberta-v2-large-japanese-char-wwm` は `pytorch_model.bin` と `model.safetensors` を**両方**持つ（usedStorage 2.6GB）。文字通り全スナップショットを取ると BERT だけ2倍になるので `allow_patterns` で safetensors に絞った。実測で `.bin` と `model.safetensors` の BERT 出力（hidden_states）は**ビット単位で一致**（3文の sha256 が完全一致、float32 指定時）。既存環境に残る `pytorch_model.bin`（1.3GB）と `tokenizer.json` は自動削除しない（transformers は safetensors を優先するので放置でも動く）。
- **`tokenizer.json` は取らない**: HF 側に存在しない（旧 git 同梱ファイル）。`vocab.txt` だけで `BertJapaneseTokenizer` が構築でき、トークン列は `tokenizer.json` ありと完全一致することを実測。
- **pyannote wespeaker を unified した理由**: 前処理 Step6 は話者埋め込みが必須なのに、ここだけ HF キャッシュ（`~/.cache/huggingface`）経由で「pretrained/ が揃えばオフラインで通る」契約の外側にあった。`Model.from_pretrained(ディレクトリ)` は `AssetFileName.Model`（=pytorch_model.bin）を直接読むのでローカル化できる。コード内に「Hugging Face の利用条件に同意しトークンを設定」という案内があったが、このリポジトリは gated でなく**無認証で取得できる**（実測 200 OK）。トークン必須は上流のセグメンテーションモデルの話で、`Model.from_pretrained` は失敗時に None を返さず例外を投げるため `if model is None` 分岐も死んでいた。追加は約25MB。
- **自動取得のタイミング**: 前処理・学習は入口で全対象。推論アプリは `TTSModelHolder.__init__` 既定 `pretrained_targets=("bert",)`（app.py は launch 前にここを通る＝GUI 側を変えずに起動時取得）。初回合成のボタン押下で 1.3GB が走り出すのを避けるため、`load_model()` 内のチェックはフォールバックとして残す（`TTSModel` を直接使う select_best・テストは holder を通さない）。`load_model()/load_tokenizer()` にパスを明示したライブラリ利用者は一切ダウンロードしない。
- **通信しない契約**: 完備なら manifest の存在チェックだけで return する。`hf_hub_download`/`snapshot_download` は revision を `main` のまま使うと**存在していても HEAD を叩く**ため、先に自前で弾く（旧実装は毎回 HEAD）。オフラインは `HF_HUB_OFFLINE=1`、暗黙の取得抑止は `SBV2_SKIP_PRETRAINED_DOWNLOAD=1`（未取得が必要になると例外＋案内）。
- **`--dry-run` の落とし穴**: hf_hub の `is_cached`/`will_download` は `~/.cache/huggingface` も見るため、`local_dir` に無い 1.3GB が「取得不要」表示になり得る。`plan_download()` は local_dir の完備判定と掛け合わせて `exists/copy/download/update` に分類している。
