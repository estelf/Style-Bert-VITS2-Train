# AGENTS.md

日本語・JP-Extra 一本化の Style-Bert-VITS2 学習コア（フォーク）。詳細は [docs/GUIDE.md](docs/GUIDE.md)、FAQ は docs/FAQ.md、分割設計は docs/SPLIT_PLAN.md を参照。コメント・ドキュメントは日本語で書かれている。

## 実行系の基本

- すべてのコマンドは `uv run ...` 経由で実行する（素の python/pip ではない）。トップレベルの呼び出し口は3つだけ:
  - `uv run preprocess_all.py -m <モデル名>`（前処理 Step1〜6、学習設定を `Data/<モデル名>/config.json` に書き込む）
  - `uv run train_model.py -m <モデル名>`（学習。新規も再開も同じコマンド。`model_assets/<モデル名>/` に推論資産があればスタイル生成は自動スキップ）
  - `uv run app.py`（試聴GUI・音声合成タブのみ。学習機能なし）
- データセット契約: `Data/<モデル名>/` に `esd.list`（1行1文 `<ファイル名>|<話者>|JP|<セリフ>`）+ `raw.zip` を置く。言語は JP のみ、英語はカタカナ入力前提で読みは `dict_data/` のユーザー辞書で調整。
- フラットレイアウト（`src/` ではない）。`style_bert_vits2/` がコアライブラリ、`train/` が学習パイプライン、`preprocess/` が前処理チェーン。

## 必ず揃える版数・環境変数

- バージョンは `pyproject.toml` の `[project].version` が単一のソースオブトリュース。`constants.py` の `VERSION` は `importlib.metadata.version()` でそこから読むため手動同期は不要（uv_build は動的バージョン非対応なので静的指定のまま）。
- データセットルートは環境変数 `SBV2_DATASET_ROOT` で差し替え可能（デフォルト `Data/`、テストは `tests/data` を使う）。pyopenjtalk ワーカーのポートは `SBV2_WORKER_PORT`（デフォルト 7861）。
- 学習の分散環境デフォルトは `constants.py` の `TRAIN_ENV_DEFAULTS` にある（MASTER_ADDR は IPv6 ループバック `::1` をリテラル指定。WSL2 mirrored mode 対策なので安易に localhost に変えない）。

## ビルド・テストの癖

- `pyopenjtalk-dict` はホイールが numpy 1.x ABI 前提のためソースビルド必須（`[tool.uv]` の no-binary-package / cython>=3.1 制約で担保済み）。GPU 環境では `uv sync` 前後に OS/CUDA に応じた torch を別途入れる（例: `uv pip install "torch" "torchaudio" --index-url https://download.pytorch.org/whl/cu128`）。
- 回帰テスト: `uv run pytest -v`（前処理→シード固定42の1エポック学習→合成→ゴールデン音声との相関差分検知。品質検証ではなく契約チェック）。
  - テストデータセット・ゴールデン音声は git に含まれない。`tests/data/<モデル名>/` に配置しないとテストは skip される（clone 直後は存在しないのが正常）。デフォルトは `SBV2_TEST_MODEL=model_1`、差し替えは環境変数で。ゴールデン（`tests/references/`）は初回実行時に自動作成される。
- リンターは ruff（dev 依存済み。`[tool.ruff.lint].select = ["E4","E7","E9","F","I","UP"]`）、フォーマットは black（dev 依存済み）。実行は `uv run ruff check .` / `uv run black .`。

## 成果物の置き場

- 学習の途中状態・ログ: `Data/<モデル名>/models/`（G_/D_/WD_*.pth、tfevents。学習曲線は `uv run tensorboard --logdir Data/<モデル名>/models`）。
- 推論・共有用3点セット: `model_assets/<モデル名>/`（config.json + `<モデル名>_e<epoch>_s<step>.safetensors` + `style_vectors.npy`）。共有時は3点をセットで渡すこと。名前が統一されていない場合は同じセットの style_vectors.npy/config.json が必要。
- `pretrained/` の事前学習モデル（bert/jp_extra/slm）は前処理時に自動取得される。一括手動取得は `uv run -m train.initialize`。
