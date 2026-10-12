"""tests/ 全体の共通環境設定。

style_bert_vits2.constants は import 時に SBV2_DATASET_ROOT / SBV2_ASSETS_ROOT を
モジュール定数として読み込んでバインドする。`pytest tests/` ではテストファイルが
 alphabetical に import されるため、test_main.py より先に import されるテスト
（test_early_stop 等）が constants を読み込んだ時点で既定の Data/ が確定し、
test_main.py 側の後からの環境変数設定がプロセス内に効かなくなる
（データセットが Data/test を指して黙って skip される）。
conftest.py は全テストモジュールより先に import されるのでここで設定する
（サブプロセスの前処理・学習にも環境変数がそのまま継承される）。
"""

import os
from pathlib import Path

# データセットルート（tests/data）と学習成果物の保存先（tests/model_assets）。
# 既定の Data/・model_assets/ を向くとユーザーの実モデルを削除・上書きするため必ず差し替える
os.environ["SBV2_DATASET_ROOT"] = str(Path(__file__).parent / "data")
os.environ["SBV2_ASSETS_ROOT"] = str(Path(__file__).parent / "model_assets")
# ゴールデン音声とのビット単位比較には決定論カーネルが必要（本番学習は速度優先で既定 OFF）
os.environ["SBV2_DETERMINISTIC"] = "1"
