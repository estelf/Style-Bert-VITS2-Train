"""
Run the pyopenjtalk worker in a separate process
to avoid user dictionary access error
"""

from typing import Any

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_client import WorkerClient
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_common import (
    MAX_PORT_PROBES,
    PROTOCOL_MAGIC,
    WORKER_PORT,
)

WORKER_CLIENT: WorkerClient | None = None


# pyopenjtalk interface
# g2p(): not used


def run_frontend(text: str) -> list[dict[str, Any]]:
    if WORKER_CLIENT is not None:
        ret = WORKER_CLIENT.dispatch_pyopenjtalk("run_frontend", text)
        assert isinstance(ret, list)
        return ret
    else:
        # without worker
        import pyopenjtalk

        return pyopenjtalk.run_frontend(text)


def make_label(njd_features: Any) -> list[str]:
    if WORKER_CLIENT is not None:
        ret = WORKER_CLIENT.dispatch_pyopenjtalk("make_label", njd_features)
        assert isinstance(ret, list)
        return ret
    else:
        # without worker
        import pyopenjtalk

        return pyopenjtalk.make_label(njd_features)


def mecab_dict_index(path: str, out_path: str, dn_mecab: str | None = None) -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("mecab_dict_index", path, out_path, dn_mecab)
    else:
        # without worker
        import pyopenjtalk

        pyopenjtalk.mecab_dict_index(path, out_path, dn_mecab)


def update_global_jtalk_with_user_dict(path: str) -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("update_global_jtalk_with_user_dict", path)
    else:
        # without worker
        import pyopenjtalk

        pyopenjtalk.update_global_jtalk_with_user_dict(path)


def unset_user_dict() -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("unset_user_dict")
    else:
        # without worker
        import pyopenjtalk

        pyopenjtalk.unset_user_dict()


# initialize module when imported


def _try_compatible_client(port: int, timeout: float = 3) -> WorkerClient | None:
    """ポートにこのライブラリ互換のワーカーサーバーがいれば接続して返す。いなければ None"""
    try:
        client = WorkerClient(port, timeout=timeout)
    except (TimeoutError, OSError):
        return None
    try:
        if client.protocol() != PROTOCOL_MAGIC:
            # 別バージョン・別アプリのサーバーがこのポートを占有している
            client.close()
            return None
    except Exception:
        client.close()
        return None
    return client


def initialize_worker(port: int = WORKER_PORT) -> None:
    import atexit
    import signal
    import socket
    import sys
    import time

    global WORKER_CLIENT
    if WORKER_CLIENT:
        return

    # このライブラリ互換のワーカーサーバーが既に立っていないか、指定ポートから順に探す
    # （他のアプリが同じポートを別の目的で使っている場合があるため、マジック文字列で互性を確認する）
    client = None
    chosen_port = port
    for candidate in range(port, port + MAX_PORT_PROBES):
        client = _try_compatible_client(candidate)
        if client is not None:
            chosen_port = candidate
            break

    if client is None:
        logger.debug("try starting pyopenjtalk worker server")
        import os
        import subprocess

        # 自前のサーバーを、この範囲で見つからなかった場合は最初の空きポートに立てる
        for candidate in range(port, port + MAX_PORT_PROBES):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.bind(("localhost", candidate))
                chosen_port = candidate
                break
            except OSError:
                continue
        else:
            raise RuntimeError(
                f"pyopenjtalk worker server を立てられる空きポートが見つかりませんでした (ports: {port}-{port + MAX_PORT_PROBES - 1})。環境変数 SBV2_WORKER_PORT で別ポートを指定してください。"
            )

        worker_pkg_path = os.path.relpath(
            os.path.dirname(__file__), os.getcwd()
        ).replace(os.sep, ".")
        args = [sys.executable, "-m", worker_pkg_path, "--port", str(chosen_port)]
        # new session, new process group
        if sys.platform.startswith("win"):
            cf = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
            si = subprocess.STARTUPINFO()  # type: ignore
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore
            si.wShowWindow = subprocess.SW_HIDE  # type: ignore
            subprocess.Popen(args, creationflags=cf, startupinfo=si)
        else:
            # align with Windows behavior
            # start_new_session is same as specifying setsid in preexec_fn
            subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # wait until server listening
        count = 0
        while True:
            client = _try_compatible_client(chosen_port)
            if client is not None:
                break
            time.sleep(0.5)
            count += 1
            # 20: max number of retries
            if count == 20:
                raise TimeoutError(
                    f"pyopenjtalk worker server に接続できませんでした (port: {chosen_port})。"
                    "他のプロセスがポートを占有していないか確認してください（環境変数 SBV2_WORKER_PORT で別ポートを指定できます）。"
                )

    logger.debug(f"pyopenjtalk worker server started (port: {chosen_port})")
    WORKER_CLIENT = client
    atexit.register(terminate_worker)

    # when the process is killed
    def signal_handler(signum: int, frame: Any):
        terminate_worker()

    try:
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        # signal only works in main thread
        pass


# top-level declaration
def terminate_worker() -> None:
    logger.debug("pyopenjtalk worker server terminated")
    global WORKER_CLIENT
    if not WORKER_CLIENT:
        return

    # prepare for unexpected errors
    try:
        if WORKER_CLIENT.status() == 1:
            WORKER_CLIENT.quit_server()
    except Exception as e:
        logger.error(e)

    WORKER_CLIENT.close()
    WORKER_CLIENT = None
