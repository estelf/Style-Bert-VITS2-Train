import socket
from typing import Any, cast

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_common import (
    RequestType,
    receive_data,
    send_data,
)


class WorkerClient:
    """pyopenjtalk worker client"""

    def __init__(self, port: int, timeout: float = 60) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # timeout: seconds
        sock.settimeout(timeout)
        sock.connect(
            ("localhost", port)
        )  # WSL2 mirrored network 等を考慮し、hostname でなく明示的に localhost を使う
        self.sock = sock

    def __enter__(self) -> "WorkerClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self.sock.close()

    def dispatch_pyopenjtalk(self, func: str, *args: Any, **kwargs: Any) -> Any:
        data = {
            "request-type": RequestType.PYOPENJTALK,
            "func": func,
            "args": args,
            "kwargs": kwargs,
        }
        logger.trace(f"client sends request: {data}")
        send_data(self.sock, data)
        logger.trace("client sent request successfully")
        response = receive_data(self.sock)
        logger.trace(f"client received response: {response}")
        return response.get("return")

    def status(self) -> int:
        data = {"request-type": RequestType.STATUS}
        logger.trace(f"client sends request: {data}")
        send_data(self.sock, data)
        logger.trace("client sent request successfully")
        response = receive_data(self.sock)
        logger.trace(f"client received response: {response}")
        return cast(int, response.get("client-count"))

    def protocol(self) -> str:
        """サーバーのプロトコルバージョン（マジック文字列）を返す。互換性チェック用。"""
        data = {"request-type": RequestType.STATUS}
        send_data(self.sock, data)
        response = receive_data(self.sock)
        return cast(str, response.get("protocol", ""))

    def quit_server(self) -> None:
        data = {"request-type": RequestType.QUIT_SERVER}
        logger.trace(f"client sends request: {data}")
        send_data(self.sock, data)
        logger.trace("client sent request successfully")
        response = receive_data(self.sock)
        logger.trace(f"client received response: {response}")
