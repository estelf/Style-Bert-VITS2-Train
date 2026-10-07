import json
import os
import socket
from enum import IntEnum, auto
from typing import Any, Final


# 他のアプリとポートが衝突した場合、環境変数 SBV2_WORKER_PORT で別のポートを指定できる
WORKER_PORT: Final[int] = int(os.environ.get("SBV2_WORKER_PORT", "7861"))
# 互換サーバー探索・自前サーバー起動時に試すポート数の上限
MAX_PORT_PROBES: Final[int] = 10
HEADER_SIZE: Final[int] = 4


class RequestType(IntEnum):
    STATUS = auto()
    QUIT_SERVER = auto()
    PYOPENJTALK = auto()


# サーバーとクライアントのバージョン（プロトコル）が一致していることを確認するためのマジック文字列。
# 同じポートを別のアプリ（異なるバージョンのサーバー）が占有している場合に、誤って接続しないために使う。
PROTOCOL_MAGIC: Final[str] = "style-bert-vits2-worker-1"


class ConnectionClosedException(Exception):
    pass


# socket communication


def send_data(sock: socket.socket, data: dict[str, Any]):
    json_data = json.dumps(data).encode()
    header = len(json_data).to_bytes(HEADER_SIZE, byteorder="big")
    sock.sendall(header + json_data)


def __receive_until(sock: socket.socket, size: int):
    data = b""
    while len(data) < size:
        part = sock.recv(size - len(data))
        if part == b"":
            raise ConnectionClosedException("接続が閉じられました")
        data += part

    return data


def receive_data(sock: socket.socket) -> dict[str, Any]:
    header = __receive_until(sock, HEADER_SIZE)
    data_length = int.from_bytes(header, byteorder="big")
    body = __receive_until(sock, data_length)
    return json.loads(body.decode())
