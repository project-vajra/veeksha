import os
import time
import uuid

from revati.client import Client, ClientType  # type: ignore

from threading import local

local_data = local()


def is_revati_enabled() -> bool:
    return os.environ.get("REVATI_ENABLED", "false").lower() == "true"


def get_revati_server_address() -> str:
    return os.environ.get("REVATI_SERVER_ADDRESS", "ipc:///tmp/revati_server.ipc")


def create_thread_local_revati_client(name: str, client_type: ClientType) -> Client:
    if not is_revati_enabled():
        return

    reverse_endpoint = f"ipc:///tmp/revati_client_{name}_{str(uuid.uuid4())[:8]}.ipc"
    client = Client(
        name,
        client_type,
        get_revati_server_address(),
        reverse_endpoint,
    )
    client.register()
    local_data.revati_client = client


def get_time() -> float:
    if not hasattr(local_data, 'revati_client'):
        return time.monotonic()
    
    client = local_data.revati_client
    return client.get_virtual_time() * 1e-3


def sleep(seconds: float) -> None:
    if not hasattr(local_data, 'revati_client'):
        time.sleep(seconds)
        return
    
    client = local_data.revati_client
    client.time_jump(seconds)
