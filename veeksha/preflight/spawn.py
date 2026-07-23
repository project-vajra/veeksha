"""Launch and manage mock-server subprocesses for preflight validation.

Servers run out-of-process (own interpreter, optionally core-pinned) so their
emit schedules stay punctual and don't contend with the veeksha clients. The
parent talks to them only over localhost HTTP: readiness via ``/health`` and
ground-truth stamps via ``/preflight/records``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

from veeksha.logger import init_logger
from veeksha.preflight.models import ServerRequestRecord

logger = init_logger(__name__)


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class MockServerHandle:
    """A running mock-server subprocess."""

    def __init__(self, proc: subprocess.Popen, host: str, port: int) -> None:
        self._proc = proc
        self.host = host
        self.port = port

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def api_base(self) -> str:
        # Trailing slash: clients append e.g. "chat/completions" to this.
        return f"http://{self.host}:{self.port}/v1/"

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def wait_until_ready(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            if not self.is_alive():
                raise RuntimeError(
                    f"mock server exited early (code {self._proc.returncode}) "
                    f"before becoming ready"
                )
            try:
                with urllib.request.urlopen(
                    f"{self.base_url}/health", timeout=0.5
                ) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
            time.sleep(0.02)
        raise TimeoutError(
            f"mock server at {self.base_url} not ready within {timeout}s "
            f"(last error: {last_err})"
        )

    def fetch_records(self) -> Dict[int, ServerRequestRecord]:
        with urllib.request.urlopen(
            f"{self.base_url}/preflight/records", timeout=5.0
        ) as resp:
            raw = resp.read()
        payload = __import__("json").loads(raw)
        return {int(k): ServerRequestRecord.from_json(v) for k, v in payload.items()}

    def close(self, timeout: float = 5.0) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
                self._proc.wait(timeout=timeout)

    def __enter__(self) -> "MockServerHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def spawn_mock_server(
    module: str,
    server_args: Dict[str, object],
    *,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    ready_timeout: float = 10.0,
) -> MockServerHandle:
    """Start a preflight mock server module in a subprocess and wait for /health.

    ``server_args`` are passed as ``--key value`` CLI flags (keys use dashes).
    """
    port = port or find_free_port(host)
    cmd = [
        sys.executable,
        "-m",
        module,
        "--host",
        host,
        "--port",
        str(port),
    ]
    for key, value in server_args.items():
        cmd += [f"--{key}", str(value)]

    # Free-threaded child; inherit PYTHONPATH so source layout resolves.
    env = dict(os.environ)
    env["PYTHON_GIL"] = "0"

    logger.info("Spawning %s on %s:%d", module, host, port)
    proc = subprocess.Popen(cmd, env=env)
    handle = MockServerHandle(proc, host, port)
    try:
        handle.wait_until_ready(timeout=ready_timeout)
    except Exception:
        handle.close()
        raise
    return handle


def spawn_mock_chat_server(
    ttfc_ms: float,
    tpoc_ms: float,
    num_chunks: int,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    ready_timeout: float = 10.0,
) -> MockServerHandle:
    """Start the streaming-chat (SSE) mock server."""
    return spawn_mock_server(
        "veeksha.preflight.servers.mock_chat_server",
        {"ttfc-ms": ttfc_ms, "tpoc-ms": tpoc_ms, "num-chunks": num_chunks},
        host=host,
        port=port,
        ready_timeout=ready_timeout,
    )


def spawn_mock_completions_server(
    ttfc_ms: float,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    ready_timeout: float = 10.0,
) -> MockServerHandle:
    """Start the non-streaming completions mock server."""
    return spawn_mock_server(
        "veeksha.preflight.servers.mock_completions_server",
        {"ttfc-ms": ttfc_ms},
        host=host,
        port=port,
        ready_timeout=ready_timeout,
    )


def spawn_mock_tts_server(
    ttfc_ms: float,
    tpoc_ms: float,
    num_chunks: int,
    chunk_bytes: int,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    ready_timeout: float = 10.0,
) -> MockServerHandle:
    """Start the streaming-audio (raw bytes) TTS mock server."""
    return spawn_mock_server(
        "veeksha.preflight.servers.mock_tts_server",
        {
            "ttfc-ms": ttfc_ms,
            "tpoc-ms": tpoc_ms,
            "num-chunks": num_chunks,
            "chunk-bytes": chunk_bytes,
        },
        host=host,
        port=port,
        ready_timeout=ready_timeout,
    )


def spawn_mock_realtime_tts_server(
    ttfc_ms: float,
    tpoc_ms: float,
    num_chunks: int,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    ready_timeout: float = 10.0,
) -> MockServerHandle:
    """Start the OpenAI-realtime TTS mock server (WebSocket)."""
    return spawn_mock_server(
        "veeksha.preflight.servers.mock_realtime_tts_server",
        {"ttfc-ms": ttfc_ms, "tpoc-ms": tpoc_ms, "num-chunks": num_chunks},
        host=host,
        port=port,
        ready_timeout=ready_timeout,
    )


def spawn_mock_vajra_tts_server(
    ttfc_ms: float,
    tpoc_ms: float,
    num_chunks: int,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    ready_timeout: float = 10.0,
) -> MockServerHandle:
    """Start the Vajra TTS-stream mock server (WebSocket, binary PCM)."""
    return spawn_mock_server(
        "veeksha.preflight.servers.mock_vajra_tts_server",
        {"ttfc-ms": ttfc_ms, "tpoc-ms": tpoc_ms, "num-chunks": num_chunks},
        host=host,
        port=port,
        ready_timeout=ready_timeout,
    )


def spawn_mock_stt_server(
    ttfc_ms: float,
    tpoc_ms: float,
    num_chunks: int,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    ready_timeout: float = 10.0,
) -> MockServerHandle:
    """Start the STT mock server (WebSocket, vllm_realtime dialect)."""
    return spawn_mock_server(
        "veeksha.preflight.servers.mock_stt_server",
        {"ttfc-ms": ttfc_ms, "tpoc-ms": tpoc_ms, "num-chunks": num_chunks},
        host=host,
        port=port,
        ready_timeout=ready_timeout,
    )
