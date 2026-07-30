"""Deterministic TTS mock (HTTP streaming raw audio) for preflight validation.

Streams ``num_chunks`` raw-audio chunks on absolute deadlines derived from a
fixed ``ttfc`` (first-chunk / ttfc delay) and ``tpoc`` (inter-chunk delay),
recording its own receive/emit stamps in the record book. The audio bytes carry
no timing; the scorer joins the server record with the client's by request id.

Emit one chunk of exactly ``chunk_bytes`` per tick so the client's
``aiter_bytes(chunk_size=chunk_bytes)`` reads them 1:1 (the driver sets the
client's chunk size to match).

Run standalone::

    python -m veeksha.preflight.servers.mock_tts_server \
        --host 127.0.0.1 --port 8125 --ttfc-ms 120 --tpoc-ms 10 \
        --num-chunks 48 --chunk-bytes 1024
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import List, Optional

from veeksha.preflight.servers.base_mock import BaseMockServer
from veeksha.preflight.servers.base_server import HttpRequest, start_streaming_response


class MockTTSServer(BaseMockServer):
    def __init__(
        self,
        host: str,
        port: int,
        ttfc_ms: float,
        tpoc_ms: float,
        num_chunks: int,
        chunk_bytes: int = 1024,
    ) -> None:
        super().__init__(host, port)
        self.ttfc_ms = ttfc_ms
        self.tpoc_ms = tpoc_ms
        self.num_chunks = num_chunks
        self._chunk = b"\x00" * chunk_bytes

    async def handle_post(self, req: HttpRequest, writer: asyncio.StreamWriter) -> None:
        _, record = self.open_record(req)
        await start_streaming_response(writer, "application/octet-stream")

        first_deadline = record.server_recv_time + self.ttfc_ms / 1000.0
        for i in range(self.num_chunks):
            deadline = first_deadline + i * self.tpoc_ms / 1000.0
            slack = deadline - time.monotonic()
            if slack > 0:
                await asyncio.sleep(slack)

            record.server_send_times.append(time.monotonic())  # t_ss_i
            writer.write(self._chunk)
            try:
                await writer.drain()
            except (ConnectionError, OSError):
                return


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight mock TTS server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ttfc-ms", type=float, default=120.0)
    p.add_argument("--tpoc-ms", type=float, default=10.0)
    p.add_argument("--num-chunks", type=int, default=48)
    p.add_argument("--chunk-bytes", type=int, default=1024)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    print(f"[mock_tts_server] listening on {args.host}:{args.port}", flush=True)
    server = MockTTSServer(
        host=args.host,
        port=args.port,
        ttfc_ms=args.ttfc_ms,
        tpoc_ms=args.tpoc_ms,
        num_chunks=args.num_chunks,
        chunk_bytes=args.chunk_bytes,
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":
    main()
