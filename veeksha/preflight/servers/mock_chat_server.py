"""Deterministic OpenAI-compatible chat mock for preflight validation.

Emits an SSE stream whose chunks land on absolute deadlines derived from a fixed
``ttfc`` (first-chunk delay) and ``tpoc`` (inter-chunk delay). The server records
its own receive/emit stamps in its record book (no timing on the wire); the
scorer joins them with the client's record book by request id to measure
delivery lag and the server's own pacing fidelity.

Run standalone (this is how :mod:`veeksha.preflight.spawn` launches it)::

    python -m veeksha.preflight.servers.mock_chat_server \
        --host 127.0.0.1 --port 8123 --ttfc-ms 200 --tpoc-ms 20 --num-chunks 64
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import List, Optional

from aiohttp import web

from veeksha.preflight.servers.base_mock import (
    BaseMockServer,
    finish,
    pace_chunks,
    start_sse,
)


class MockChatServer(BaseMockServer):
    def __init__(
        self,
        host: str,
        port: int,
        ttfc_ms: float,
        tpoc_ms: float,
        num_chunks: int,
        chunk_text: str = "tok ",
    ) -> None:
        super().__init__(host, port)
        self.ttfc_ms = ttfc_ms
        self.tpoc_ms = tpoc_ms
        self.num_chunks = num_chunks
        self.chunk_text = chunk_text

    async def handle_post(self, request: web.Request) -> web.StreamResponse:
        _, record = self.open_record(request)
        await request.read()
        response = await start_sse(request)

        frame = json.dumps(
            {"choices": [{"delta": {"content": self.chunk_text}}]}
        ).encode()
        payloads = (b"data: " + frame + b"\n\n" for _ in range(self.num_chunks))
        if await pace_chunks(response, record, payloads, self.ttfc_ms, self.tpoc_ms):
            await finish(response, b"data: [DONE]\n\n")
        return response


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight mock chat server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ttfc-ms", type=float, default=200.0)
    p.add_argument("--tpoc-ms", type=float, default=20.0)
    p.add_argument("--num-chunks", type=int, default=64)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    print(f"[mock_chat_server] listening on {args.host}:{args.port}", flush=True)
    server = MockChatServer(
        host=args.host,
        port=args.port,
        ttfc_ms=args.ttfc_ms,
        tpoc_ms=args.tpoc_ms,
        num_chunks=args.num_chunks,
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":
    main()
