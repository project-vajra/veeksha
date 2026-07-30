"""Deterministic OpenAI-compatible completions mock (non-streaming).

The completions client does a single request/response, so there is one server
send stamp (``ttfc`` = time to produce the whole response); the mock waits that
long, records the emit time in its record book, then returns one JSON body. No
timing values on the wire, no tpoc.

Run standalone::

    python -m veeksha.preflight.servers.mock_completions_server \
        --host 127.0.0.1 --port 8124 --ttfc-ms 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import List, Optional

from aiohttp import web

from veeksha.preflight.servers.base_mock import BaseMockServer


class MockCompletionsServer(BaseMockServer):
    def __init__(
        self,
        host: str,
        port: int,
        ttfc_ms: float,
        completion_text: str = "tok tok tok",
    ) -> None:
        super().__init__(host, port)
        self.ttfc_ms = ttfc_ms
        self.completion_text = completion_text

    async def handle_post(self, request: web.Request) -> web.StreamResponse:
        _, record = self.open_record(request)
        await request.read()

        deadline = record.server_recv_time + self.ttfc_ms / 1000.0
        slack = deadline - time.monotonic()
        if slack > 0:
            await asyncio.sleep(slack)

        # single response -> one send stamp in the server's record book
        record.server_send_times.append(time.monotonic())
        body = json.dumps(
            {"choices": [{"text": self.completion_text, "index": 0}]}
        ).encode()
        return web.Response(body=body, content_type="application/json")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight mock completions server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ttfc-ms", type=float, default=150.0)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    print(f"[mock_completions_server] listening on {args.host}:{args.port}", flush=True)
    server = MockCompletionsServer(host=args.host, port=args.port, ttfc_ms=args.ttfc_ms)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":
    main()
