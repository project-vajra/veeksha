"""Shared skeleton for preflight WebSocket mock servers.

Like base_mock.py but over WebSockets. A single ``websockets`` server both:
- upgrades WS connections (one per request) to ``serve_session`` (overridden per
  protocol), stamping the accept time and reading the request id off the
  handshake headers; and
- answers plain HTTP ``GET /health`` and ``GET /preflight/records`` via
  ``process_request`` so the parent can probe readiness and fetch ground truth.

No timing values cross the wire -- the request id (handshake header) is the only
thing needed to correlate this server-side record with the client's record book.
"""

from __future__ import annotations

import asyncio
import json
import time
from http import HTTPStatus
from typing import Dict, Tuple

from websockets.asyncio.server import serve

from veeksha.preflight.models import ServerRequestRecord


class BaseWSMockServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.records: Dict[int, ServerRequestRecord] = {}
        self._synthetic = -1

    def _next_synthetic_id(self) -> int:
        rid = self._synthetic
        self._synthetic -= 1
        return rid

    def _request_id(self, headers) -> int:
        value = headers.get("X-Veeksha-Request-Id")
        if value is None:
            return self._next_synthetic_id()
        try:
            return int(value)
        except ValueError:
            return self._next_synthetic_id()

    def _process_request(self, connection, request):
        path = request.path
        if path.startswith("/health"):
            return connection.respond(HTTPStatus.OK, "ok")
        if path.startswith("/preflight/records"):
            body = json.dumps({str(k): v.to_json() for k, v in self.records.items()})
            return connection.respond(HTTPStatus.OK, body)
        return None  # proceed with the WebSocket handshake

    def open_record(self, connection) -> Tuple[int, ServerRequestRecord]:
        """Stamp the accept time (the request-receipt stamp), start a record."""
        server_recv_time = time.monotonic()
        request_id = self._request_id(connection.request.headers)
        record = ServerRequestRecord(request_id, server_recv_time, [])
        self.records[request_id] = record
        return request_id, record

    async def _handler(self, connection) -> None:
        _, record = self.open_record(connection)
        try:
            await self.serve_session(connection, record)
        except Exception:
            # A client that hangs up mid-stream is normal; the record is kept.
            pass

    async def serve_session(self, connection, record: ServerRequestRecord) -> None:
        raise NotImplementedError

    async def serve_forever(self) -> None:
        async with serve(
            self._handler,
            self.host,
            self.port,
            process_request=self._process_request,
            max_size=None,
            compression=None,
        ):
            await asyncio.get_running_loop().create_future()  # run forever
