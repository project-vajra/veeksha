"""Shared skeleton for preflight mock servers.

Handles the parts every mock has in common -- ``/health``, ``/preflight/records``,
request-id / client-sent-time header parsing, the ground-truth record book, and
the accept loop -- leaving each concrete mock to implement only ``handle_post``:
how it paces and shapes its response.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, Optional, Tuple

from veeksha.preflight.models import ServerRequestRecord
from veeksha.preflight.servers.base_server import (
    HttpRequest,
    close_writer,
    read_http_request,
    write_response,
)


class BaseMockServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.records: Dict[int, ServerRequestRecord] = {}
        self._synthetic = -1

    def _next_synthetic_id(self) -> int:
        rid = self._synthetic
        self._synthetic -= 1
        return rid

    def _request_id(self, header_value: Optional[str]) -> int:
        if header_value is None:
            return self._next_synthetic_id()
        try:
            return int(header_value)
        except ValueError:
            return self._next_synthetic_id()

    def open_record(self, req: HttpRequest) -> Tuple[int, ServerRequestRecord]:
        """Stamp receipt (t_sr), key by request id, and start a record.

        Call this first thing in ``handle_post``, before any response work. The
        request id (``X-Veeksha-Request-Id``) is the only thing we need off the
        wire -- it correlates this server-side record with the client's own
        record book. No timing values are shipped; each side keeps its own.
        """
        server_recv_time = time.monotonic()
        request_id = self._request_id(req.headers.get("x-veeksha-request-id"))
        record = ServerRequestRecord(request_id, server_recv_time, [])
        self.records[request_id] = record
        return request_id, record

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        req = await read_http_request(reader)
        if req is None:
            close_writer(writer)
            return
        try:
            if req.path.startswith("/health"):
                await write_response(writer, "200 OK", b"ok")
            elif req.path.startswith("/preflight/records"):
                payload = json.dumps(
                    {str(k): v.to_json() for k, v in self.records.items()}
                ).encode()
                await write_response(writer, "200 OK", payload, "application/json")
            elif req.method == "POST":
                await self.handle_post(req, writer)
            else:
                await write_response(writer, "404 Not Found", b"not found")
        finally:
            close_writer(writer)

    async def handle_post(
        self, req: HttpRequest, writer: asyncio.StreamWriter
    ) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    async def serve_forever(self) -> None:
        server = await asyncio.start_server(self.handle, self.host, self.port)
        async with server:
            await server.serve_forever()
