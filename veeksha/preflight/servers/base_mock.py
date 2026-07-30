"""Shared skeleton for preflight mock servers.

Handles the parts every mock has in common -- ``/health``, ``/preflight/records``,
request-id header parsing, the ground-truth record book, and the accept loop --
leaving each concrete mock to implement only ``handle_post``: how it paces and
shapes its response.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, Iterable, Optional, Tuple

from aiohttp import web

from veeksha.preflight.models import ServerRequestRecord


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

    def open_record(self, request: web.Request) -> Tuple[int, ServerRequestRecord]:
        """Stamp the request-accept time, key by request id, start a record.

        Call this first thing in ``handle_post``, before any response work. The
        request id (``X-Veeksha-Request-Id``) is the only thing we need off the
        wire -- it correlates this server-side record with the client's own
        record book. No timing values are shipped; each side keeps its own.
        """
        server_recv_time = time.monotonic()
        request_id = self._request_id(request.headers.get("X-Veeksha-Request-Id"))
        record = ServerRequestRecord(request_id, server_recv_time, [])
        self.records[request_id] = record
        return request_id, record

    async def _health(self, _request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def _records(self, _request: web.Request) -> web.Response:
        payload = json.dumps({str(k): v.to_json() for k, v in self.records.items()})
        return web.Response(body=payload.encode(), content_type="application/json")

    async def handle_post(
        self, request: web.Request
    ) -> web.StreamResponse:  # pragma: no cover - overridden
        raise NotImplementedError

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/preflight/records", self._records)
        app.router.add_route("POST", "/{tail:.*}", self.handle_post)
        return app

    async def serve_forever(self) -> None:
        runner = web.AppRunner(self.build_app(), access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port, backlog=4096)
        await site.start()
        await asyncio.get_running_loop().create_future()


async def start_sse(request: web.Request) -> web.StreamResponse:
    """Open a server-sent-events response; the caller writes the chunks."""
    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
    )
    await response.prepare(request)
    return response


async def start_binary(request: web.Request, content_type: str) -> web.StreamResponse:
    """Open a streaming binary response; the caller writes the chunks."""
    response = web.StreamResponse(status=200, headers={"Content-Type": content_type})
    await response.prepare(request)
    return response


async def pace_chunks(
    response: web.StreamResponse,
    record: ServerRequestRecord,
    payloads: Iterable[bytes],
    ttfc_ms: float,
    tpoc_ms: float,
) -> bool:
    """Emit ``payloads`` on the configured ttfc/tpoc schedule.

    Deadlines are absolute from receipt, so a slow write cannot let the schedule
    drift. Each send time is stamped into the record book immediately
    before its write. Returns False if the client hung up mid-stream.
    """
    first_deadline = record.server_recv_time + ttfc_ms / 1000.0
    for i, payload in enumerate(payloads):
        deadline = first_deadline + i * tpoc_ms / 1000.0
        slack = deadline - time.monotonic()
        if slack > 0:
            await asyncio.sleep(slack)
        record.server_send_times.append(time.monotonic())
        try:
            await response.write(payload)
        except (ConnectionError, OSError):
            return False
    return True


async def finish(response: web.StreamResponse, trailer: Optional[bytes] = None) -> None:
    """Write an optional trailer and close the response, tolerating a dead peer."""
    try:
        if trailer:
            await response.write(trailer)
        await response.write_eof()
    except (ConnectionError, OSError):
        pass
