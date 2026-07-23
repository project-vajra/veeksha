"""Minimal asyncio HTTP/1.1 plumbing shared by the mock servers.

Deliberately tiny and dependency-free (stdlib only): a real framework would drag
in more scheduling jitter than we can afford when the whole point is to emit on
sub-millisecond deadlines. We parse just enough HTTP to serve the handful of
routes the clients and the preflight harness use.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HttpRequest:
    method: str
    path: str
    headers: Dict[str, str]  # header names lower-cased
    body: bytes


async def read_http_request(reader: asyncio.StreamReader) -> Optional[HttpRequest]:
    """Read one HTTP/1.1 request. Returns None if the peer closed cleanly."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError:
        return None
    except (ConnectionError, OSError):
        return None

    text = head[:-4].decode("latin1")
    lines = text.split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) < 2:
        return None
    method, path = parts[0], parts[1]

    headers: Dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    body = b""
    n = int(headers.get("content-length", "0") or "0")
    if n > 0:
        try:
            body = await reader.readexactly(n)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            return None

    return HttpRequest(method=method, path=path, headers=headers, body=body)


async def write_response(
    writer: asyncio.StreamWriter,
    status: str,
    body: bytes,
    content_type: str = "text/plain",
) -> None:
    """Write a complete (non-streaming) response and flush."""
    head = (
        f"HTTP/1.1 {status}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("latin1")
    writer.write(head + body)
    try:
        await writer.drain()
    except (ConnectionError, OSError):
        pass


async def start_streaming_response(
    writer: asyncio.StreamWriter, content_type: str
) -> None:
    """Begin a streamed response (no Content-Length; ends at EOF/close)."""
    head = (
        "HTTP/1.1 200 OK\r\n"
        f"Content-Type: {content_type}\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n"
    ).encode("latin1")
    writer.write(head)
    await writer.drain()


async def start_sse_response(writer: asyncio.StreamWriter) -> None:
    """Begin a Server-Sent-Events stream."""
    await start_streaming_response(writer, "text/event-stream")


def close_writer(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
    except (ConnectionError, OSError):
        pass
