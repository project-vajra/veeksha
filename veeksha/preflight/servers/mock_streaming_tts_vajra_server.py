"""Deterministic streaming-TTS mock speaking Vajra's native WS protocol.

Drives ``StreamingTTSClient(provider="vajra")``: drains the client's
``input.text`` segments (recording per-segment receipt) until
``input.done``, then emits ``num_chunks`` raw int16-PCM *binary* frames on
absolute deadlines (ttfc first-frame, tpoc inter-frame) anchored at response
start. Emit times go in the record book; nothing timing-related
crosses the wire.

Binary frames (rather than base64-in-JSON) mean this variant also exercises the
client's binary receive path.

Run standalone::

    python -m veeksha.preflight.servers.mock_streaming_tts_vajra_server \
        --host 127.0.0.1 --port 8131 --ttfc-ms 120 --tpoc-ms 10 --num-chunks 48
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import List, Optional

from veeksha.preflight.servers.base_ws_mock import BaseWSMockServer


class MockStreamingTTSVajraServer(BaseWSMockServer):
    def __init__(
        self,
        host: str,
        port: int,
        ttfc_ms: float,
        tpoc_ms: float,
        num_chunks: int,
        sample_rate: int = 24000,
        frame_bytes: int = 640,
    ) -> None:
        super().__init__(host, port)
        self.ttfc_ms = ttfc_ms
        self.tpoc_ms = tpoc_ms
        self.num_chunks = num_chunks
        self.sample_rate = sample_rate
        self._frame = b"\x00\x00" * (frame_bytes // 2)

    async def serve_session(self, connection, record) -> None:
        # drain client input (session.config, input.text*, input.done), recording
        # per-segment receipt for each text delta.
        async for raw in connection:
            recv_time = time.monotonic()
            if isinstance(raw, (bytes, bytearray, memoryview)):
                continue
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            etype = event.get("type")
            if etype == "input.text":
                record.input_recv_times.append(recv_time)
            elif etype == "input.done":
                break

        record.response_start_time = time.monotonic()
        await connection.send(
            json.dumps({"type": "audio.start", "sample_rate": self.sample_rate})
        )

        first_deadline = record.response_start_time + self.ttfc_ms / 1000.0
        for i in range(self.num_chunks):
            deadline = first_deadline + i * self.tpoc_ms / 1000.0
            slack = deadline - time.monotonic()
            if slack > 0:
                await asyncio.sleep(slack)
            record.server_send_times.append(time.monotonic())
            await connection.send(self._frame)  # binary PCM frame

        await connection.send(json.dumps({"type": "audio.done", "error": False}))
        await connection.send(json.dumps({"type": "session.done"}))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preflight mock streaming-TTS server (Vajra native protocol)"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ttfc-ms", type=float, default=120.0)
    p.add_argument("--tpoc-ms", type=float, default=10.0)
    p.add_argument("--num-chunks", type=int, default=48)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    print(
        f"[mock_streaming_tts_vajra_server] listening on {args.host}:{args.port}",
        flush=True,
    )
    server = MockStreamingTTSVajraServer(
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
