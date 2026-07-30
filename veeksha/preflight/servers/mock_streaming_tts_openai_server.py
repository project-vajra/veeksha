"""Deterministic streaming-TTS mock speaking the OpenAI-realtime WS protocol.

Drives ``StreamingTTSClient(provider="openai_realtime")``: handshake -> drain
the client's text input until ``response.create`` -> emit ``num_chunks`` base64
PCM ``response.output_audio.delta`` events on absolute deadlines (ttfc
first-chunk, tpoc inter-chunk) anchored at response start -> terminal
``response.done``. Emit times go in the record book; nothing timing-
related crosses the wire.

Run standalone::

    python -m veeksha.preflight.servers.mock_streaming_tts_openai_server \
        --host 127.0.0.1 --port 8130 --ttfc-ms 120 --tpoc-ms 10 --num-chunks 48
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from typing import List, Optional

from veeksha.preflight.servers.base_ws_mock import BaseWSMockServer

_PCM_CHUNK = base64.b64encode(b"\x00" * 640).decode("ascii")  # ~20ms @16k mono


class MockStreamingTTSOpenAIServer(BaseWSMockServer):
    def __init__(
        self,
        host: str,
        port: int,
        ttfc_ms: float,
        tpoc_ms: float,
        num_chunks: int,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__(host, port)
        self.ttfc_ms = ttfc_ms
        self.tpoc_ms = tpoc_ms
        self.num_chunks = num_chunks
        self.sample_rate = sample_rate

    async def serve_session(self, connection, record) -> None:
        # handshake
        await connection.send(
            json.dumps(
                {
                    "type": "session.updated",
                    "session": {
                        "audio": {"output": {"format": {"rate": self.sample_rate}}}
                    },
                }
            )
        )

        # drain client input until it asks for a response, recording per-segment
        # receipt for each text item.
        async for raw in connection:
            recv_time = time.monotonic()
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            etype = event.get("type")
            if etype == "conversation.item.create":
                record.input_recv_times.append(recv_time)
            elif etype == "response.create":
                break

        # anchor the ttfc/tpoc schedule at response start (after the input phase)
        record.response_start_time = time.monotonic()
        await connection.send(json.dumps({"type": "response.created"}))

        first_deadline = record.response_start_time + self.ttfc_ms / 1000.0
        for i in range(self.num_chunks):
            deadline = first_deadline + i * self.tpoc_ms / 1000.0
            slack = deadline - time.monotonic()
            if slack > 0:
                await asyncio.sleep(slack)
            record.server_send_times.append(time.monotonic())
            await connection.send(
                json.dumps({"type": "response.output_audio.delta", "delta": _PCM_CHUNK})
            )

        await connection.send(json.dumps({"type": "response.output_audio.done"}))
        await connection.send(
            json.dumps({"type": "response.done", "response": {"status": "completed"}})
        )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preflight mock streaming-TTS server (OpenAI realtime protocol)"
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
        f"[mock_streaming_tts_openai_server] listening on {args.host}:{args.port}",
        flush=True,
    )
    server = MockStreamingTTSOpenAIServer(
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
