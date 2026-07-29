"""Deterministic STT mock (WebSocket, vllm_realtime dialect) for preflight.

Handshake ``session.created`` -> drain the client's ``input_audio_buffer.append``
chunks (recording per-chunk receipt, t_sr_i) until the final
``input_audio_buffer.commit`` -> emit ``num_chunks`` ``transcription.delta``
events on absolute deadlines (ttfc/tpoc) anchored at response start -> terminal
``transcription.done``. Emit times (t_ss_i) go in the record book.

Run standalone::

    python -m veeksha.preflight.servers.mock_stt_server \
        --host 127.0.0.1 --port 8132 --ttfc-ms 120 --tpoc-ms 10 --num-chunks 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import List, Optional

from veeksha.preflight.servers.base_ws_mock import BaseWSMockServer


class MockSTTServer(BaseWSMockServer):
    def __init__(
        self,
        host: str,
        port: int,
        ttfc_ms: float,
        tpoc_ms: float,
        num_chunks: int,
        delta_text: str = "word ",
    ) -> None:
        super().__init__(host, port)
        self.ttfc_ms = ttfc_ms
        self.tpoc_ms = tpoc_ms
        self.num_chunks = num_chunks
        self.delta_text = delta_text

    async def serve_session(self, connection, record) -> None:
        # vllm_realtime handshake: server announces the session first.
        await connection.send(json.dumps({"type": "session.created"}))

        # drain client input (session.update, initial commit, append*, final
        # commit), recording per-audio-chunk receipt.
        async for raw in connection:
            recv_time = time.monotonic()
            if isinstance(raw, (bytes, bytearray, memoryview)):
                continue
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            etype = event.get("type")
            if etype == "input_audio_buffer.append":
                record.input_recv_times.append(recv_time)
            elif etype == "input_audio_buffer.commit" and event.get("final"):
                break

        record.response_start_time = time.monotonic()
        first_deadline = record.response_start_time + self.ttfc_ms / 1000.0
        for i in range(self.num_chunks):
            deadline = first_deadline + i * self.tpoc_ms / 1000.0
            slack = deadline - time.monotonic()
            if slack > 0:
                await asyncio.sleep(slack)
            record.server_send_times.append(time.monotonic())  # t_ss_i
            await connection.send(
                json.dumps({"type": "transcription.delta", "delta": self.delta_text})
            )

        await connection.send(
            json.dumps(
                {
                    "type": "transcription.done",
                    "text": self.delta_text * self.num_chunks,
                }
            )
        )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight mock STT server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ttfc-ms", type=float, default=120.0)
    p.add_argument("--tpoc-ms", type=float, default=10.0)
    p.add_argument("--num-chunks", type=int, default=32)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    print(f"[mock_stt_server] listening on {args.host}:{args.port}", flush=True)
    server = MockSTTServer(
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
