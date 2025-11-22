#!/usr/bin/env python3
"""Minimal OpenAI-compatible server used for orchestration E2E tests."""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Tuple


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = "DummyLLM/0.1"

    def _send_json(self, payload: Dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/v1/models":
            model_name = getattr(self.server, "model_name", "dummy")
            payload = {"data": [{"id": model_name, "object": "model"}]}
            self._send_json(payload)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown path")

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path.endswith("/chat/completions"):
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
                return

            messages = body.get("messages", [])
            reply_text = "Hello from dummy server"
            if messages:
                reply_text = f"Echo: {messages[-1].get('content', '')}"

            response = {
                "id": "dummy-chatcmpl",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": reply_text},
                    }
                ],
            }
            self._send_json(response)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown path")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Silence noisy default logging during tests
        return


def _parse_args() -> Tuple[str, int, str]:
    parser = argparse.ArgumentParser(description="Run a dummy LLM server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, required=True, help="Port to bind")
    parser.add_argument("--model", default="dummy", help="Model name")
    args = parser.parse_args()
    return args.host, args.port, args.model


def main() -> None:
    host, port, model = _parse_args()
    server = ThreadingHTTPServer((host, port), _RequestHandler)
    server.model_name = model  # type: ignore[attr-defined]

    # NOTE: We intentionally do not register SIGTERM/SIGINT handlers. Calling
    # server.shutdown() from a signal handler can deadlock if shutdown waits
    # for serve_forever() to exit on the same thread. We rely on the
    # parent/test harness to terminate this process and the existing
    # KeyboardInterrupt handling below to close the server gracefully.

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - surfaced in test logs
        print(f"Dummy server failed: {exc}", file=sys.stderr)
        raise
