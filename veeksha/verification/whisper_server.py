"""HTTP wrapper around faster-whisper for post-run TTS verification."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any


def create_app(model_name: str, device: str, compute_type: str) -> Any:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from faster_whisper import WhisperModel

    app = FastAPI()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(await file.read())

            segments, _ = model.transcribe(tmp_path, beam_size=1)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            return {"text": text}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    args = parser.parse_args()

    import uvicorn

    app = create_app(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
