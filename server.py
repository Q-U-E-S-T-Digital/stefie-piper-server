"""Stefie Piper TTS server — port 7777.

Save-time synthesis for the mobile app: phone POSTs complete text,
server returns spoken audio. Nothing realtime, no streaming protocol.

Endpoints:
  GET  /health              -> { ok, voices_loaded, voices_available }
  GET  /generate?text=..    -> legacy compat (short texts, old piper_server.py shape)
  POST /synthesize          -> primary mobile contract { text, lang }

Run:
  uvicorn server:app --host 0.0.0.0 --port 7777 --workers 2
"""

import asyncio
import io
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import voices

app = FastAPI(title="Stefie Piper TTS Server")

PORT = 7777
MAX_CHARS = 3000  # one giant transcript must not OOM the box
LEGACY_MAX_CHARS = 500  # GET URLs were never meant for paragraphs

# ONNX Runtime releases the GIL during inference, so threads (not
# processes) are enough to keep the event loop unblocked while staying
# pickle-free. If CPU saturates, scale with `--workers 2..4` instead.
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="piper")


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    lang: Optional[str] = Field(default="de")


def _synthesize_wav_bytes(text: str, lang: str) -> tuple[bytes, str, int]:
    """Blocking synth — always called inside the pool, never on the loop."""
    voice, model_name = voices.get_voice(lang)
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    data = wav_io.getvalue()
    return data, model_name, len(data)


@app.get("/health")
def health():
    return {
        "ok": True,
        "port": PORT,
        "voices_dir": str(voices.VOICES_DIR),
        "voices_available": voices.available_langs(),
        "voice_files": voices.available_files(),
        "max_chars": MAX_CHARS,
    }


@app.get("/generate")
async def generate_audio(
    text: str = Query(min_length=1),
    lang: str = Query(default="de"),
):
    """Legacy compat for the old :8080 piper_server.py shape. Short texts only."""
    if len(text) > LEGACY_MAX_CHARS:
        return JSONResponse(
            status_code=413,
            content={
                "error": f"text too long for GET ({len(text)} > {LEGACY_MAX_CHARS}). "
                "Use POST /synthesize."
            },
        )
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    try:
        data, model_name, _ = await loop.run_in_executor(
            _pool, _synthesize_wav_bytes, text, lang
        )
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        print(f"[error] GET /generate lang={lang}: {e}", flush=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    ms = int((time.perf_counter() - t0) * 1000)
    print(
        f"[synth] GET lang={lang} model={model_name} "
        f"chars={len(text)} bytes={len(data)} ms={ms}",
        flush=True,
    )
    return StreamingResponse(
        io.BytesIO(data),
        media_type="audio/wav",
        headers={"X-Synth-Ms": str(ms), "X-Audio-Bytes": str(len(data))},
    )


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Primary mobile contract. Complete text in, spoken WAV bytes out."""
    text = req.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "empty text"})
    if len(text) > MAX_CHARS:
        return JSONResponse(
            status_code=413,
            content={
                "error": f"text too long ({len(text)} > {MAX_CHARS}). "
                "Split per paragraph and call again."
            },
        )
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    try:
        data, model_name, _ = await loop.run_in_executor(
            _pool, _synthesize_wav_bytes, text, req.lang or "de"
        )
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        print(f"[error] POST /synthesize lang={req.lang}: {e}", flush=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    ms = int((time.perf_counter() - t0) * 1000)
    tag = voices.normalize_lang(req.lang)
    print(
        f"[synth] POST lang={tag} model={model_name} "
        f"chars={len(text)} bytes={len(data)} ms={ms}",
        flush=True,
    )
    return Response(
        content=data,
        media_type="audio/wav",
        headers={
            "X-Synth-Ms": str(ms),
            "X-Audio-Bytes": str(len(data)),
            "X-Voice-Model": model_name,
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
