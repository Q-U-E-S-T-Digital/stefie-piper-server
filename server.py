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
import os
import time
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Body, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import voices


def _load_dotenv(path: str) -> None:
    """Load a .env file into os.environ (no external dependency)."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        pass


# Load local .env so STEFIE_BOOT_TOKEN (and other secrets) are available
# when running with `uvicorn server:app` directly.
_load_dotenv(Path(__file__).with_name(".env"))

app = FastAPI(title="Stefie Piper TTS Server")

PORT = 7777
MAX_CHARS = 3000  # one giant transcript must not OOM the box
LEGACY_MAX_CHARS = 500  # GET URLs were never meant for paragraphs

# ONNX Runtime releases the GIL during inference, so threads (not
# processes) are enough to keep the event loop unblocked while staying
# pickle-free. If CPU saturates, scale with `--workers 2..4` instead.
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="piper")

# Model download endpoints reuse the same boot token the desktop app sends
# to the Cloudflare relay /config endpoint. If STEFIE_BOOT_TOKEN is not set
# on the server, all model endpoints fail closed (401) — never allow all.
MODEL_SERVER_TOKEN = os.getenv("STEFIE_BOOT_TOKEN", "")


def _check_auth(authorization: str | None):
    """Require Bearer <STEFIE_BOOT_TOKEN> on model endpoints.

    Fails closed when STEFIE_BOOT_TOKEN is not configured on the server.
    """
    if not MODEL_SERVER_TOKEN:
        raise HTTPException(status_code=401, detail="server not configured")
    if authorization != f"Bearer {MODEL_SERVER_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


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


@app.get("/models")
def list_available_models(authorization: str | None = Header(default=None)):
    """List available Piper model .onnx + .json pairs on the server."""
    _check_auth(authorization)
    return {"models": voices.list_models()}


@app.post("/models/download")
def download_model(
    language: str = Body(...),
    speaker: str = Body(...),
    level: str = Body(...),
    authorization: str | None = Header(default=None),
):
    """Download a Piper model pair as a ZIP containing .onnx + .onnx.json."""
    _check_auth(authorization)
    match = voices.find_model_file(language, speaker, level)
    if not match:
        raise HTTPException(status_code=404, detail="model not found")
    onnx_path = voices.VOICES_DIR / match["onnx_file"]
    json_path = voices.VOICES_DIR / match["json_file"]
    if not onnx_path.is_file() or not json_path.is_file():
        raise HTTPException(status_code=404, detail="model files missing on disk")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(onnx_path, arcname=match["onnx_file"])
        z.write(json_path, arcname=match["json_file"])
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{match["onnx_file"]}.zip"',
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
