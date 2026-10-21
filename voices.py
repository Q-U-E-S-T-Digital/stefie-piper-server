"""Voice model registry: lang -> .onnx file, lazy load + warm cache.

Models live in ./voices/ (or $VOICES_DIR). Copy the .onnx + .onnx.json
pairs there — e.g. from ~/piper or the stefie desktop repo's
language_model/ dir. Missing files raise a clear error listing what
is actually on disk, instead of a bare traceback.
"""

import os
import re
import threading
from pathlib import Path

from piper import PiperVoice

VOICES_DIR = Path(os.getenv("VOICES_DIR", Path(__file__).parent / "voices"))

# lang -> preferred .onnx filename (first existing entry wins).
MODEL_FILES: dict[str, list[str]] = {
    "de": ["de_DE-thorsten-medium.onnx", "de_DE-thorsten-high.onnx"],
    "en": ["en_US-amy-medium.onnx"],
    "es": ["es_ES-sharvard-medium.onnx"],
    "fr": ["fr_FR-tom-medium.onnx"],
    "sw": ["sw_CD-lanfrica-medium.onnx"],
}

# Piper filename convention: <lang>_<REGION>-<speaker>-<level>.onnx
# e.g. es_ES-sharvard-medium.onnx -> language=es, speaker=sharvard, level=medium
_PIPER_NAME_RE = re.compile(r"^([a-z]{2})_[A-Z]{2}-(.+)-(.+)\.onnx$")


_lock = threading.Lock()
_voices: dict[str, PiperVoice] = {}


def normalize_lang(lang: str | None) -> str:
    tag = (lang or "de").strip().lower()
    aliases = {
        "german": "de", "deutsch": "de",
        "english": "en",
        "spanish": "es", "espanol": "es", "español": "es",
        "french": "fr", "francais": "fr", "français": "fr",
        "swahili": "sw",
    }
    return aliases.get(tag, tag[:2])


def available_files() -> list[str]:
    if not VOICES_DIR.is_dir():
        return []
    return sorted(p.name for p in VOICES_DIR.glob("*.onnx"))


def list_models() -> list[dict]:
    """Discover Piper model pairs in VOICES_DIR.

    Returns a list of dicts with keys: language, speaker, level,
    onnx_file, json_file. Files that don't match the Piper naming
    convention are skipped.
    """
    if not VOICES_DIR.is_dir():
        return []
    models = []
    for onnx in sorted(VOICES_DIR.glob("*.onnx")):
        m = _PIPER_NAME_RE.match(onnx.name)
        if not m:
            continue
        language, speaker, level = m.group(1), m.group(2), m.group(3)
        models.append({
            "language": language,
            "speaker": speaker,
            "level": level,
            "onnx_file": onnx.name,
            "json_file": onnx.name + ".json",
        })
    return models


def find_model_file(language: str, speaker: str, level: str) -> dict | None:
    """Return {onnx_file, json_file} for a matching model pair, or None."""
    for m in list_models():
        if (m["language"] == language and m["speaker"] == speaker
                and m["level"] == level):
            return {"onnx_file": m["onnx_file"], "json_file": m["json_file"]}
    return None


def available_langs() -> list[str]:
    found = []
    for lang, candidates in MODEL_FILES.items():
        if any((VOICES_DIR / f).exists() for f in candidates):
            found.append(lang)
    return found


def get_voice(lang: str | None) -> tuple[PiperVoice, str]:
    """Return (voice, model_filename), loading once and caching."""
    tag = normalize_lang(lang)
    with _lock:
        if tag in _voices:
            name = next(
                (f for f in MODEL_FILES.get(tag, []) if (VOICES_DIR / f).exists()),
                tag,
            )
            return _voices[tag], name
        for filename in MODEL_FILES.get(tag, []):
            path = VOICES_DIR / filename
            if path.exists():
                print(f"[voices] Loading {tag} <- {filename} ...", flush=True)
                voice = PiperVoice.load(str(path))
                _voices[tag] = voice
                print(f"[voices] Loaded {tag}.", flush=True)
                return voice, filename
    raise FileNotFoundError(
        f"No voice model for lang={tag!r}. "
        f"voices/ contains: {available_files() or '(empty)'}. "
        f"Copy the .onnx + .onnx.json pair into {VOICES_DIR}/."
    )
