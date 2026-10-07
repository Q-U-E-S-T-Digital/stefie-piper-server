"""Voice model registry: lang -> .onnx file, lazy load + warm cache.

Models live in ./voices/ (or $VOICES_DIR). Copy the .onnx + .onnx.json
pairs there — e.g. from ~/piper or the stefie desktop repo's
language_model/ dir. Missing files raise a clear error listing what
is actually on disk, instead of a bare traceback.
"""

import os
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
