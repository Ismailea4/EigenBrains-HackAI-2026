"""Reusable Darija->French audio transcription helpers.

Wraps the same OpenAI logic used by `whisper_record_transcriber.py` so that
the rest of the system (e.g. the consolidated report app) can call:

    from src.audio2text import transcribe_file, record_and_transcribe

Both return a `TranscriptionResult` with `.transcript` (Arabizi) and
`.french` (French translation).
"""

from __future__ import annotations

import io
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .whisper_record_transcriber import (
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    SAMPLE_RATE,
    CHANNELS,
    TRANSCRIPTION_PROMPT,
    TRANSLATION_SYSTEM_PROMPT,
    clean_transcript,
    load_env_file,
    make_wav_file,
)


@dataclass
class TranscriptionResult:
    transcript: str  # Arabizi / Latin form
    french: str      # French translation


def _load_env_chain() -> None:
    """Load .env from the audio2text folder AND the repo root."""
    here = Path(__file__).resolve()
    load_env_file(here.parent / ".env")
    load_env_file(here.parents[2] / ".env")  # repo root


def _build_client():
    _load_env_chain()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY missing. Put it in EigenBrains-HackAI-2026/.env "
            "or set it in the environment."
        )
    from openai import OpenAI
    return OpenAI()


def _translate_to_french(client, text: str, model: str) -> str:
    if not text:
        return ""
    response = client.responses.create(
        model=model,
        instructions=TRANSLATION_SYSTEM_PROMPT,
        input=text,
    )
    return response.output_text.strip()


def _wav_from_path(path: Path) -> io.BytesIO:
    buf = io.BytesIO(path.read_bytes())
    buf.name = path.name
    return buf


def transcribe_file(
    audio_path: str | Path,
    transcription_model: Optional[str] = None,
    translation_model: Optional[str] = None,
) -> TranscriptionResult:
    """Transcribe an existing audio file (wav/mp3/m4a/...) and translate to French."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    client = _build_client()
    tx_model = transcription_model or os.getenv("OPENAI_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIPTION_MODEL)
    tr_model = translation_model or os.getenv("OPENAI_TRANSLATE_MODEL", DEFAULT_TRANSLATION_MODEL)

    result = client.audio.transcriptions.create(
        model=tx_model,
        file=_wav_from_path(audio_path),
        prompt=TRANSCRIPTION_PROMPT,
    )
    transcript = clean_transcript(result.text)
    french = _translate_to_french(client, transcript, tr_model)
    return TranscriptionResult(transcript=transcript, french=french)


def transcribe_bytes(
    audio_bytes: bytes,
    filename: str = "recording.wav",
    transcription_model: Optional[str] = None,
    translation_model: Optional[str] = None,
) -> TranscriptionResult:
    """Transcribe in-memory audio bytes (e.g. from a browser recorder)."""
    client = _build_client()
    tx_model = transcription_model or os.getenv("OPENAI_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIPTION_MODEL)
    tr_model = translation_model or os.getenv("OPENAI_TRANSLATE_MODEL", DEFAULT_TRANSLATION_MODEL)

    # Detect real container from magic bytes - mic_recorder often returns
    # webm/ogg with a .wav name, which the API may silently return empty for.
    head = audio_bytes[:12]
    if head.startswith(b"RIFF") and b"WAVE" in head:
        ext = "wav"
    elif head.startswith(b"\x1aE\xdf\xa3"):
        ext = "webm"
    elif head.startswith(b"OggS"):
        ext = "ogg"
    elif head[4:8] == b"ftyp":
        ext = "m4a"
    elif head[:3] == b"ID3" or head[:2] == b"\xff\xfb":
        ext = "mp3"
    else:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"

    safe_name = f"recording.{ext}"
    buf = io.BytesIO(audio_bytes)
    buf.name = safe_name
    print(f"[audio2text] Sending {len(audio_bytes)} bytes as {safe_name} (detected {ext})")

    result = client.audio.transcriptions.create(
        model=tx_model,
        file=buf,
        prompt=TRANSCRIPTION_PROMPT,
    )
    raw_text = (getattr(result, "text", "") or "").strip()
    print(f"[audio2text] Whisper raw: {raw_text!r}")
    transcript = clean_transcript(raw_text)
    french = _translate_to_french(client, transcript or raw_text, tr_model)
    return TranscriptionResult(transcript=transcript or raw_text, french=french)


def record_and_transcribe(
    seconds: float = 10.0,
    transcription_model: Optional[str] = None,
    translation_model: Optional[str] = None,
) -> TranscriptionResult:
    """Record microphone for `seconds`, then transcribe + translate.

    Useful for CLI/headless usage (no Tk window).
    """
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Install sounddevice and numpy: pip install sounddevice numpy"
        ) from exc

    client = _build_client()
    tx_model = transcription_model or os.getenv("OPENAI_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIPTION_MODEL)
    tr_model = translation_model or os.getenv("OPENAI_TRANSLATE_MODEL", DEFAULT_TRANSLATION_MODEL)

    print(f"[REC] Recording {seconds:.1f}s at {SAMPLE_RATE} Hz...  (parlez maintenant)")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    print("[REC] Done. Transcribing...")

    audio = audio.reshape(-1)
    wav_file = make_wav_file(audio)
    result = client.audio.transcriptions.create(
        model=tx_model,
        file=wav_file,
        prompt=TRANSCRIPTION_PROMPT,
    )
    transcript = clean_transcript(result.text)
    french = _translate_to_french(client, transcript, tr_model)
    return TranscriptionResult(transcript=transcript, french=french)


__all__ = [
    "TranscriptionResult",
    "transcribe_file",
    "transcribe_bytes",
    "record_and_transcribe",
]
