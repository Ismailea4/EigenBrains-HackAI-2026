"""Reusable Darija->French audio transcription helpers.

Whisper transcribes natively in Arabic script (most accurate for Darija).
We then build the Arabizi form locally and translate to French from Arabic.
"""

from __future__ import annotations

import io
import os
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .whisper_record_transcriber import (
    ARABIC_LETTER_MAP,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    SAMPLE_RATE,
    CHANNELS,
    clean_transcript,
    load_env_file,
    make_wav_file,
)


# Arabic-first prompt: ask Whisper to stay in Arabic script even for Darija.
TRANSCRIPTION_PROMPT_AR = (
    "تفريغ صوتي بالدارجة المغربية. اكتب كل شيء بالأحرف العربية فقط، "
    "بدون ترجمة وبدون أحرف لاتينية. حافظ على لهجة المتحدث."
)

# French translation system prompt - input will be Arabic Darija.
TRANSLATION_SYSTEM_PROMPT_FR = (
    "Tu traduis un texte medical en darija marocaine (ecrit en arabe) vers "
    "le francais clinique. Rends une traduction naturelle, fidele et complete, "
    "adaptee a une consultation infirmiere. Reponds uniquement avec la "
    "traduction francaise, sans commentaire."
)


@dataclass
class TranscriptionResult:
    arabic: str       # native Arabic-script Darija
    transcript: str   # Arabizi (Latin + numbers) for quick reading
    french: str       # French translation


def arabic_to_arabizi(text: str) -> str:
    """Convert Arabic-script text to Arabizi using ARABIC_LETTER_MAP."""
    out = []
    for ch in text:
        if "\u0610" <= ch <= "\u061a" or "\u064b" <= ch <= "\u065f":
            continue  # diacritics
        if "\u0660" <= ch <= "\u0669":
            out.append(str(ord(ch) - ord("\u0660")))
            continue
        if "\u06f0" <= ch <= "\u06f9":
            out.append(str(ord(ch) - ord("\u06f0")))
            continue
        out.append(ARABIC_LETTER_MAP.get(ch, ch))
    s = "".join(out)
    # Keep readable punctuation; collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_env_chain() -> None:
    here = Path(__file__).resolve()
    load_env_file(here.parent / ".env")
    load_env_file(here.parents[2] / ".env")


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
        instructions=TRANSLATION_SYSTEM_PROMPT_FR,
        input=text,
    )
    return response.output_text.strip()


def _whisper_transcribe(client, file_obj, model: str) -> str:
    """Call Whisper pinned to Arabic. Try whisper-1 for `language` support."""
    # gpt-4o-transcribe doesn't accept `language`, whisper-1 does.
    use_whisper1 = "whisper" in model
    kwargs = {
        "model": model,
        "file": file_obj,
        "prompt": TRANSCRIPTION_PROMPT_AR,
    }
    if use_whisper1:
        kwargs["language"] = "ar"
    result = client.audio.transcriptions.create(**kwargs)
    return (getattr(result, "text", "") or "").strip()


def _wav_from_path(path: Path) -> io.BytesIO:
    buf = io.BytesIO(path.read_bytes())
    buf.name = path.name
    return buf


def _pick_tx_model(override: Optional[str]) -> str:
    # Prefer whisper-1 (supports language=ar) unless user explicitly overrides.
    return (
        override
        or os.getenv("OPENAI_TRANSCRIBE_MODEL")
        or "whisper-1"
    )


def transcribe_file(
    audio_path: str | Path,
    transcription_model: Optional[str] = None,
    translation_model: Optional[str] = None,
) -> TranscriptionResult:
    """Transcribe an existing audio file (Darija) -> Arabic + Arabizi + French."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    client = _build_client()
    tx_model = _pick_tx_model(transcription_model)
    tr_model = translation_model or os.getenv("OPENAI_TRANSLATE_MODEL", DEFAULT_TRANSLATION_MODEL)

    arabic = _whisper_transcribe(client, _wav_from_path(audio_path), tx_model)
    arabizi = arabic_to_arabizi(arabic)
    french = _translate_to_french(client, arabic or arabizi, tr_model)
    return TranscriptionResult(arabic=arabic, transcript=arabizi, french=french)


def transcribe_bytes(
    audio_bytes: bytes,
    filename: str = "recording.wav",
    transcription_model: Optional[str] = None,
    translation_model: Optional[str] = None,
) -> TranscriptionResult:
    """Transcribe in-memory audio bytes (e.g. from a browser recorder)."""
    client = _build_client()
    tx_model = _pick_tx_model(transcription_model)
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
    print(f"[audio2text] Sending {len(audio_bytes)} bytes as {safe_name} (detected {ext}) model={tx_model}")

    arabic = _whisper_transcribe(client, buf, tx_model)
    print(f"[audio2text] Arabic raw: {arabic!r}")
    arabizi = arabic_to_arabizi(arabic)
    french = _translate_to_french(client, arabic or arabizi, tr_model)
    return TranscriptionResult(arabic=arabic, transcript=arabizi, french=french)


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
    tx_model = _pick_tx_model(transcription_model)
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
    arabic = _whisper_transcribe(client, wav_file, tx_model)
    arabizi = arabic_to_arabizi(arabic)
    french = _translate_to_french(client, arabic or arabizi, tr_model)
    return TranscriptionResult(arabic=arabic, transcript=arabizi, french=french)


__all__ = [
    "TranscriptionResult",
    "transcribe_file",
    "transcribe_bytes",
    "record_and_transcribe",
]
