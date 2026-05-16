"""Audio-to-text (Darija/Arabic -> Arabizi -> French) module for ShifA'I."""

from .api import (
    TranscriptionResult,
    record_and_transcribe,
    transcribe_bytes,
    transcribe_file,
)

__all__ = [
    "TranscriptionResult",
    "transcribe_file",
    "transcribe_bytes",
    "record_and_transcribe",
]
