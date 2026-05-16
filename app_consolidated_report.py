"""Demo: generate ONE consolidated French clinical report.

Pipeline (Darija audio -> French PDF):
    1. Optional: record mic or load audio file -> OpenAI Whisper -> Arabizi
    2. OpenAI translate -> French transcript
    3. Local MedGemma (GGUF, llama-cpp-python) -> consolidated French markdown
    4. ReportLab -> PDF in data/reports/

CLI:
    py -3.10 app_consolidated_report.py                    # uses sample text
    py -3.10 app_consolidated_report.py --record 15        # record 15s from mic
    py -3.10 app_consolidated_report.py --audio path/to.wav

Requirements (Python 3.10):
    py -3.10 -m pip install llama-cpp-python reportlab matplotlib requests \
                            openai sounddevice numpy
And put OPENAI_API_KEY in EigenBrains-HackAI-2026/.env
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.reporting import generate_consolidated_report  # noqa: E402
from src.reporting.medgemma_engine import MedGemmaEngine  # noqa: E402


# --- Local model wiring -------------------------------------------------------
MODEL_PATH = ROOT / "models" / "medgemma-1.5-medical-Q4_K_M.gguf"


def build_local_engine() -> MedGemmaEngine:
    """Build an engine that ONLY uses the local GGUF (no HF, no server)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local MedGemma GGUF not found at {MODEL_PATH}. "
            "Place the model there or update MODEL_PATH."
        )

    # Disable HuggingFace fallback so we don't hit the gated 404 on
    # google/medgemma-1.5-4b-it and don't spam stderr.
    os.environ.pop("HF_TOKEN", None)

    engine = MedGemmaEngine(
        hf_token="",                       # no HF API
        server_url="http://127.0.0.1:1",   # unreachable -> skip llama-server
        model_path=str(MODEL_PATH),        # force local GGUF
        temperature=0.3,
        max_tokens=1200,                   # cap CPU inference time
        n_ctx=4096,
        n_gpu_layers=0,                    # CPU; raise if you have CUDA build
    )

    if engine.mode != "llama-cpp-python":
        raise RuntimeError(
            f"Expected local llama-cpp-python mode, got '{engine.mode}'. "
            "Install with: py -3.10 -m pip install llama-cpp-python"
        )
    return engine


def _load_transcript() -> str:
    candidate = ROOT / "data" / "last_transcript.txt"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()

    return (
        "Le patient se plaint de fievre depuis 3 jours, toux seche, douleurs "
        "thoraciques moderees a la respiration profonde. Pas d'antecedents "
        "cardiaques connus. Pas d'allergies medicamenteuses. La famille rapporte "
        "une perte d'appetit et une fatigue importante."
    )


def _persist_transcript(arabizi: str, french: str) -> None:
    """Save last transcript so subsequent runs can reuse it."""
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "last_transcript.txt").write_text(french, encoding="utf-8")
    (data_dir / "last_transcript_arabizi.txt").write_text(arabizi, encoding="utf-8")


def _transcript_from_audio(args: argparse.Namespace) -> str | None:
    """If --audio or --record is set, run the Darija->FR pipeline and return FR text."""
    if not args.audio and not args.record:
        return None

    from src.audio2text import record_and_transcribe, transcribe_file

    if args.audio:
        print(f"[i] Transcribing audio file: {args.audio}")
        result = transcribe_file(args.audio)
    else:
        print(f"[i] Recording {args.record}s from microphone...")
        result = record_and_transcribe(seconds=float(args.record))

    print(f"[OK] Darija (ar): {result.arabic}")
    print(f"[OK] Arabizi   : {result.transcript}")
    print(f"[OK] Francais  : {result.french}")
    _persist_transcript(result.transcript, result.french)
    return result.french


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ShifA'I consolidated French report.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--audio", type=str, help="Path to an audio file to transcribe.")
    src.add_argument(
        "--record",
        type=float,
        nargs="?",
        const=15.0,
        help="Record N seconds from microphone (default 15).",
    )
    p.add_argument("--patient-id", default="P-1042")
    p.add_argument("--patient-name", default="Patient Dispensaire")
    p.add_argument("--room", default="Dispensaire rural - Salle 1")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"[i] Using local model: {MODEL_PATH}")
    engine = build_local_engine()
    print(f"[OK] Engine mode: {engine.mode}")

    transcript = _transcript_from_audio(args) or _load_transcript()

    data = {
        "patient_id": args.patient_id,
        "patient_name": args.patient_name,
        "room": args.room,
        "patient": {
            "name": args.patient_name,
            "age": "47",
            "sex": "M",
        },
        "vitals": {
            "Frequence cardiaque": "108 bpm",
            "SpO2": "93%",
            "Temperature": "38.7 C",
            "Tension arterielle": "128/82 mmHg",
            "Frequence respiratoire": "22 /min",
        },
        "transcript": transcript,
        "image_findings": (
            "Radiographie thoracique : opacite alveolaire du lobe inferieur droit "
            "evoquant une pneumopathie communautaire. Pas d'epanchement pleural."
        ),
    }

    output_dir = ROOT / "data" / "reports"
    pdf_path = generate_consolidated_report(
        data,
        output_dir=str(output_dir),
        engine=engine,
    )
    print(f"[OK] Rapport consolide genere : {pdf_path}")


if __name__ == "__main__":
    main()
