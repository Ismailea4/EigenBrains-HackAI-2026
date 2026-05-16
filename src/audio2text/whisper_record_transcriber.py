"""Small desktop recorder that transcribes microphone audio with OpenAI.

Install dependencies:
    pip install -U openai sounddevice numpy

Run:
    python whisper_record_transcriber.py
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import queue
import re
import threading
import tkinter as tk
import wave
from tkinter import messagebox, ttk

try:
    import numpy as np
    from openai import OpenAI
    import sounddevice as sd
except ImportError as error:
    messagebox.showerror(
        "Missing dependency",
        "Install the required packages first:\n\n"
        "pip install -U openai sounddevice numpy\n\n"
        f"Missing: {error.name}",
    )
    raise SystemExit(1) from error


SAMPLE_RATE = 16_000
CHANNELS = 1
DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
DEFAULT_TRANSLATION_MODEL = "gpt-4o-mini"
TRANSCRIPTION_PROMPT = (
    "Transcribe the audio in the same spoken language. Do not translate to English. "
    "If the speech is Arabic, Darija, or Arabizi, write the result using Latin letters "
    "and common Arabizi numbers such as 2, 3, 4, 5, 7, 8, and 9 instead of Arabic script."
)
TRANSLATION_SYSTEM_PROMPT = (
    "You translate user-provided transcription text into natural French. "
    "Return only the French translation, with no explanations or extra labels."
)

ARABIC_LETTER_MAP = {
    "\u0621": "2",
    "\u0622": "a",
    "\u0623": "a",
    "\u0624": "2",
    "\u0625": "i",
    "\u0626": "2",
    "\u0627": "a",
    "\u0628": "b",
    "\u0629": "a",
    "\u062a": "t",
    "\u062b": "th",
    "\u062c": "j",
    "\u062d": "7",
    "\u062e": "5",
    "\u062f": "d",
    "\u0630": "dh",
    "\u0631": "r",
    "\u0632": "z",
    "\u0633": "s",
    "\u0634": "4",
    "\u0635": "s",
    "\u0636": "d",
    "\u0637": "t",
    "\u0638": "dh",
    "\u0639": "3",
    "\u063a": "8",
    "\u0640": "",
    "\u0641": "f",
    "\u0642": "9",
    "\u0643": "k",
    "\u0644": "l",
    "\u0645": "m",
    "\u0646": "n",
    "\u0647": "h",
    "\u0648": "w",
    "\u0649": "a",
    "\u064a": "y",
    "\u0671": "a",
    "\u067e": "p",
    "\u0686": "ch",
    "\u06a4": "v",
    "\u06af": "g",
}


def clean_transcript(text: str) -> str:
    """Display Arabic-script transcripts as Arabizi letters/numbers."""
    converted: list[str] = []

    for character in text:
        if "\u0610" <= character <= "\u061a" or "\u064b" <= character <= "\u065f":
            continue
        if "\u0660" <= character <= "\u0669":
            converted.append(str(ord(character) - ord("\u0660")))
            continue
        if "\u06f0" <= character <= "\u06f9":
            converted.append(str(ord(character) - ord("\u06f0")))
            continue

        converted.append(ARABIC_LETTER_MAP.get(character, character))

    transcript = "".join(converted).encode("ascii", "ignore").decode("ascii")
    transcript = re.sub(r"[^A-Za-z0-9\s]", " ", transcript)
    return re.sub(r"\s+", " ", transcript).strip().lower()


def load_env_file(path: Path | None = None) -> None:
    env_path = path or Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def make_wav_file(audio: np.ndarray) -> io.BytesIO:
    audio = np.clip(audio, -1.0, 1.0)
    pcm_audio = (audio * 32767).astype(np.int16)

    wav_file = io.BytesIO()
    with wave.open(wav_file, "wb") as writer:
        writer.setnchannels(CHANNELS)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm_audio.tobytes())

    wav_file.seek(0)
    wav_file.name = "recording.wav"
    return wav_file


class WhisperRecorderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Whisper Recorder")
        self.root.geometry("680x420")
        self.root.minsize(520, 360)

        self.client: OpenAI | None = None
        self.transcription_model = DEFAULT_TRANSCRIPTION_MODEL
        self.translation_model = DEFAULT_TRANSLATION_MODEL
        self.recording = False
        self.audio_frames: list[np.ndarray] = []
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stream: sd.InputStream | None = None

        self.status_text = tk.StringVar(value=f"Connecting to {DEFAULT_TRANSCRIPTION_MODEL}...")
        self.button_text = tk.StringVar(value="Record")

        self._build_ui()
        threading.Thread(target=self._setup_client, daemon=True).start()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X)

        self.record_button = ttk.Button(
            controls,
            textvariable=self.button_text,
            command=self._toggle_recording,
            state=tk.DISABLED,
        )
        self.record_button.pack(side=tk.LEFT)

        status_label = ttk.Label(controls, textvariable=self.status_text)
        status_label.pack(side=tk.LEFT, padx=(14, 0))

        result_label = ttk.Label(container, text="Transcript")
        result_label.pack(anchor=tk.W, pady=(18, 6))

        self.result_box = tk.Text(container, wrap=tk.WORD, height=12, font=("Segoe UI", 12))
        self.result_box.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.result_box, command=self.result_box.yview)
        self.result_box.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _setup_client(self) -> None:
        try:
            load_env_file()
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY was not found in .env or your environment.")
            self.client = OpenAI()
            self.transcription_model = os.getenv("OPENAI_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIPTION_MODEL)
            self.translation_model = os.getenv("OPENAI_TRANSLATE_MODEL", DEFAULT_TRANSLATION_MODEL)
        except Exception as error:
            self.root.after(0, self._show_client_error, error)
            return

        self.root.after(0, self._client_ready)

    def _client_ready(self) -> None:
        self.status_text.set("Ready")
        self.record_button.configure(state=tk.NORMAL)

    def _show_client_error(self, error: Exception) -> None:
        self.status_text.set("OpenAI setup failed")
        messagebox.showerror("OpenAI setup error", str(error))

    def _toggle_recording(self) -> None:
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        self.audio_frames.clear()
        self._drain_audio_queue()

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self._capture_audio,
            )
            self.stream.start()
        except Exception as error:
            messagebox.showerror("Microphone error", str(error))
            return

        self.recording = True
        self.button_text.set("Stop")
        self.status_text.set("Recording...")
        self.root.after(100, self._collect_audio)

    def _capture_audio(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def _collect_audio(self) -> None:
        while not self.audio_queue.empty():
            self.audio_frames.append(self.audio_queue.get())

        if self.recording:
            self.root.after(100, self._collect_audio)

    def _stop_recording(self) -> None:
        self.recording = False
        self.button_text.set("Record")
        self.status_text.set("Preparing audio...")
        self.record_button.configure(state=tk.DISABLED)

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self._collect_audio()

        if not self.audio_frames:
            self.status_text.set("No audio captured")
            self.record_button.configure(state=tk.NORMAL)
            return

        audio = np.concatenate(self.audio_frames, axis=0).reshape(-1)
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio: np.ndarray) -> None:
        self.root.after(0, self.status_text.set, "Transcribing...")

        try:
            if self.client is None:
                raise RuntimeError("OpenAI client is not ready yet.")

            wav_file = make_wav_file(audio)
            result = self.client.audio.transcriptions.create(
                model=self.transcription_model,
                file=wav_file,
                prompt=TRANSCRIPTION_PROMPT,
            )
            transcript = clean_transcript(result.text)
            self.root.after(0, self.status_text.set, "Translating to French...")
            translation = self._translate_to_french(transcript) if transcript else ""
        except Exception as error:
            self.root.after(0, self._show_transcription_error, error)
            return

        self.root.after(0, self._display_result, transcript, translation)

    def _translate_to_french(self, transcript: str) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI client is not ready yet.")

        response = self.client.responses.create(
            model=self.translation_model,
            instructions=TRANSLATION_SYSTEM_PROMPT,
            input=transcript,
        )
        return response.output_text.strip()

    def _display_result(self, transcript: str, translation: str) -> None:
        self.result_box.delete("1.0", tk.END)
        if not transcript:
            self.result_box.insert(tk.END, "NO SPEECH DETECTED")
        else:
            self.result_box.insert(tk.END, f"Transcript:\n{transcript}\n\nFrench translation:\n{translation}")
        self.status_text.set("Ready")
        self.record_button.configure(state=tk.NORMAL)

    def _show_transcription_error(self, error: Exception) -> None:
        self.status_text.set("Transcription failed")
        self.record_button.configure(state=tk.NORMAL)
        messagebox.showerror("Transcription error", str(error))

    def _drain_audio_queue(self) -> None:
        while not self.audio_queue.empty():
            self.audio_queue.get()


def main() -> None:
    root = tk.Tk()
    app = WhisperRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()