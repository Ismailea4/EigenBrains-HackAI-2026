"""ShifA'I - Streamlit UI for the consolidated French clinical report.

Run with:
    py -3.10 -m streamlit run streamlit_app.py

Pipeline:
    1. Record from browser mic OR upload an audio file (Darija/Arabic OK)
    2. OpenAI Whisper -> Arabizi transcript
    3. OpenAI translate -> French transcript (editable)
    4. Local MedGemma GGUF (llama-cpp-python) -> consolidated French markdown
    5. ReportLab -> downloadable PDF
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.audio2text import transcribe_bytes, transcribe_file  # noqa: E402
from src.reporting import generate_consolidated_report  # noqa: E402
from src.reporting.medgemma_engine import MedGemmaEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ShifA'I - Rapport clinique consolide",
    page_icon="medical_symbol",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
        h1, h2, h3 { color: #0f4c75; }
        .stButton>button {
            background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
            color: white; border: 0; font-weight: 600; border-radius: 8px;
            padding: 0.55rem 1.2rem;
        }
        .stButton>button:hover { background: linear-gradient(135deg, #3282b8 0%, #0f4c75 100%); }
        .stDownloadButton>button {
            background: #1f8a4c; color: white; border: 0; font-weight: 600;
            border-radius: 8px; padding: 0.6rem 1.4rem;
        }
        .pill {
            display:inline-block; padding: 2px 10px; border-radius: 999px;
            background:#e3f2fd; color:#0f4c75; font-size: 0.78rem; font-weight: 600;
            margin-right: 6px;
        }
        .ok-pill { background:#e8f5e9; color:#1f8a4c; }
        .warn-pill { background:#fff3e0; color:#b26a00; }
        .err-pill { background:#ffebee; color:#c62828; }
        .step-card {
            border: 1px solid #e0e6ed; border-radius: 12px; padding: 16px 20px;
            background: #ffffff; margin-bottom: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Engine / paths
# ---------------------------------------------------------------------------
MODEL_PATH = ROOT / "models" / "medgemma-1.5-medical-Q4_K_M.gguf"
REPORTS_DIR = ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner="Loading local MedGemma (GGUF)...")
def get_engine() -> MedGemmaEngine:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model GGUF introuvable : {MODEL_PATH}")
    os.environ.pop("HF_TOKEN", None)
    engine = MedGemmaEngine(
        hf_token="",
        server_url="http://127.0.0.1:1",   # unreachable -> skip
        model_path=str(MODEL_PATH),
        temperature=0.3,
        max_tokens=1200,
        n_ctx=4096,
        n_gpu_layers=0,
    )
    if engine.mode != "llama-cpp-python":
        raise RuntimeError(
            f"Engine mode is '{engine.mode}', expected llama-cpp-python."
        )
    return engine


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _ss(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


_ss("arabic", "")
_ss("arabizi", "")
_ss("french", "")
_ss("arabic_view", "")
_ss("french_edit", "")
_ss("audio_bytes", None)
_ss("audio_name", "")
_ss("last_pdf", None)
_ss("last_markdown", "")

# ---------------------------------------------------------------------------
# Sidebar - patient + vitals
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Patient")
    patient_id = st.text_input("ID patient", value="P-1042")
    patient_name = st.text_input("Nom complet", value="Patient Dispensaire")
    col_a, col_b = st.columns(2)
    age = col_a.text_input("Age", value="47")
    sex = col_b.selectbox("Sexe", ["M", "F", "Autre"], index=0)
    room = st.text_input("Lieu", value="Dispensaire rural - Salle 1")

    st.markdown("---")
    st.markdown("### Constantes vitales")
    fc = st.text_input("Frequence cardiaque", value="108 bpm")
    spo2 = st.text_input("SpO2", value="93%")
    temp = st.text_input("Temperature", value="38.7 C")
    bp = st.text_input("Tension arterielle", value="128/82 mmHg")
    rr = st.text_input("Frequence respiratoire", value="22 /min")

    st.markdown("---")
    st.markdown("### Imagerie / observations")
    image_findings = st.text_area(
        "Resume imagerie (optionnel)",
        value=(
            "Radiographie thoracique : opacite alveolaire du lobe inferieur droit "
            "evoquant une pneumopathie communautaire. Pas d'epanchement pleural."
        ),
        height=110,
    )

    st.markdown("---")
    engine_ready = MODEL_PATH.exists()
    if engine_ready:
        st.markdown(
            '<span class="pill ok-pill">MedGemma local</span> '
            f'<span class="pill">{MODEL_PATH.name}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="pill err-pill">GGUF manquant</span>',
            unsafe_allow_html=True,
        )
    if os.getenv("OPENAI_API_KEY"):
        st.markdown('<span class="pill ok-pill">OpenAI key OK</span>', unsafe_allow_html=True)
    else:
        # The audio module also looks in .env files, so this is just a hint.
        st.markdown(
            '<span class="pill warn-pill">OPENAI_API_KEY: verifier .env</span>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("ShifA'I - Rapport medical consolide")
st.caption(
    "Triage IA pour dispensaires ruraux : audio darija -> francais -> rapport clinique PDF."
)

# ---------------------------------------------------------------------------
# Step 1 - Audio capture
# ---------------------------------------------------------------------------
st.markdown("## 1. Audio de la consultation")
audio_col1, audio_col2 = st.columns([1, 1])

with audio_col1:
    st.markdown("**Enregistrer depuis le micro**")
    try:
        from streamlit_mic_recorder import mic_recorder
        rec = mic_recorder(
            start_prompt="Demarrer l'enregistrement",
            stop_prompt="Arreter",
            just_once=False,
            use_container_width=True,
            format="wav",
            key="mic",
        )
        if isinstance(rec, dict) and rec.get("bytes"):
            st.session_state.audio_bytes = rec["bytes"]
            st.session_state.audio_name = "mic_recording.wav"
    except Exception as exc:  # pragma: no cover - component might not load
        st.warning(f"Composant micro indisponible ({exc}). Utilisez l'upload ci-contre.")

with audio_col2:
    st.markdown("**Ou importer un fichier audio**")
    uploaded = st.file_uploader(
        "wav, mp3, m4a, ogg...",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        st.session_state.audio_bytes = uploaded.read()
        st.session_state.audio_name = uploaded.name

has_audio = bool(st.session_state.audio_bytes)
if has_audio:
    st.audio(st.session_state.audio_bytes)
    st.caption(
        f"Audio pret : **{st.session_state.audio_name}** "
        f"({len(st.session_state.audio_bytes)/1024:.1f} KB)"
    )
else:
    st.info(
        "Enregistrez puis appuyez sur **Arreter** (bouton vert deviendra rouge), "
        "ou importez un fichier. Le bouton Transcrire s'active ensuite."
    )

if st.button(
    "Transcrire et traduire",
    type="primary",
    use_container_width=False,
):
    if not st.session_state.audio_bytes:
        st.error(
            "Aucun audio detecte. Si vous avez utilise le micro, verifiez que "
            "vous avez bien clique sur 'Arreter' et autorise l'acces au micro."
        )
    else:
        with st.spinner("Whisper (arabe) + traduction francaise en cours..."):
            try:
                result = transcribe_bytes(
                    st.session_state.audio_bytes,
                    filename=st.session_state.audio_name or "recording.wav",
                )
                # Write to the WIDGET keys so the textareas refresh on rerun.
                st.session_state.arabic_view = result.arabic
                st.session_state.french_edit = result.french
                st.session_state.arabic = result.arabic
                st.session_state.arabizi = result.transcript
                st.session_state.french = result.french
                st.success("Transcription terminee.")
            except Exception as exc:
                st.error(f"Echec transcription : {exc}")

# ---------------------------------------------------------------------------
# Step 2 - Transcript edit
# ---------------------------------------------------------------------------
st.markdown("## 2. Transcription")
tx_col1, tx_col2 = st.columns(2)
with tx_col1:
    st.text_area(
        "Darija (arabe - lecture seule)",
        height=180,
        disabled=True,
        key="arabic_view",
    )
    if st.session_state.get("arabizi"):
        with st.expander("Voir en Arabizi (latin + chiffres)"):
            st.code(st.session_state.arabizi, language="text")
with tx_col2:
    st.text_area(
        "Francais (editable - corrigez si besoin)",
        height=180,
        key="french_edit",
    )
# Mirror widget state into the "logical" keys used downstream.
st.session_state.arabic = st.session_state.get("arabic_view", "")
st.session_state.french = st.session_state.get("french_edit", "")

# ---------------------------------------------------------------------------
# Step 3 - Generate report
# ---------------------------------------------------------------------------
st.markdown("## 3. Generer le rapport consolide")

if st.button("Lancer MedGemma -> PDF", type="primary"):
    if not st.session_state.french.strip():
        st.error("Le transcript francais est vide. Tapez du texte ou transcrivez un audio.")
        st.stop()
    if not engine_ready:
        st.error(f"Modele GGUF introuvable : {MODEL_PATH}")
        st.stop()
    try:
        engine = get_engine()
    except Exception as exc:
        st.error(f"Impossible de charger MedGemma : {exc}")
        st.stop()

    data = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "room": room,
        "patient": {"name": patient_name, "age": age, "sex": sex},
        "vitals": {
            "Frequence cardiaque": fc,
            "SpO2": spo2,
            "Temperature": temp,
            "Tension arterielle": bp,
            "Frequence respiratoire": rr,
        },
        "transcript": st.session_state.french,
        "image_findings": image_findings,
    }

    progress = st.progress(0, text="Initialisation...")
    start = time.time()
    progress.progress(20, text="MedGemma genere le rapport (CPU, ~30-90s)...")

    try:
        pdf_path = generate_consolidated_report(
            data,
            output_dir=str(REPORTS_DIR),
            engine=engine,
        )
        progress.progress(100, text=f"Termine en {time.time()-start:.1f}s")
        st.session_state.last_pdf = pdf_path
        st.success(f"Rapport genere : {Path(pdf_path).name}")
    except Exception as exc:
        progress.empty()
        st.error(f"Echec generation : {exc}")

# ---------------------------------------------------------------------------
# Step 4 - Download / preview
# ---------------------------------------------------------------------------
if st.session_state.last_pdf and Path(st.session_state.last_pdf).exists():
    st.markdown("## 4. Telecharger / consulter")
    pdf_bytes = Path(st.session_state.last_pdf).read_bytes()
    st.download_button(
        "Telecharger le PDF",
        data=pdf_bytes,
        file_name=Path(st.session_state.last_pdf).name,
        mime="application/pdf",
        use_container_width=False,
    )
    with st.expander("Tous les rapports generes (data/reports/)"):
        for pdf in sorted(REPORTS_DIR.glob("*.pdf"), reverse=True)[:20]:
            st.write(f"- {pdf.name}  -  {datetime.fromtimestamp(pdf.stat().st_mtime):%d/%m/%Y %H:%M}")

st.markdown("---")
st.caption("ShifA'I - EigenBrains HackAI 2026")
