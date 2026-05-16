"""Consolidated French clinical report for ShifA'I.

Generates a single unified PDF report (no day/night split) from a rural
nurse consultation: Darija transcript + optional image findings + vitals.
Uses MedGemma for the clinical reasoning section and ReportLab for layout.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - reportlab is required
    REPORTLAB_AVAILABLE = False

from .medgemma_engine import MedGemmaEngine
from .pdf_generator import PDFReportGenerator


CONSOLIDATED_SYSTEM_PROMPT = (
    "Tu es ShifA'I, un assistant clinique IA destine aux infirmieres des dispensaires "
    "ruraux marocains. Tu produis un RAPPORT MEDICAL CONSOLIDE UNIQUE en francais. "
    "Tu n'utilises jamais l'arabe ni le darija dans la sortie finale, meme si la "
    "transcription est en darija. Tu integres dans un seul document : motif de "
    "consultation, anamnese, constantes vitales, observations cliniques, analyse "
    "d'imagerie si fournie, hypotheses diagnostiques avec niveau de probabilite, "
    "drapeaux rouges, conduite a tenir et niveau d'urgence."
    "\n\nSTRUCTURE OBLIGATOIRE EN MARKDOWN :"
    "\n## Resume Executif"
    "\n## Motif de Consultation"
    "\n## Anamnese"
    "\n## Constantes Vitales"
    "\n## Examen Clinique"
    "\n## Analyse d'Imagerie"
    "\n## Hypotheses Diagnostiques"
    "\n## Drapeaux Rouges"
    "\n## Conduite a Tenir"
    "\n## Niveau d'Urgence"
    "\n\nN'inclus aucun raisonnement intermediaire ni texte hors structure. "
    "Sois concis, factuel et orienté action pour une infirmiere."
)


def _build_user_prompt(data: Dict[str, Any]) -> str:
    transcript = data.get("transcript", "").strip() or "Non disponible."
    image_findings = data.get("image_findings", "").strip() or "Aucune imagerie fournie."
    vitals = data.get("vitals") or {}
    patient = data.get("patient") or {}

    vitals_lines = (
        "\n".join(f"- {k}: {v}" for k, v in vitals.items()) if vitals else "- Non renseignees"
    )
    patient_lines = (
        "\n".join(f"- {k}: {v}" for k, v in patient.items()) if patient else "- Non renseignees"
    )

    return (
        "INFORMATIONS PATIENT :\n"
        f"{patient_lines}\n\n"
        "TRANSCRIPTION DARIJA -> FR (consultation infirmiere) :\n"
        f"{transcript}\n\n"
        "CONSTANTES VITALES :\n"
        f"{vitals_lines}\n\n"
        "ANALYSE D'IMAGERIE (radio, echographie, photo lesion) :\n"
        f"{image_findings}\n\n"
        "Produis le rapport medical consolide en francais selon la structure imposee."
    )


def generate_report_text(
    data: Dict[str, Any],
    engine: Optional[MedGemmaEngine] = None,
) -> str:
    """Run MedGemma (or simulation) to produce the consolidated French markdown."""
    engine = engine or MedGemmaEngine()
    messages = [
        {"role": "system", "content": CONSOLIDATED_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(data)},
    ]

    if engine.is_loaded:
        result = engine._call_model(messages)
        if result:
            return result.strip()

    return _simulate_report(data)


def _simulate_report(data: Dict[str, Any]) -> str:
    transcript = data.get("transcript", "").strip() or "Non disponible."
    return (
        "## Resume Executif\n"
        "Rapport simule - MedGemma non connecte.\n\n"
        "## Motif de Consultation\n"
        f"{transcript[:200]}\n\n"
        "## Anamnese\nA documenter par l'infirmiere.\n\n"
        "## Constantes Vitales\nNon renseignees.\n\n"
        "## Examen Clinique\nA completer.\n\n"
        "## Analyse d'Imagerie\nAucune imagerie fournie.\n\n"
        "## Hypotheses Diagnostiques\n- A evaluer cliniquement.\n\n"
        "## Drapeaux Rouges\n- Aucun identifie automatiquement.\n\n"
        "## Conduite a Tenir\n- Surveillance et reevaluation.\n\n"
        "## Niveau d'Urgence\nModere (par defaut en mode simulation)."
    )


def generate_consolidated_report(
    data: Dict[str, Any],
    output_dir: str = "./data/reports",
    filename: Optional[str] = None,
    engine: Optional[MedGemmaEngine] = None,
) -> str:
    """Generate one consolidated French PDF report.

    Expected ``data`` keys (all optional except ``transcript``):
        patient_id, patient_name, room, age, sex
        transcript (Darija->French transcript text)
        vitals (dict like {"FC": "92 bpm", "SpO2": "97%"})
        image_findings (str, e.g. radiology summary)
        recommendations (list of strings - manually added if needed)
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is required to generate the PDF report.")

    engine = engine or MedGemmaEngine()
    report_markdown = generate_report_text(data, engine=engine)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    patient_id = data.get("patient_id", "unknown")
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = filename or f"rapport_consolide_{patient_id}_{date_str}.pdf"
    filepath = output_path / filename

    generator = PDFReportGenerator(output_dir=str(output_path))
    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story: List = []
    story.extend(generator._build_header("Rapport Medical Consolide", data))
    story.extend(_build_patient_summary(generator, data))
    story.extend(generator._parse_markdown_to_elements(report_markdown))
    story.append(Spacer(1, 12))
    story.extend(generator._build_footer())

    doc.build(story)
    return str(filepath)


def _build_patient_summary(generator: PDFReportGenerator, data: Dict[str, Any]) -> List:
    elements: List = []
    patient = data.get("patient") or {}
    vitals = data.get("vitals") or {}

    elements.append(Paragraph("Synthese Patient", generator.styles["SectionTitle"]))

    summary_rows = [
        ["Nom", patient.get("name", data.get("patient_name", "N/A")), "Age", patient.get("age", "N/A")],
        ["Sexe", patient.get("sex", "N/A"), "Chambre", data.get("room", "N/A")],
        ["Date", datetime.now().strftime("%d/%m/%Y %H:%M"), "ID", data.get("patient_id", "N/A")],
    ]
    info_table = Table(summary_rows, colWidths=[3 * cm, 5 * cm, 3 * cm, 5 * cm])
    info_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fafc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(info_table)
    elements.append(Spacer(1, 10))

    if vitals:
        vitals_header = [["Parametre", "Valeur"]]
        vitals_rows = vitals_header + [[k, str(v)] for k, v in vitals.items()]
        vitals_table = Table(vitals_rows, colWidths=[6 * cm, 10 * cm])
        vitals_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), generator.COLORS["secondary"]),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, generator.COLORS["light_bg"]]),
                ]
            )
        )
        elements.append(vitals_table)
        elements.append(Spacer(1, 10))

    elements.append(HRFlowable(width="100%", thickness=0.6, color=generator.COLORS["muted"]))
    elements.append(Spacer(1, 6))

    return elements
