To integrate **MedGemma** (and its multimodal counterparts) as the backbone of **ShifA'I**, we need a plan that balances its high-fidelity medical reasoning with the low-latency requirements of a live consultation.

Since MedGemma excels at clinical grounding, we will use it for the "Critical Reasoning" and "Visual Analysis" nodes, while leveraging faster edge-quantized models or Groq for the initial Darija-to-English extraction to keep the UI responsive.

---

## ShifA'I Development Roadmap

### Phase 1: Core Orchestration & State Schema

* Define the `TriageState` using `langgraph`.
* Establish the `TypedDict` that tracks patient history, extracted symptoms, vital signs, and image embeddings.
* **Goal:** A functional "skeleton" that can pass data between nodes.

### Phase 2: The "Linguistic Front-End" (Darija Intake)

* Implement Whisper (or a Groq-accelerated Whisper instance) for speech-to-text.
* Develop a lightweight "Translation & Extraction" node to convert Darija input into structured medical concepts for MedGemma to process.

### Phase 3: The "Medical Brain" (MedGemma Integration)

* Set up the **MedGemma** inference pipeline (via Vertex AI or a self-hosted vLLM instance).
* **Node:** `clinical_reasoning_node`. This node takes the structured symptoms and uses MedGemma to identify potential pathologies and "red flags."

### Phase 4: The "Visual Diagnostic" (MedGemma-V / Gemini)

* Integrate the visual component of MedGemma (or Gemini 1.5 Pro for its massive context window in radiology).
* **Node:** `image_analysis_node`. This handles Chest X-ray/Ultrasound interpretation and updates the state with visual findings.

### Phase 5: The "Escalation & Artifact" Engine

* Implement the logic to detect "Emergency" status in the LangGraph state.
* Automated PDF generation and the routing protocol for urban specialist alerts.