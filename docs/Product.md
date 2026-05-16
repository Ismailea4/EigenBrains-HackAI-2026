**Project name : ShifA'I**

multimodal AI triage agent that empowers rural Moroccan nurses by processing spoken Darija consultations and medical imagery to generate structured diagnostic reports and automatically escalate critical emergencies to urban specialists.


**Context:** My team, EigenBrains, is competing in HackAI 2026. We are building a multimodal medical triage agent designed for under-resourced rural Moroccan dispensaries. The system will assist non-expert nurses by processing patient consultations in Moroccan Darija, generating medical summaries, and analyzing medical imagery, ultimately escalating emergencies to urban doctors. 

**Core Workflow:**
1. **Audio Intake & Transcription:** The system captures the live Darija conversation between the nurse and the patient and transcribes it into text.
2. **Agentic NLP Processing:** An LLM analyzes the transcript to extract symptoms, patient history, and the core medical complaint. We plan to utilize Groq for ultra-fast conversational inference, alongside quantized edge models we've previously finetuned for vital sign interpretation and cardiology monitoring. 
3. **Computer Vision Diagnostic:** If applicable, the nurse uploads a Chest X-ray or an Ultrasound image. A vision model (via Google AI Studio/Gemini or a specialized Hugging Face model) analyzes the image for anomalies.
4. **Artifact Generation:** The agent synthesizes the NLP and Vision data into a structured, highly readable PDF report. 
5. **Emergency Routing:** If the agent's logic detects critical red flags, it automatically flags the PDF as "URGENT" and routes the full context to a specialized doctor in the city, saving them crucial triage time.

**Mandatory Tech Stack (HackAI API Constraints):**
* **LLM Logic & Routing:** Groq API (for fast text inference) and LangGraph/LangChain for the agentic workflow.
* **Multimodal/Vision:** Google AI Studio API (Gemini models) for image analysis and complex reasoning.
* **Model Hub:** Hugging Face (for pulling specialized medical datasets or quantized models).
* **Tracking:** Weights & Biases (wandb) for logging our agent's diagnostic accuracy during the hackathon.

**Task Request:**
Based on this architecture, I need you to help me build the technical blueprint. Please provide:
1.  **The LangGraph/LangChain Node Structure:** How should we define the state and nodes for this specific workflow (e.g., Audio Node -> Extraction Node -> Vision Node -> PDF Node -> Routing Node)?
2.  **Data Schema:** Propose a JSON schema for the data that needs to be extracted from the Darija conversation to populate the final PDF.
3.  **Phase 1 Code:** Write the initial Python boilerplate using LangChain and the Groq API to handle step 2 (extracting structured medical data from a raw transcript).